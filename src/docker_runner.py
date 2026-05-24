"""Docker runner — executes commands and reads files inside eval containers.

Adapted from the SWE-bench Pro agent's DockerRunner. Uses subprocess.run
with `docker exec` rather than docker-py's exec_run, which hangs through
amber's TCP Docker proxy.
"""

from __future__ import annotations

import io
import logging
import subprocess
import tarfile
import time
from dataclasses import dataclass

import docker
from docker.models.containers import Container

logger = logging.getLogger(__name__)

EXEC_TIMEOUT = 120  # seconds per command
EXEC_WALL_GRACE = 30  # extra seconds for subprocess deadline


@dataclass
class ExecResult:
    exit_code: int
    output: str


class DockerRunner:
    """Manage a throwaway container for a single evaluation instance."""

    def __init__(
        self,
        image_uri: str,
        working_dir: str = "/app",
        platform: str | None = None,
    ):
        self._image_uri = image_uri
        self._working_dir = working_dir
        self._platform = platform
        self._client: docker.DockerClient | None = None
        self._container: Container | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Pull the image and start a long-running container."""
        self._client = docker.from_env()
        try:
            pull_kwargs = {"platform": self._platform} if self._platform else {}
            self._client.images.pull(self._image_uri, **pull_kwargs)
        except Exception:
            try:
                self._client.images.get(self._image_uri)
                logger.info("Using locally cached image: %s", self._image_uri)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot pull or find image {self._image_uri}: {exc}"
                ) from exc

        create_kwargs: dict = {
            "detach": True,
            "entrypoint": "/bin/bash",
            "command": ["-c", "tail -f /dev/null"],
            "working_dir": self._working_dir,
        }
        if self._platform:
            create_kwargs["platform"] = self._platform

        self._container = self._client.containers.create(
            self._image_uri, **create_kwargs
        )
        self._container.start()
        logger.info(
            "Started container %s from %s",
            self._container.short_id,
            self._image_uri,
        )

    def stop(self) -> None:
        """Stop and remove the container."""
        if self._container:
            try:
                self._container.stop(timeout=5)
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            logger.info("Removed container %s", self._container.short_id)
            self._container = None

    def cleanup_image(self) -> None:
        """Remove the pulled image to reclaim disk space."""
        if not self._client:
            return
        try:
            self._client.images.remove(self._image_uri, force=True)
            logger.info("Removed image: %s", self._image_uri)
        except Exception as exc:
            logger.warning("Failed to remove image %s: %s", self._image_uri, exc)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def run(self, cmd: str, timeout: int = EXEC_TIMEOUT) -> ExecResult:
        """Execute a shell command inside the container.

        Uses subprocess.run + docker exec instead of docker-py exec_run
        which hangs through amber's TCP Docker proxy.
        """
        if not self._container:
            raise RuntimeError("Container not started")

        cmd_preview = cmd[:120] + ("..." if len(cmd) > 120 else "")
        logger.info("[exec] running (timeout=%ds): %s", timeout, cmd_preview)
        t0 = time.monotonic()

        docker_cmd = [
            "docker", "exec", "-w", self._working_dir, self._container.id,
            "timeout", "-k", "5", f"{timeout}s",
            "bash", "-c", cmd,
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=timeout + EXEC_WALL_GRACE,
            )
            exit_code = result.returncode
            stdout = result.stdout.decode(errors="replace")
            stderr = result.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            logger.error(
                "[exec] subprocess timed out after %.1fs: %s", elapsed, cmd_preview
            )
            return ExecResult(
                exit_code=137,
                output=f"[subprocess timed out after {elapsed:.0f}s]",
            )

        elapsed = time.monotonic() - t0
        logger.info(
            "[exec] done in %.1fs (exit=%s): %s", elapsed, exit_code, cmd_preview
        )

        combined = stdout
        if stderr:
            combined = combined + "\n" + stderr if combined else stderr
        if exit_code in (124, 137):
            note = f"\n[command timed out after {timeout}s]"
            combined = combined + note if combined else note.lstrip("\n")
        return ExecResult(exit_code=exit_code, output=combined)

    def read_file(self, path: str, max_bytes: int = 100_000) -> str:
        """Read a file from the container."""
        result = self.run(f"head -c {max_bytes} {path}")
        if result.exit_code != 0:
            raise FileNotFoundError(f"Cannot read {path}: {result.output}")
        return result.output

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the container using a tar archive."""
        if not self._container:
            raise RuntimeError("Container not started")
        parent = "/".join(path.split("/")[:-1])
        if parent:
            self.run(f"mkdir -p '{parent}'")
        stat_result = self.run(f"stat -c '%a' '{path}' 2>/dev/null")
        mode = (
            int(stat_result.output.strip(), 8)
            if stat_result.exit_code == 0
            else 0o644
        )
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
        tar_buf.seek(0)
        self._container.put_archive(self._working_dir, tar_buf)

    def list_files(self, path: str = ".", max_depth: int = 3) -> str:
        """List directory tree inside the container."""
        result = self.run(f"find {path} -maxdepth {max_depth} -type f | head -500")
        return result.output

    def is_running(self) -> bool:
        """Check if the container is still running."""
        if not self._container:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except Exception:
            return False

    def get_diff(self) -> str:
        """Return the current git diff in the container."""
        if not self._container:
            raise RuntimeError("Container not started")
        if not self.is_running():
            logger.error(
                "Container %s is not running — cannot collect diff",
                self._container.short_id,
            )
            return ""

        _EXCLUDE_DIRS = (
            "appendonlydir", "node_modules", "__pycache__", ".tox",
            ".venv", "venv", ".eggs", "htmlcov",
            ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
        )
        _EXCLUDE_SUBSTR = (".egg-info/",)
        _EXCLUDE_EXTS = (
            "aof", "rdb", "db", "sqlite", "sqlite3",
            "pyc", "pyo", "o", "so", "dylib", "a",
            "class", "jar", "war", "whl", "egg",
            "log", "pid",
            "png", "jpg", "jpeg", "gif", "ico", "svg",
            "zip", "tar", "gz", "bz2", "xz",
        )

        patch_path = "/tmp/_patch.diff"
        dir_pat = "|".join(d.replace(".", r"\.") for d in _EXCLUDE_DIRS)
        ext_pat = "|".join(_EXCLUDE_EXTS)
        substr_pat = "|".join(
            s.replace(".", r"\.") for s in _EXCLUDE_SUBSTR
        )

        self.run(
            f"(git ls-files --others --exclude-standard"
            f" | grep -v -E '(^|/)({dir_pat})(/)'"
            f" | grep -v -E '({substr_pat})'"
            f" | grep -v -E '\\.({ext_pat})$'"
            f" | xargs -r -d '\\n' git add -N -- || true)"
            f" && git diff HEAD -- . > {patch_path}"
            f" ; git reset 2>/dev/null || true"
        )
        bits, _stat = self._container.get_archive(patch_path)
        raw = b"".join(bits)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            member = tar.getmembers()[0]
            data = tar.extractfile(member).read()
        self.run(f"rm -f {patch_path}")
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
