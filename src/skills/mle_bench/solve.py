"""MLE-Bench solve -- main entry point for the skill.

Receives a full A2A Message with competition.tar.gz, extracts data,
detects modality, generates and executes an ML script, and returns
submission.csv content.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

from a2a.types import FilePart, FileWithBytes, Message, TextPart

from .audio import generate_audio_script
from .eda import run_eda
from .image import generate_image_script
from .modality import detect_modality
from .runner import EXECUTION_TIMEOUT, PHASE0_TIMEOUT, run_code
from .tabular import generate_tabular_script
from .text import generate_text_script
from .utils import (
    detect_competition_metric,
    extract_tar,
    find_data_dir,
    pre_extract_zips,
    read_description,
)

logger = logging.getLogger(__name__)


async def solve_mle_bench(
    message: Message,
    client,  # AsyncOpenAI
    model: str,
    *,
    on_status=None,
) -> str:
    """Solve an MLE-Bench competition from an A2A message.

    Returns the submission.csv content as a string, or an error message.
    """
    # Extract text instructions and tar bytes from message
    instructions_text = ""
    tar_bytes: bytes | None = None

    if message.parts:
        for part in message.parts:
            root = part.root if hasattr(part, "root") else part
            if isinstance(root, TextPart):
                instructions_text += "\n" + root.text
            elif isinstance(root, FilePart):
                file_data = root.file
                if isinstance(file_data, FileWithBytes):
                    tar_bytes = base64.b64decode(file_data.bytes)

    if not tar_bytes:
        return "ERROR: No competition.tar.gz received in message"

    if on_status:
        await on_status("Extracting competition data...")

    # Extract into temp directory
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        extract_tar(tar_bytes, work_dir)
        data_dir = find_data_dir(work_dir)
        description = read_description(work_dir)

        logger.info("Data directory: %s", data_dir)
        logger.info("Description: %s...", description[:200])

        # Pre-extract zip files for modality detection
        pre_extract_zips(data_dir)

        # Detect modality
        modality = detect_modality(data_dir, description)
        logger.info("Detected modality: %s", modality)

        if on_status:
            await on_status(f"Modality: {modality}. Running pipeline...")

        submission_path = data_dir / "submission.csv"

        # ===== Non-tabular pipelines =====
        if modality == "text":
            script = generate_text_script(data_dir, description)
            if script:
                logger.info("Text script: %d chars", len(script))
                success, output = await run_code(script, data_dir, timeout=EXECUTION_TIMEOUT)
                logger.info("Text pipeline: %s\n%s", "OK" if success else "FAILED", output[-1000:])
                if success and submission_path.exists():
                    return submission_path.read_text()
                # Fall through to tabular
                logger.warning("Text pipeline failed, falling back to tabular")

        elif modality in ("image", "image_regression"):
            # Deterministic image pipeline first
            script = generate_image_script(data_dir, description)
            if script:
                success, output = await run_code(script, data_dir, timeout=EXECUTION_TIMEOUT)
                logger.info("Image pipeline: %s", "OK" if success else "FAILED")
                if success and submission_path.exists():
                    return submission_path.read_text()
            # LLM fallback for image
            script = await _generate_llm_solution(
                data_dir, description, modality, client, model
            )
            if script:
                success, output = await run_code(script, data_dir, timeout=EXECUTION_TIMEOUT)
                if success and submission_path.exists():
                    return submission_path.read_text()
            logger.warning("Image pipeline failed, trying tabular fallback")

        elif modality == "audio":
            # Deterministic audio pipeline first
            script = generate_audio_script(data_dir, description)
            if script:
                success, output = await run_code(script, data_dir, timeout=EXECUTION_TIMEOUT)
                logger.info("Audio pipeline: %s", "OK" if success else "FAILED")
                if success and submission_path.exists():
                    return submission_path.read_text()
            # LLM fallback for audio
            script = await _generate_llm_solution(
                data_dir, description, modality, client, model
            )
            if script:
                success, output = await run_code(script, data_dir, timeout=EXECUTION_TIMEOUT)
                if success and submission_path.exists():
                    return submission_path.read_text()
            logger.warning("Audio pipeline failed, trying tabular fallback")

        # ===== Tabular pipeline (Phase 0: zero LLM calls) =====
        if on_status:
            await on_status("Running EDA...")

        eda_report, eda_meta = run_eda(data_dir)
        logger.info("EDA complete: %d columns profiled", len(eda_meta.get("columns", {})))

        # Detect metric
        comp_scoring, comp_metric = detect_competition_metric(description, eda_meta)
        eda_meta["competition_scoring"] = comp_scoring
        eda_meta["competition_metric_label"] = comp_metric
        logger.info("Metric: %s (sklearn: %s)", comp_metric, comp_scoring)

        # Generate Phase 0 script
        phase0_script = generate_tabular_script(eda_meta, str(data_dir))
        if phase0_script:
            if on_status:
                await on_status("Phase 0: running deterministic solution...")
            logger.info("Phase 0 script: %d chars", len(phase0_script))
            success, output = await run_code(phase0_script, data_dir, timeout=PHASE0_TIMEOUT)
            logger.info("Phase 0: %s\n%s", "OK" if success else "FAILED", output[-1000:])

            if success and submission_path.exists():
                return submission_path.read_text()

            # Phase 0 failed -- try LLM fallback
            logger.warning("Phase 0 failed, trying LLM fallback")

        # ===== LLM fallback =====
        if on_status:
            await on_status("Generating LLM solution...")

        script = await _generate_llm_solution(
            data_dir, description, "tabular", client, model,
            eda_report=eda_report,
        )
        if script:
            success, output = await run_code(script, data_dir, timeout=EXECUTION_TIMEOUT)
            logger.info("LLM solution: %s", "OK" if success else "FAILED")
            if success and submission_path.exists():
                return submission_path.read_text()

        return "ERROR: All pipelines failed to produce submission.csv"


async def _generate_llm_solution(
    data_dir: Path,
    description: str,
    modality: str,
    client,
    model: str,
    *,
    eda_report: str = "",
) -> str | None:
    """Use LLM to generate a solution script as fallback."""
    # List files in data directory
    files_info = []
    for f in sorted(data_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            files_info.append(f"  {f.name} ({size:,} bytes)")

    files_str = "\n".join(files_info[:20])

    prompt = f"""You are solving a Kaggle-style ML competition. Generate a COMPLETE, self-contained Python script.

DATA DIRECTORY: {data_dir}
FILES:
{files_str}

MODALITY: {modality}

DESCRIPTION (first 3000 chars):
{description[:3000]}

{f"EDA REPORT:{chr(10)}{eda_report[:2000]}" if eda_report else ""}

REQUIREMENTS:
- Script must be completely self-contained (all imports at top)
- Load data from DATA_DIR using os.path.join
- Produce submission.csv in DATA_DIR
- Use sklearn, pandas, numpy (always available)
- lightgbm and catboost may be available
- Complete within 600 seconds
- Print progress to stdout with flush=True

Write ONLY the Python script, no explanation."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=4096,
        )
        content = response.choices[0].message.content or ""

        # Extract code from markdown fences if present
        if "```python" in content:
            content = content.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]

        return content.strip()
    except Exception as exc:
        logger.warning("LLM solution generation failed: %s", exc)
        return None
