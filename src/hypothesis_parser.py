"""Parse CyberGym vulnerability descriptions into structured signals.

Pure regex-based extraction -- zero LLM tokens consumed.

Reference: github.com/sharathbaddam/AgentWhetters-cybergym
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VulnSignal:
    """Structured representation of a vulnerability hypothesis."""

    vuln_class: str = "unknown"
    crash_type: str = "unknown"
    vulnerable_function: str = "unknown"
    file_hint: Optional[str] = None
    input_type: str = "binary_file"
    project_domain: str = "unknown"
    stack_trace: List[str] = field(default_factory=list)
    cve_id: Optional[str] = None
    asan_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Vulnerability class patterns (covers all 28 CyberGym types)
# ---------------------------------------------------------------------------

_VULN_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Memory
    ("heap-buffer-overflow", re.compile(r"heap[\-_ ]buffer[\-_ ]overflow", re.I)),
    ("stack-buffer-overflow", re.compile(r"stack[\-_ ]buffer[\-_ ]overflow", re.I)),
    ("global-buffer-overflow", re.compile(r"global[\-_ ]buffer[\-_ ]overflow", re.I)),
    ("heap-buffer-overflow", re.compile(r"heap[\-_ ]overflow", re.I)),
    ("stack-buffer-overflow", re.compile(r"stack[\-_ ]overflow(?!.*recursion)", re.I)),
    ("buffer-overflow", re.compile(r"buffer[\-_ ]overflow", re.I)),
    ("out-of-bounds-read", re.compile(r"out[\-_ ]of[\-_ ]bounds[\-_ ]read|OOB[\-_ ]?read", re.I)),
    ("out-of-bounds-write", re.compile(r"out[\-_ ]of[\-_ ]bounds[\-_ ]write|OOB[\-_ ]?write", re.I)),
    ("out-of-bounds", re.compile(r"out[\-_ ]of[\-_ ]bounds|OOB", re.I)),
    # Pointer
    ("use-after-free", re.compile(r"use[\-_ ]after[\-_ ]free|UAF", re.I)),
    ("double-free", re.compile(r"double[\-_ ]free", re.I)),
    ("null-pointer-dereference", re.compile(r"null[\-_ ](?:ptr|pointer)[\-_ ]deref(?:erence)?|SEGV[\-_ ]on[\-_ ]unknown", re.I)),
    ("wild-pointer", re.compile(r"wild[\-_ ]pointer|invalid[\-_ ]pointer", re.I)),
    # Arithmetic
    ("integer-overflow", re.compile(r"integer[\-_ ]overflow|int[\-_ ]overflow|signed[\-_ ]integer", re.I)),
    ("integer-underflow", re.compile(r"integer[\-_ ]underflow", re.I)),
    ("divide-by-zero", re.compile(r"divide[\-_ ]by[\-_ ]zero|division[\-_ ]by[\-_ ]zero|FPE", re.I)),
    # Logic
    ("assertion-failure", re.compile(r"assertion[\-_ ]fail(?:ure|ed)?|abort|ABRT", re.I)),
    ("stack-overflow-recursion", re.compile(r"stack[\-_ ]overflow.*recursion|infinite[\-_ ]recursion", re.I)),
    ("infinite-loop", re.compile(r"infinite[\-_ ]loop|timeout|hang", re.I)),
    # Concurrency
    ("data-race", re.compile(r"data[\-_ ]race|TSAN|ThreadSanitizer", re.I)),
    ("deadlock", re.compile(r"deadlock", re.I)),
    # Memory safety misc
    ("memory-leak", re.compile(r"memory[\-_ ]leak|LeakSanitizer|LSAN", re.I)),
    ("uninitialized-memory", re.compile(r"uninitialized[\-_ ](?:memory|value|read)|MSAN|MemorySanitizer", re.I)),
    ("type-confusion", re.compile(r"type[\-_ ]confusion|bad[\-_ ]cast|downcast", re.I)),
    ("format-string", re.compile(r"format[\-_ ]string", re.I)),
    # Generic catch-all
    ("overflow", re.compile(r"\boverflow\b", re.I)),
    ("underflow", re.compile(r"\bunderflow\b", re.I)),
    # SEGV as last resort
    ("segmentation-fault", re.compile(r"SEGV|segfault|segmentation[\-_ ]fault", re.I)),
]

# Crash type extraction
_CRASH_TYPE_RE = re.compile(r"(READ|WRITE)\s+of\s+size\s+(\d+)", re.I)

# Function name extraction (order matters -- try most specific first)
_FUNC_PATTERNS = [
    re.compile(r"#0\s+0x[\da-fA-F]+\s+in\s+(\w+)"),
    re.compile(r"in\s+(\w+)\s*\("),
    re.compile(r"in\s+(\w+)\s+at\s+", re.I),
    re.compile(r"in\s+(\w+)\s*$", re.M),
    re.compile(r"in\s+(\w+)\b", re.I),
    re.compile(r"function\s+['\"]?(\w+)['\"]?", re.I),
    re.compile(r"(\w+)\s+at\s+\w+/[\w\.\-]+\.[ch]", re.I),
    re.compile(r"crash\s+in\s+(\w+)", re.I),
    re.compile(r"vulnerable\s+(?:function|code)\s+(?:is\s+)?['\"]?(\w+)['\"]?", re.I),
]

_FUNC_IGNORE = frozenset({
    "in", "at", "the", "a", "an", "is", "on", "of",
    "addresssanitizer", "error", "warning",
})

# File hint, CVE, ASan, stack trace
_FILE_HINT_RE = re.compile(r"([\w\-]+/[\w\-\.]+\.[ch](?:pp|xx)?)")
_CVE_RE = re.compile(r"(CVE-\d{4}-\d+)", re.I)
_ASAN_ERROR_RE = re.compile(r"ERROR:\s*AddressSanitizer:\s*(.+?)(?:\s+on\s+address|\n|$)")
_STACK_TRACE_RE = re.compile(r"#\d+\s+0x[\da-fA-F]+\s+in\s+(\w+)")

# Input type keywords
_INPUT_TYPE_MAP = [
    ("binary_file", re.compile(r"\b(png|jpeg|jpg|gif|tiff|bmp|ico|webp|image|svg)\b", re.I)),
    ("binary_file", re.compile(r"\b(mp3|mp4|ogg|wav|flac|aac|audio|video|avi|mkv)\b", re.I)),
    ("binary_file", re.compile(r"\b(pdf|ttf|otf|woff|font|elf|pe|mach[\-_ ]o)\b", re.I)),
    ("binary_file", re.compile(r"\b(zip|tar|gzip|bzip2|lz4|zstd|xz|lzma|archive|compress)\b", re.I)),
    ("binary_file", re.compile(r"\b(protobuf|capnp|flatbuf|msgpack|cbor|asn1|ber|der)\b", re.I)),
    ("text_file", re.compile(r"(?:^|[\s\-_./])(xml|html|json|yaml|toml|csv|ini|config|text)(?:[\s\-_./]|$)", re.I)),
    ("text_file", re.compile(r"(?:^|[\s\-_./])(regex|url|uri|email|http|ftp|dns|protocol)(?:[\s\-_./]|$)", re.I)),
    ("command_arg", re.compile(r"\b(command|argv|argument|flag|option|cli|param)\b", re.I)),
]

# Project domain keywords
_DOMAIN_MAP = [
    ("image_parser", re.compile(r"\b(png|jpeg|tiff|gif|bmp|webp|ico|image|ImageMagick|graphicsmagick|libpng|libjpeg|openjpeg|mng)\b", re.I)),
    ("audio_video", re.compile(r"\b(mp3|mp4|ogg|wav|flac|ffmpeg|libav|audio|video|avi|mkv)\b", re.I)),
    ("font_parser", re.compile(r"\b(ttf|otf|woff|font|freetype|harfbuzz)\b", re.I)),
    ("compression", re.compile(r"\b(zip|gzip|bzip2|lz4|zstd|xz|lzma|zlib|compress|deflate|archive|tar|libarchive)\b", re.I)),
    ("crypto_lib", re.compile(r"\b(openssl|boringssl|mbedtls|crypto|tls|ssl|certificate|x509)\b", re.I)),
    ("pdf_parser", re.compile(r"\b(pdf|poppler|mupdf|xpdf)\b", re.I)),
    ("network_protocol", re.compile(r"\b(http|ftp|dns|smtp|tcp|udp|protocol|packet)\b", re.I)),
    ("xml_parser", re.compile(r"(?:^|[\s\-_./])(xml|libxml|expat|xerces|html|sax|dom)(?:[\s\-_./]|$)", re.I)),
    ("serialization", re.compile(r"\b(protobuf|json|yaml|toml|msgpack|cbor|flatbuf|capnp)\b", re.I)),
    ("database", re.compile(r"\b(sqlite|mysql|postgres|sql|database|db)\b", re.I)),
    ("regex_engine", re.compile(r"\b(regex|regexp|pcre|re2|oniguruma)\b", re.I)),
]


def parse_hypothesis(description: str) -> VulnSignal:
    """Parse a vulnerability description into a structured VulnSignal."""
    signal = VulnSignal()
    if not description:
        return signal

    for vuln_class, pattern in _VULN_PATTERNS:
        if pattern.search(description):
            signal.vuln_class = vuln_class
            break

    crash_match = _CRASH_TYPE_RE.search(description)
    if crash_match:
        signal.crash_type = f"{crash_match.group(1)} {crash_match.group(2)}"

    for pattern in _FUNC_PATTERNS:
        match = pattern.search(description)
        if match:
            func_name = match.group(1)
            if func_name.lower() not in _FUNC_IGNORE:
                signal.vulnerable_function = func_name
                break

    file_match = _FILE_HINT_RE.search(description)
    if file_match:
        signal.file_hint = file_match.group(1)

    cve_match = _CVE_RE.search(description)
    if cve_match:
        signal.cve_id = cve_match.group(1)

    asan_match = _ASAN_ERROR_RE.search(description)
    if asan_match:
        signal.asan_error = asan_match.group(1).strip()

    signal.stack_trace = _STACK_TRACE_RE.findall(description)

    for input_type, pattern in _INPUT_TYPE_MAP:
        if pattern.search(description):
            signal.input_type = input_type
            break

    for domain, pattern in _DOMAIN_MAP:
        if pattern.search(description):
            signal.project_domain = domain
            break

    return signal
