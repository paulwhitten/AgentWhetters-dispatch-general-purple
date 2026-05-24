"""OfficeQA RAG pipeline -- BM25 retrieval over Treasury Bulletin corpus.

Downloads the pre-parsed Treasury Bulletin corpus at startup, chunks
documents into overlapping character windows, builds a BM25 index, and
performs two-pass retrieval to answer questions about U.S. Treasury data.

Approach informed by leaderboard-winning strategies (MIDS4LIFE, AgentSWE):
- Simple character-based chunking aligned to newlines (~275K chunks)
- BM25 lexical search with year-based filtering
- Two-pass retrieval with query refinement from first-draft reasoning
- Footnote/source-note downweighting to prefer data chunks
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import pickle
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openai import AsyncOpenAI

from usage import tracker

logger = logging.getLogger(__name__)

CORPUS_URL = (
    "https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/"
    "treasury_bulletins_parsed/transformed/"
    "treasury_bulletins_transformed.zip"
)
CORPUS_DIR = Path("/tmp/treasury_corpus")
CORPUS_ZIP = CORPUS_DIR / "corpus.zip"
INDEX_PKL = CORPUS_DIR / "index.pkl"

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 500
TOP_K = 30
YEAR_WINDOW = 2

# Footnote/source-note lines that indicate a chunk is metadata, not data.
# Chunks dominated by these patterns get BM25 scores penalized.
_FOOTNOTE_RE = re.compile(
    r"(^Source:|^Note[: ]|^See Table|^\d+/\s|^Revised|^Preliminary|^Estimated|^r/\s)",
    re.MULTILINE,
)

# Chunks that lack table data rows (no | delimiters) and contain footnote
# vocabulary are considered footnote chunks even without explicit markers.
_FOOTNOTE_VOCAB = re.compile(
    r"\b(footnote|formerly|reclassified|revised|effective|"
    r"beginning|prior to|includes?|exclud(?:es?|ing)|"
    r"represents?|comprising|consisted? of)\b",
    re.IGNORECASE,
)

# Detect table/section title lines to propagate into chunks for BM25 matching.
_TABLE_TITLE_RE = re.compile(
    r"^(Table \d|Budget |Federal |Detail |Cash |CASH |Total Cash|"
    r"Expenditures |Receipts )",
)

_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# ── Table metadata ───────────────────────────────────────────────────

# Regex matching the start of a table section (title line).
_TABLE_SECTION_RE = re.compile(
    r"^(Table \d+[A-Za-z]?\.?\-?\s.+|"
    r"CASH INCOME AND OUTGO.+|"
    r"Cash Income and Outgo.+|"
    r"Budget [A-Z].+|"
    r"BUDGET RECEIPTS AND EXPENDITURES|"
    r"Detail of .+|"
    r"Receipts and Expenditures.+|"
    r"Federal .+)",
    re.MULTILINE,
)


@dataclass
class _TableMeta:
    """Metadata for one table section within a bulletin."""

    table_id: str  # "{source}::{offset}" unique key
    title: str
    source: str
    file_year: int
    file_month: int
    units: str
    columns: str  # first header row (truncated)
    char_offset: int
    char_end: int  # start of next table or end of file
    year_min: int
    year_max: int
    table_type: str = ""  # normalized type tag for LLM selection


# Table type patterns: (regex, type_tag) -- first match wins
_TABLE_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"cash\s+(income|operating)\s+(and|&)\s+outgo", re.I), "cash_income_outgo"),
    (re.compile(r"cash\s+outgo|cash\s+income|income\s+and\s+outgo", re.I), "cash_income_outgo"),
    (re.compile(r"national\s+defense", re.I), "national_defense"),
    (re.compile(r"budget\s+receipts?\s+(and|&)\s+expenditures?", re.I), "budget_expenditures"),
    (re.compile(r"expenditures?\s+by\s+major\s+classif", re.I), "budget_expenditures"),
    (re.compile(r"detail\s+of\s+expenditures?\s+by\s+months", re.I), "expenditure_detail_fiscal"),
    (re.compile(r"table\s+6b", re.I), "expenditure_detail_fiscal"),
    (re.compile(r"receipts?\s+by\s+(major\s+)?classif|internal\s+revenue", re.I), "receipts"),
    (re.compile(r"public\s+debt|gross\s+debt|outstanding\s+debt", re.I), "debt"),
    (re.compile(r"federal\s+debt|interest.+debt", re.I), "debt"),
    (re.compile(r"trust\s+fund|social\s+security", re.I), "trust_funds"),
]


def _classify_table_type(title: str) -> str:
    """Classify a table title into a normalized type tag."""
    for pattern, tag in _TABLE_TYPE_PATTERNS:
        if pattern.search(title):
            return tag
    return ""


def _extract_table_metadata(filepath: Path) -> list[_TableMeta]:
    """Scan a document and extract metadata for each table section."""
    m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", filepath.stem)
    file_year = int(m.group(1)) if m else 0
    file_month = int(m.group(2)) if m else 0

    content = filepath.read_text(encoding="utf-8", errors="replace")
    matches = list(_TABLE_SECTION_RE.finditer(content))
    if not matches:
        return []

    tables: list[_TableMeta] = []
    for idx, tm in enumerate(matches):
        title = tm.group(1).strip()[:150]
        start = tm.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        section = content[start:end]

        # Extract units
        units = ""
        units_m = re.search(
            r"\(In (millions|thousands|billions)[^)]*\)", section
        )
        if units_m:
            units = units_m.group(0)

        # Extract column header (first pipe-delimited non-separator line)
        columns = ""
        for ln in section.split("\n"):
            ln_s = ln.strip()
            if (
                "|" in ln_s
                and "---" not in ln_s
                and ln_s.startswith("|")
                and not re.match(r"^\|\s*\d", ln_s)
            ):
                columns = ln_s[:200]
                break

        # Extract year range from data rows
        data_years = {
            int(y)
            for y in re.findall(r"\b(19\d{2}|20\d{2})\b", section)
            if 1900 <= int(y) <= 2030
        }
        year_min = min(data_years) if data_years else file_year
        year_max = max(data_years) if data_years else file_year

        table_id = f"{filepath.name}::{start}"
        tables.append(
            _TableMeta(
                table_id=table_id,
                title=title,
                source=filepath.name,
                file_year=file_year,
                file_month=file_month,
                units=units,
                columns=columns,
                char_offset=start,
                char_end=end,
                year_min=year_min,
                year_max=year_max,
                table_type=_classify_table_type(title),
            )
        )
    return tables


# ── Lightweight BM25 (avoids external dependency) ────────────────────


class _BM25:
    """BM25-Okapi scorer using numpy-free pure Python."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_lens = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 1.0
        self.corpus = corpus

        # Document frequency
        df: Counter[str] = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1

        # Precompute IDF
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1
            )

        # Precompute term frequencies per document
        self.tf: list[Counter[str]] = [Counter(doc) for doc in corpus]

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.N):
                tf = self.tf[i].get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                numer = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * numer / denom
        return scores


# ── Corpus index ─────────────────────────────────────────────────────

_index: _CorpusIndex | None = None
_index_lock = asyncio.Lock()


class _CorpusIndex:
    """BM25 index over chunked Treasury Bulletin documents."""

    def __init__(
        self,
        chunks: list[dict],
        bm25: _BM25,
        table_catalog: list[_TableMeta] | None = None,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self.table_catalog: list[_TableMeta] = table_catalog or []
        self._year_to_indices: dict[int, list[int]] = {}
        # Build source-ordered index for context windowing
        self._source_indices: dict[str, list[int]] = {}
        # Build table_id → chunk indices mapping
        self._table_to_indices: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            for year in chunk.get("years", []):
                self._year_to_indices.setdefault(year, []).append(i)
            src = chunk["source"]
            self._source_indices.setdefault(src, []).append(i)
            tid = chunk.get("table_id", "")
            if tid:
                self._table_to_indices.setdefault(tid, []).append(i)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        years: list[int] | None = None,
    ) -> list[dict]:
        """Retrieve top-k chunks, optionally filtered by year window."""
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        if years:
            year_set: set[int] = set()
            for y in years:
                for offset in range(-YEAR_WINDOW, YEAR_WINDOW + 1):
                    year_set.add(y + offset)
            valid = set()
            for y in year_set:
                valid.update(self._year_to_indices.get(y, []))
            for i in range(len(scores)):
                if i not in valid:
                    scores[i] = 0.0

        # Downweight footnote/source-note chunks that repeat topic keywords
        # but lack actual data values
        for i in range(len(scores)):
            if scores[i] > 0 and self.chunks[i].get("is_footnote"):
                scores[i] *= 0.5

        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.chunks[i] for i in ranked if scores[i] > 0]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = TOP_K,
        years: list[int] | None = None,
        context_window: int = 1,
    ) -> list[dict]:
        """Retrieve top-k chunks plus adjacent chunks from same file.

        For each retrieved chunk, also includes the *context_window* chunks
        before and after it from the same source file. This ensures table
        headers (which may be in a previous chunk) travel with data rows.
        """
        core = self.retrieve(query, top_k=top_k, years=years)
        if not context_window:
            return core

        seen: set[int] = set()
        expanded: list[dict] = []
        for chunk in core:
            src = chunk["source"]
            siblings = self._source_indices.get(src, [])
            # Find this chunk's position among its siblings
            chunk_id = id(chunk)
            pos = None
            for j, si in enumerate(siblings):
                if id(self.chunks[si]) == chunk_id:
                    pos = j
                    break
            if pos is None:
                if id(chunk) not in seen:
                    seen.add(id(chunk))
                    expanded.append(chunk)
                continue

            # Add context window: chunks before and after
            for offset in range(-context_window, context_window + 1):
                idx = pos + offset
                if 0 <= idx < len(siblings):
                    global_idx = siblings[idx]
                    c = self.chunks[global_idx]
                    cid = id(c)
                    if cid not in seen:
                        seen.add(cid)
                        expanded.append(c)
        return expanded

    def retrieve_by_tables(
        self,
        query: str,
        table_ids: list[str],
        top_k: int = TOP_K,
        years: list[int] | None = None,
    ) -> list[dict]:
        """Retrieve top-k chunks restricted to specific table sections.

        This is the metadata-guided retrieval path: the LLM picks table_ids
        from the catalog, and we only score chunks belonging to those tables.
        """
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Restrict to chunks from the specified tables
        valid_indices: set[int] = set()
        for tid in table_ids:
            valid_indices.update(self._table_to_indices.get(tid, []))

        # Apply year filter if provided
        if years:
            year_set: set[int] = set()
            for y in years:
                for offset in range(-YEAR_WINDOW, YEAR_WINDOW + 1):
                    year_set.add(y + offset)
            year_valid: set[int] = set()
            for y in year_set:
                year_valid.update(self._year_to_indices.get(y, []))
            valid_indices &= year_valid

        for i in range(len(scores)):
            if i not in valid_indices:
                scores[i] = 0.0
            elif scores[i] > 0 and self.chunks[i].get("is_footnote"):
                scores[i] *= 0.5

        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.chunks[i] for i in ranked if scores[i] > 0]

    def catalog_for_years(self, years: list[int]) -> list[_TableMeta]:
        """Return catalog entries whose year range overlaps the query years."""
        if not years:
            return self.table_catalog
        result = []
        for tm in self.table_catalog:
            for y in years:
                if tm.year_min - 2 <= y <= tm.year_max + 2:
                    result.append(tm)
                    break
        return result

    def catalog_for_query(
        self, query: str, years: list[int],
    ) -> list[tuple[int, _TableMeta]]:
        """Return catalog entries matching years AND query keywords.

        Uses TF-IDF-weighted keyword matching on table title + columns to
        pre-filter the catalog to a manageable size for LLM selection.
        Returns (relevance_score_0_100, meta) tuples sorted by score.
        """
        candidates = self.catalog_for_years(years)
        if not candidates:
            return []

        query_tokens = _tokenize(query)
        query_token_set = set(query_tokens)

        # Build document frequency across candidate titles
        doc_count = len(candidates)
        df: Counter[str] = Counter()
        for tm in candidates:
            searchable = f"{tm.title} {tm.columns} {tm.units}".lower()
            for tok in set(re.findall(r"\w+", searchable)):
                df[tok] += 1

        # Compute TF-IDF score per candidate
        scored: list[tuple[float, _TableMeta]] = []
        max_score = 0.0
        # Ideal publication date: 1-2 months after end of target year
        ideal_months = []
        for y in (years if years else [1950]):
            ideal_months.append((y + 1) * 12 + 2)
        for tm in candidates:
            searchable = f"{tm.title} {tm.columns} {tm.units}".lower()
            searchable_tokens = re.findall(r"\w+", searchable)
            searchable_set = set(searchable_tokens)
            overlap = query_token_set & searchable_set
            if not overlap:
                continue
            # TF-IDF: sum of idf for each matching token, weighted by
            # position (title matches worth more than column matches)
            title_tokens = set(re.findall(r"\w+", tm.title.lower()))
            score = 0.0
            for tok in overlap:
                idf = math.log((doc_count + 1) / (df.get(tok, 0) + 1)) + 1
                # Title match bonus: 2x weight for tokens in the title
                weight = 2.0 if tok in title_tokens else 1.0
                score += idf * weight
            # Publication-year proximity boost: bulletins published near the
            # target year get up to 50% bonus; distant ones get no bonus.
            # This prevents 1947 bulletins from crowding out 1954 ones when
            # both have the same keyword relevance for a 1953 question.
            pub_month = tm.file_year * 12 + tm.file_month
            min_dist = min(abs(pub_month - im) for im in ideal_months)
            # Decay: 1.0 at distance 0, ~0.5 at 24 months, ~0.25 at 48
            proximity = 1.0 / (1.0 + min_dist / 24.0)
            score *= (1.0 + 0.5 * proximity)
            scored.append((score, tm))
            if score > max_score:
                max_score = score

        # Normalize scores to 0-100 and filter
        if max_score > 0:
            scored = [(s / max_score * 100, tm) for s, tm in scored]

        # Sort by score descending, cap at 50 entries
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(int(round(s)), tm) for s, tm in scored[:50]]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ── Corpus download and chunking ─────────────────────────────────────


async def _ensure_index() -> _CorpusIndex:
    global _index
    if _index is not None:
        return _index
    async with _index_lock:
        if _index is not None:
            return _index

        # Fast path: load pre-built index from pickle
        if INDEX_PKL.exists():
            logger.info("Loading pre-built index from %s...", INDEX_PKL)
            import gzip as _gzip
            with open(INDEX_PKL, "rb") as f:
                data = pickle.loads(_gzip.decompress(f.read()))
            _index = _CorpusIndex(
                data["chunks"], data["bm25"],
                table_catalog=data["table_catalog"],
            )
            # Backfill table_type for indexes built before classification
            for tm in _index.table_catalog:
                if not tm.table_type:
                    tm.table_type = _classify_table_type(tm.title)
            logger.info(
                "Index loaded: %d chunks, %d table sections",
                len(_index.chunks), len(_index.table_catalog),
            )
            return _index

        # Slow path: download, chunk, and build from scratch
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        texts_dir = CORPUS_DIR / "texts"

        if not texts_dir.exists() or not any(texts_dir.glob("*.txt")):
            logger.info("Downloading Treasury Bulletin corpus (%s)...", CORPUS_URL)
            proc = await asyncio.create_subprocess_exec(
                "curl", "-sL", CORPUS_URL, "-o", str(CORPUS_ZIP),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            if not CORPUS_ZIP.exists() or CORPUS_ZIP.stat().st_size < 1_000_000:
                raise RuntimeError("Corpus download failed")

            logger.info("Extracting corpus...")
            texts_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(CORPUS_ZIP) as zf:
                zf.extractall(texts_dir)
            CORPUS_ZIP.unlink(missing_ok=True)

        logger.info("Chunking documents...")
        txt_files = sorted(texts_dir.glob("*.txt"))

        chunks: list[dict] = []
        all_table_metas: list[_TableMeta] = []
        for txt_file in txt_files:
            file_metas = _extract_table_metadata(txt_file)
            all_table_metas.extend(file_metas)
            chunks.extend(_chunk_document(txt_file, file_metas))

        logger.info("Building BM25 index over %d chunks...", len(chunks))
        tokenized = [_tokenize(c["text"]) for c in chunks]
        bm25 = _BM25(tokenized)

        _index = _CorpusIndex(chunks, bm25, table_catalog=all_table_metas)
        logger.info(
            "Index ready: %d chunks from %d documents, %d table sections",
            len(chunks),
            len(list(texts_dir.glob("*.txt"))),
            len(all_table_metas),
        )
        return _index


def _chunk_document(
    filepath: Path,
    table_metas: list[_TableMeta] | None = None,
) -> list[dict]:
    """Split a document into overlapping character chunks, aligned to newlines.

    Each chunk is prefixed with the last-seen table/section title so that
    BM25 can match query terms (e.g. "national defense expenditures") even
    in data-only chunks that only contain numeric table rows. Chunks snap to
    newline boundaries to keep table rows intact.

    When *table_metas* is provided, each chunk is tagged with the table_id of
    the table section it falls within.
    """
    match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", filepath.stem)
    file_year = int(match.group(1)) if match else 0

    content = filepath.read_text(encoding="utf-8", errors="replace")

    # Pre-scan: build a map of character offset → active table title.
    # We track the last title seen before each position in the file.
    titles_at: list[tuple[int, str]] = []  # (offset, title)
    for line_match in re.finditer(r"^(.+)$", content, re.MULTILINE):
        line = line_match.group(1).strip()
        if _TABLE_TITLE_RE.match(line) and len(line) < 200:
            titles_at.append((line_match.start(), line))

    def _title_for_offset(offset: int) -> str:
        """Return the most recent table title at or before this offset."""
        title = ""
        for t_off, t_text in titles_at:
            if t_off <= offset:
                title = t_text
            else:
                break
        return title

    def _table_id_for_offset(offset: int) -> str:
        """Return the table_id for the table section containing this offset."""
        if not table_metas:
            return ""
        for tm in reversed(table_metas):
            if tm.char_offset <= offset < tm.char_end:
                return tm.table_id
        return ""

    chunks: list[dict] = []
    i = 0
    n = len(content)
    while i < n:
        end = min(i + CHUNK_SIZE, n)
        # Snap end to a newline to keep table rows intact
        if end < n:
            nl = content.rfind("\n", i + CHUNK_SIZE // 2, end)
            if nl != -1:
                end = nl
        piece = content[i:end].strip()
        if piece:
            # Prepend the active table/section title if not already in chunk
            title = _title_for_offset(i)
            if title and title not in piece:
                indexed_text = title + "\n" + piece
            else:
                indexed_text = piece

            years_in_chunk = {
                int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", piece)
            }
            if file_year:
                years_in_chunk.add(file_year)
            # Detect footnote-heavy chunks: chunks with multiple explicit
            # footnote markers (Source:, Note:, numbered footnotes like 1/)
            footnote_lines = len(_FOOTNOTE_RE.findall(piece))
            total_lines = piece.count("\n") + 1
            footnote_vocab_hits = len(_FOOTNOTE_VOCAB.findall(piece))
            is_footnote = (
                footnote_lines >= 3
                or (total_lines > 0 and footnote_lines / total_lines > 0.4)
                or (footnote_vocab_hits >= 5 and footnote_lines >= 1)
            )
            chunks.append({
                "text": indexed_text,
                "source": filepath.name,
                "years": sorted(years_in_chunk),
                "is_footnote": is_footnote,
                "table_id": _table_id_for_offset(i),
            })
        if end >= n:
            break
        i = max(end - CHUNK_OVERLAP, i + 1)

    return chunks


# ── Year extraction ──────────────────────────────────────────────────


def _extract_years(question: str) -> list[int]:
    return sorted(set(int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)))


# ── Answer validation ────────────────────────────────────────────────

_INVALID_ANSWERS = frozenset({
    "nan", "none", "inf", "-inf", "null", "", "undefined",
    "na", "n/a", "error",
})


def _is_invalid_answer(answer: str) -> bool:
    """Check if a code-gen answer is degenerate (nan, inf, empty, etc.)."""
    normalized = answer.strip().lower().rstrip("%")
    if normalized in _INVALID_ANSWERS:
        return True
    # Check for float nan/inf
    try:
        val = float(normalized.replace(",", ""))
        if math.isnan(val) or math.isinf(val):
            return True
    except (ValueError, OverflowError):
        pass
    return False


# ── Deterministic table selection ────────────────────────────────────

# Maps question keywords to table types. Order matters: first match wins.
_QUESTION_TYPE_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"national\s+defense|defense\s+expenditures?|defense\s+and\s+(related|associated)", re.I),
     ["national_defense", "budget_expenditures", "cash_income_outgo"]),
    (re.compile(r"cash\s+(income|outgo|operating)|cash\s+disbursement", re.I),
     ["cash_income_outgo"]),
    (re.compile(r"expenditures?\s+by\s+major|major\s+classif", re.I),
     ["budget_expenditures"]),
    (re.compile(r"budget\s+(receipts?|expenditures?)", re.I),
     ["budget_expenditures"]),
    (re.compile(r"veterans?\s+admin|post\s+office|agriculture|interior|treasury", re.I),
     ["budget_expenditures", "expenditure_detail_fiscal"]),
    (re.compile(r"receipts?\s+by|internal\s+revenue|income\s+tax", re.I),
     ["receipts"]),
    (re.compile(r"public\s+debt|gross\s+debt|outstanding\s+debt|federal\s+debt", re.I),
     ["debt"]),
    (re.compile(r"trust\s+fund|social\s+security", re.I),
     ["trust_funds"]),
    (re.compile(r"expenditures?|outlay|spending", re.I),
     ["budget_expenditures", "national_defense", "cash_income_outgo"]),
]


def _question_to_table_types(question: str) -> list[str]:
    """Deterministically map a question to candidate table types."""
    for pattern, types in _QUESTION_TYPE_PATTERNS:
        if pattern.search(question):
            return types
    return []


def _validate_year_in_table(
    meta: _TableMeta,
    year: int,
    corpus_dir: Path = CORPUS_DIR,
) -> bool:
    """Verify that a table section actually contains data rows for *year*.

    Reads the raw section text and checks that a pipe-delimited data row
    (not a footnote or header) contains the year as a standalone token.
    """
    filepath = corpus_dir / "texts" / meta.source
    if not filepath.exists():
        return False
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    section = content[meta.char_offset : meta.char_end]
    year_str = str(year)
    # Check column headers for year (multi-column tables like Cash Outgo)
    # and data rows for year (row-based tables like Table 2)
    for line in section.split("\n"):
        ls = line.strip()
        if not ls.startswith("|"):
            continue
        # Skip separator rows
        if re.match(r"^\|\s*-+", ls):
            continue
        if year_str in ls:
            return True
    return False


def _deterministic_table_select(
    question: str,
    years: list[int],
    catalog: list[_TableMeta],
) -> list[str]:
    """Select table IDs deterministically based on question keywords, year
    proximity, and data-presence validation.

    Returns a list of table_ids or an empty list if no confident match.
    """
    table_types = _question_to_table_types(question)
    if not table_types or not years:
        return []

    # Group catalog by (table_type, title_normalized) for deduplication
    # across monthly editions of the same bulletin table.
    from collections import defaultdict
    type_candidates: dict[str, list[_TableMeta]] = defaultdict(list)
    for tm in catalog:
        if tm.table_type in table_types:
            type_candidates[tm.table_type].append(tm)

    if not type_candidates:
        return []

    selected: list[str] = []
    seen_ids: set[str] = set()

    for year in years:
        # For each year, find the best bulletin per matching table type.
        # Ideal: published 1-3 months after the year ends (complete data).
        ideal_pub = (year + 1) * 12 + 2  # Feb of year+1
        for ttype in table_types:
            candidates = type_candidates.get(ttype, [])
            if not candidates:
                continue
            # Score by publication proximity, prefer within [year, year+2]
            scored = []
            for tm in candidates:
                pub_month = tm.file_year * 12 + tm.file_month
                dist = abs(pub_month - ideal_pub)
                # Strong preference for bulletins published after the year
                if tm.file_year < year:
                    dist += 24  # penalty for pre-year bulletins
                scored.append((dist, tm))
            scored.sort(key=lambda x: x[0])

            # Try candidates in order until one passes validation
            for _, tm in scored[:8]:
                if tm.table_id in seen_ids:
                    continue
                if _validate_year_in_table(tm, year):
                    selected.append(tm.table_id)
                    seen_ids.add(tm.table_id)
                    logger.info(
                        "Deterministic select: year=%d type=%s -> %s (%s)",
                        year, ttype, tm.table_id, tm.source,
                    )
                    break

    return selected


# ── Raw table extraction ─────────────────────────────────────────────


def _extract_raw_tables(
    chunks: list[dict],
    index: "_CorpusIndex",
    max_tables: int = 8,
    max_chars: int = 60_000,
) -> str:
    """Extract raw markdown tables from the corpus for the retrieved chunks.

    Walks back from each chunk to the table section it belongs to,
    extracts the full markdown table text (header + data rows), and
    returns them formatted for the code-gen LLM.

    Tables are ranked by how many retrieved chunks reference them (more
    chunks = more relevant).
    """
    # Count chunk hits per table_id to rank by relevance
    table_hit_count: dict[str, int] = {}
    for chunk in chunks:
        tid = chunk.get("table_id", "")
        if tid:
            table_hit_count[tid] = table_hit_count.get(tid, 0) + 1

    # Sort by hit count descending
    ranked_ids = sorted(table_hit_count, key=table_hit_count.get, reverse=True)

    # Build a lookup for table catalog
    meta_by_id = {tm.table_id: tm for tm in index.table_catalog}

    seen_table_ids: set[str] = set()
    tables: list[str] = []
    total_chars = 0

    for table_id in ranked_ids:
        if table_id in seen_table_ids:
            continue
        seen_table_ids.add(table_id)

        meta = meta_by_id.get(table_id)
        if not meta:
            continue

        # Read the raw section from disk
        filepath = CORPUS_DIR / "texts" / meta.source
        if not filepath.exists():
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        section = content[meta.char_offset : meta.char_end]

        # Extract title, units, and markdown table rows
        lines = section.split("\n")
        title_line = lines[0].strip() if lines else meta.title
        units_line = ""
        table_lines: list[str] = []
        for ln in lines:
            ls = ln.strip()
            if ls.startswith("(In ") and ")" in ls:
                units_line = ls
            elif ls.startswith("|"):
                table_lines.append(ls)

        if not table_lines:
            continue

        # Cap individual table: keep header + separator + first 300 data rows.
        # Use a generous cap so multi-year monthly tables aren't truncated
        # (a 15-year monthly table has ~180 rows). The total char budget
        # (max_chars) still guards overall context size.
        table_text = "\n".join(table_lines)
        if len(table_lines) > 302:  # header + sep + 300 data
            table_text = "\n".join(table_lines[:302])

        entry = (
            f"### TABLE {len(tables) + 1}: {title_line}\n"
            f"Source: {meta.source}\n"
        )
        if units_line:
            entry += f"Units: {units_line}\n"
        entry += f"\n{table_text}\n"

        if total_chars + len(entry) > max_chars:
            break
        tables.append(entry)
        total_chars += len(entry)
        if len(tables) >= max_tables:
            break

    return "\n\n".join(tables)


# ── Code execution sandbox ───────────────────────────────────────────


def _exec_pandas_code(code: str, tables_text: str) -> str:
    """Execute LLM-generated pandas code in a restricted namespace.

    The code receives a ``parse_table(table_number)`` helper that parses
    the Nth markdown table (1-indexed) from the retrieved tables into a
    pandas DataFrame.  It must ``print()`` its final answer.
    """
    import pandas as pd  # noqa: F811 -- intentional late import

    # Pre-parse all markdown tables from the tables_text
    raw_tables: list[list[str]] = []  # each is list of pipe-delimited lines
    current: list[str] = []
    for line in tables_text.split("\n"):
        ls = line.strip()
        if ls.startswith("|"):
            current.append(ls)
        else:
            if current:
                raw_tables.append(current)
                current = []
    if current:
        raw_tables.append(current)

    def parse_table(table_number: int) -> pd.DataFrame:
        """Parse markdown table *table_number* (1-indexed) into a DataFrame."""
        idx = table_number - 1
        if idx < 0 or idx >= len(raw_tables):
            raise ValueError(
                f"Table {table_number} not found (have {len(raw_tables)} tables)"
            )
        lines = raw_tables[idx]
        # Remove separator rows
        lines = [l for l in lines if not re.match(r"^\|\s*-+", l)]
        if not lines:
            raise ValueError(f"Table {table_number} has no data rows")

        # Parse header
        cols = [c.strip() for c in lines[0].split("|")[1:-1]]
        data_rows = []
        for line in lines[1:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # Pad or truncate to match header length
            while len(cells) < len(cols):
                cells.append("")
            data_rows.append(cells[: len(cols)])

        df = pd.DataFrame(data_rows, columns=cols)

        # Clean numeric columns: remove footnote markers (14/, 15/),
        # commas, leading/trailing whitespace
        for c in df.columns[1:]:
            cleaned = (
                df[c]
                .astype(str)
                .str.replace(r"\s*\d+/", "", regex=True)
                .str.replace(",", "")
                .str.strip()
            )
            cleaned = cleaned.replace(
                {"nan": None, "": None, "-": None, "*": None, "...": None}
            )
            df[c] = pd.to_numeric(cleaned, errors="coerce")

        return df

    # Build restricted namespace
    import io as _io
    import math as _math
    import statistics as _stat

    output_buf = _io.StringIO()

    # Safe import that only allows pre-approved modules
    _allowed_modules = {"re": re, "math": _math, "statistics": _stat, "pd": pd, "pandas": pd}
    def _safe_import(name, *args, **kwargs):
        if name in _allowed_modules:
            return _allowed_modules[name]
        raise ImportError(f"Module '{name}' is not available. Use pre-imported: re, pd, math, statistics")

    namespace = {
        "parse_table": parse_table,
        "pd": pd,
        "re": re,
        "math": _math,
        "statistics": _stat,
        "print": lambda *args, **kw: output_buf.write(
            " ".join(str(a) for a in args) + kw.get("end", "\n")
        ),
        "__builtins__": {
            "__import__": _safe_import,
            "abs": abs,
            "all": all,
            "any": any,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ZeroDivisionError": ZeroDivisionError,
            "RuntimeError": RuntimeError,
            "StopIteration": StopIteration,
        },
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        return f"EXEC_ERROR: {type(exc).__name__}: {exc}"

    result = output_buf.getvalue().strip()
    if not result:
        return "EXEC_ERROR: Code produced no output (must use print())"
    # Normalize: strip trailing .0 from integer results (e.g. '2602.0' -> '2602')
    try:
        fval = float(result.replace(",", ""))
        if fval == int(fval) and "%" not in result:
            result = str(int(fval))
    except (ValueError, OverflowError):
        pass
    return result


# ── LLM prompts ──────────────────────────────────────────────────────

CODEGEN_PROMPT = """\
You are an expert data analyst. Write Python/pandas code to answer a question \
about U.S. Treasury financial data.

You have access to these tools:
- ``parse_table(n)`` -- parses the Nth table (1-indexed) from the data below \
into a pandas DataFrame. The first column is typically the row label (category \
or date). Remaining columns are numeric (already cleaned: NaN for missing values).
- ``pd`` (pandas), ``re``, ``math``, ``statistics`` are available.
- ``print()`` your final answer as a single value or list.

IMPORTANT RULES:
1. Column headers may use multi-level format with ``>`` separator: \
e.g., ``1939 > Dec.`` means the column is under the ``1939`` group.
2. **Fiscal year column groupings**: In tables with monthly columns, the year \
in the header may be the FISCAL YEAR, not calendar year. Before 1977, the US \
fiscal year ran July-June. Example: ``1939 > Jan.`` actually means January \
1940 (it falls in the fiscal year that started July 1939). The column \
sequence ``1939 > Dec. | 1939 > Jan. | 1939 > Feb.`` represents December 1939, \
January 1940, February 1940 chronologically.
3. To get calendar year 1940 data from fiscal-year-grouped columns, you need \
columns where the ACTUAL MONTH is in 1940: that includes ``1939 > Jan.`` \
through ``1939 > June`` (Jan-June 1940 in FY1939-40) AND ``1940 > July`` \
through ``1940 > Dec.`` (July-Dec 1940).
4. For row-based month tables (e.g., rows like ``1953-Jan.``, ``Feb.``, etc.), \
months are already in calendar order.
5. Remove footnote markers (e.g., ``14/``, ``15/``) before numeric conversion \
-- ``parse_table()`` handles this automatically.
6. When computing totals of individual monthly values, sum the 12 individual \
month rows/columns. Do NOT use a pre-printed annual total row.
7. ``print()`` exactly one final numeric result (or list if multiple values \
are requested). No extra text.
8. Handle ``nan`` values with ``pd.notna()`` or ``.dropna()``.
9. When multiple tables cover the same topic, prefer "Table 2" (broad \
category totals like "Expenditures by Major Classifications") over "Table 3" \
(detailed sub-breakdowns like "Expenditures for National Defense and Related \
Activities"). Table 2 has the authoritative "National defense and related \
activities" column; Table 3 lists sub-items whose "Total" column does NOT \
equal Table 2's aggregate value. For questions about "national defense" \
totals, always use the "National defense and related activities" column from \
the "Major Classifications" table (Table 2), NOT Table 3's "Total".
10. For "percent difference" or "percent change" questions: compute each \
year's value separately, then calculate \
``pct = abs(value_B - value_A) / value_A * 100``. Print the result with \
``print(f"{{pct:.2f}}")`` (number only, no % sign). Do NOT confuse this with \
absolute difference (which omits the division by value_A).
11. Known CPI-U annual averages (BLS, Federal Reserve Bank of Minneapolis, \
not seasonally adjusted): 1913=9.9, 1920=20.0, 1930=16.7, 1934=13.4, \
1940=14.0, 1945=18.0, 1950=24.1, 1953=26.8, 1955=26.8, 1960=29.6, \
1965=31.5, 1970=38.8, 1975=53.8, 1980=82.4, 1985=107.6, 1990=130.7, \
1995=152.4, 2000=172.2, 2005=195.3, 2010=218.1, 2015=237.0, 2020=258.8, \
2025=321.9. For CPI-adjusted calculations, multiply by \
``cpi_target / cpi_base``.

TABLES:
{tables}

QUESTION: {question}

Write ONLY executable Python code. No markdown fences. No explanation.
"""

SYSTEM_PROMPT = """\
You are an expert financial analyst specializing in U.S. Treasury data.
Answer the question using ONLY the provided Treasury Bulletin excerpts.

RULES:
- Extract exact values from the provided excerpts. Do NOT guess or estimate.
- Pay close attention to units (millions, billions, thousands) in table headers.
- Distinguish fiscal year (July-June before 1977, Oct-Sep after) vs calendar year.
- If a table header says "In thousands of dollars", divide values by 1,000 to \
get millions.
- Use exact arithmetic. Show your calculations step by step.
- When the question asks for "sum of individual months" or a "calendar year total", \
compute the sum of the 12 individual monthly values (Jan-Dec) yourself rather \
than using a pre-printed annual/calendar-year row total. Published totals may \
differ from the arithmetic sum.
- Always show the individual monthly values and your arithmetic in the reasoning.
- You MUST provide a numeric answer. NEVER answer "data not available".

TABLE SELECTION (CRITICAL -- follow this strictly):
- Treasury Bulletins contain multiple tables under different accounting bases.
- **DO NOT use Table 6B** ("Detail of Expenditures by Months and Years") for \
calendar year questions. Table 6B organizes data by FISCAL YEAR periods \
(Jul-Jun). Its monthly columns do NOT correspond to calendar year months \
(Jan-Dec). Values from Table 6B will give WRONG calendar year totals.
- **For calendar year expenditures**: Use "Cash Income and Outgo of the Treasury" \
(Cash Outgo) which shows actual cash disbursements labeled by calendar month \
and year. This is the PREFERRED source for calendar year questions.
- **"Table 2 - Expenditures by Major Classifications"** lists months individually \
as rows (e.g. "1953-Jan.", "Feb.", "Mar." etc.) and is authoritative for broad \
category totals like "National defense and related activities".
- **Table 3** ("Expenditures for National Defense and Related Activities") shows \
a detailed sub-breakdown. When both Table 2 and Table 3 are available, prefer \
Table 2 for the overall "national defense" total.
- When a Cash Outgo table has 13 monthly columns, the first column is typically \
the last month of the PRIOR year. For calendar year totals, sum only the 12 \
columns corresponding to Jan-Dec of the target year (skip the prior year column).

REQUIRED OUTPUT FORMAT:
<REASONING>
[Show the specific excerpts used, values extracted, and calculations]
</REASONING>
<FINAL_ANSWER>
[Your precise numerical or text answer -- just the value, no extra text]
</FINAL_ANSWER>
"""

ANALYZE_PROMPT = """\
Extract search information from this U.S. Treasury data question.

Question: {question}

Treasury Bulletins use different terms for similar concepts:
- "expenditures" may appear as "cash outgo", "budget outgo", or "outlays"
- Data appears in "Table 3" (general), "Table 6B" (detail by month), or \
"Cash Income and Outgo" tables

Return ONLY a JSON object (no markdown, no explanation):
{{"search_queries": [\
"primary keyword-rich query", \
"alternative query using different Treasury vocabulary (e.g. cash outgo vs expenditures)"\
], \
"years": [list of integer years], \
"table_hint": "likely table name e.g. Table 3, Table 5"}}
"""

TABLE_SELECT_PROMPT = """\
You are selecting the best Treasury Bulletin tables to answer a question.

Question: {question}

Below is a catalog of available table sections from U.S. Treasury Bulletins, \
ordered by relevance score (0-100). Each entry shows a relevance score, \
table type, table ID, title, source bulletin, units, column headers, \
and year range.

SELECTION STRATEGY (follow these steps in order):
1. Identify what TYPE of data the question needs: cash_income_outgo, \
budget_expenditures, national_defense, debt, receipts, or other.
2. Among matching types, prefer tables with HIGHER relevance scores.
3. Prefer bulletins published 1-2 months AFTER the target year \
(they contain complete calendar year data).
4. For CALENDAR YEAR expenditure questions, prefer "Cash Income and Outgo" \
or "Table 2 - Expenditures by Major Classifications" (monthly rows like \
"1953-Jan."). DO NOT select "Table 6B" or "Detail of Expenditures by Months \
and Years" -- those use FISCAL YEAR periods.
5. If the question references MULTIPLE YEARS (e.g. 1940 and 1953), you MUST \
select tables from bulletins near EACH year. For 1940 data select from 1941 \
bulletins; for 1953 data select from 1954 bulletins.
6. Select 3-8 tables. When uncertain between two tables of the same type, \
pick the one with the higher relevance score.

CATALOG:
{catalog}

Return a JSON object with a single key "table_ids" containing an array of \
table ID strings. Example: {{"table_ids": ["id1", "id2"]}}
"""

# JSON schema for structured table selection output
_TABLE_SELECT_SCHEMA = {
    "type": "json_schema",
    "name": "table_selection",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "table_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of selected table IDs from the catalog",
            }
        },
        "required": ["table_ids"],
        "additionalProperties": False,
    },
}


# ── Main pipeline ────────────────────────────────────────────────────


async def solve_officeqa(
    input_text: str,
    client: AsyncOpenAI,
    model: str,
    cheap_model: str = "gpt-4o-mini",
    on_status: Callable | None = None,
) -> str:
    """Answer an OfficeQA question via code-gen over retrieved tables."""

    # 1 ── Ensure index
    if on_status:
        await on_status("Loading Treasury Bulletin corpus...")
    index = await _ensure_index()

    # 2 ── Analyze: extract years + search terms via reasoning model
    if on_status:
        await on_status("Analyzing question...")
    years = _extract_years(input_text)
    search_queries = [input_text]
    is_reasoning = any(model.startswith(p) for p in _REASONING_PREFIXES)

    try:
        if is_reasoning:
            analyze_resp = await asyncio.wait_for(
                client.responses.create(
                    model=model,
                    input=[
                        {"role": "user", "content": ANALYZE_PROMPT.format(question=input_text)},
                    ],
                    reasoning={"effort": "low", "summary": "auto"},
                    max_output_tokens=4000,
                    store=False,
                ),
                timeout=60,
            )
            tracker.record(analyze_resp, label="officeqa-analyze")
            raw = analyze_resp.output_text or ""
        else:
            analyze_resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": ANALYZE_PROMPT.format(question=input_text)},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                ),
                timeout=60,
            )
            tracker.record(analyze_resp, label="officeqa-analyze")
            raw = analyze_resp.choices[0].message.content or ""
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        analysis = json.loads(raw.strip())
        if "search_queries" in analysis and analysis["search_queries"]:
            search_queries = analysis["search_queries"]
        elif "search_query" in analysis:
            search_queries = [analysis["search_query"]]
        if "years" in analysis and not years:
            years = [int(y) for y in analysis["years"]]
    except Exception as exc:
        logger.warning("Analyze step failed, using raw question: %s", exc)

    # 3 ── Table selection: deterministic first, LLM fallback
    selected_table_ids: list[str] = []
    if index.table_catalog:
        if on_status:
            await on_status("Selecting relevant tables...")

        # 3a ── Deterministic selection: keyword→type + year→bulletin + validation
        selected_table_ids = _deterministic_table_select(
            input_text, years, index.table_catalog,
        )

        # 3b ── LLM fallback if deterministic found nothing
        if not selected_table_ids:
            logger.info("Deterministic table select found nothing, falling back to LLM")
            combined_query = input_text + " " + " ".join(search_queries)
            scored_entries = index.catalog_for_query(combined_query, years)
            if scored_entries:
                target_years = years if years else [1950]
                title_year_best: dict[tuple[str, int], tuple[int, _TableMeta]] = {}
                for score, tm in scored_entries:
                    for ty in target_years:
                        key = (tm.title, ty)
                        existing = title_year_best.get(key)
                        ideal = (ty + 1) * 12 + 2
                        cur_score = abs(tm.file_year * 12 + tm.file_month - ideal)
                        if existing is None:
                            title_year_best[key] = (score, tm)
                        else:
                            old_score = abs(
                                existing[1].file_year * 12 + existing[1].file_month - ideal
                            )
                            if cur_score < old_score:
                                title_year_best[key] = (score, tm)
                seen_ids: set[str] = set()
                deduped: list[tuple[int, _TableMeta]] = []
                for score, tm in title_year_best.values():
                    if tm.table_id not in seen_ids:
                        seen_ids.add(tm.table_id)
                        deduped.append((score, tm))
                deduped.sort(key=lambda x: x[0], reverse=True)

                catalog_text = _format_catalog(deduped)
                select_prompt = TABLE_SELECT_PROMPT.format(
                    question=input_text, catalog=catalog_text
                )

                try:
                    if is_reasoning:
                        select_resp = await asyncio.wait_for(
                            client.responses.create(
                                model=model,
                                input=[{"role": "user", "content": select_prompt}],
                                reasoning={"effort": "medium", "summary": "auto"},
                                max_output_tokens=2000,
                                text={"format": _TABLE_SELECT_SCHEMA},
                                store=False,
                            ),
                            timeout=60,
                        )
                        tracker.record(select_resp, label="officeqa-select")
                        select_raw = select_resp.output_text or ""
                    else:
                        select_resp = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "user", "content": select_prompt},
                                ],
                                temperature=0.0,
                                max_tokens=500,
                                response_format={"type": "json_object"},
                            ),
                            timeout=60,
                        )
                        tracker.record(select_resp, label="officeqa-select")
                        select_raw = select_resp.choices[0].message.content or ""

                    select_raw = re.sub(r"```json\s*", "", select_raw)
                    select_raw = re.sub(r"```\s*$", "", select_raw)
                    parsed = json.loads(select_raw.strip())
                    if isinstance(parsed, dict) and "table_ids" in parsed:
                        selected_table_ids = parsed["table_ids"]
                    elif isinstance(parsed, list):
                        selected_table_ids = parsed
                    else:
                        selected_table_ids = []
                    if not isinstance(selected_table_ids, list):
                        selected_table_ids = []
                    valid_ids = {tm.table_id for _, tm in deduped}
                    selected_table_ids = [
                        tid for tid in selected_table_ids if tid in valid_ids
                    ]
                    logger.info(
                        "LLM table selection fallback: %d tables: %s",
                        len(selected_table_ids), selected_table_ids,
                    )
                except Exception as exc:
                    logger.warning("LLM table selection failed: %s", exc)
        else:
            logger.info(
                "Deterministic table select: %d tables: %s",
                len(selected_table_ids), selected_table_ids,
            )

    # 4 ── Retrieval: metadata-guided + BM25 fallback
    if on_status:
        await on_status("Searching corpus...")
    seen: set[str] = set()
    chunks: list[dict] = []

    if selected_table_ids:
        for q in search_queries:
            for c in index.retrieve_by_tables(
                q, selected_table_ids, top_k=TOP_K, years=years or None,
            ):
                key = c["text"][:100]
                if key not in seen:
                    seen.add(key)
                    chunks.append(c)

    if len(chunks) < 5:
        for q in search_queries:
            for c in index.retrieve(q, top_k=TOP_K, years=years or None):
                key = c["text"][:100]
                if key not in seen:
                    seen.add(key)
                    chunks.append(c)
    chunks = chunks[: TOP_K + 10]
    if not chunks:
        chunks = index.retrieve(input_text, top_k=TOP_K)

    # 5 ── Extract raw markdown tables for code-gen
    if on_status:
        await on_status("Extracting tables for analysis...")
    tables_text = _extract_raw_tables(chunks, index)

    # 6 ── Code-gen: LLM writes pandas code to answer the question
    codegen_result = ""
    if tables_text:
        if on_status:
            await on_status("Generating analysis code...")
        codegen_prompt = CODEGEN_PROMPT.format(
            tables=tables_text, question=input_text,
        )

        for attempt in range(4):
            try:
                if is_reasoning:
                    code_resp = await asyncio.wait_for(
                        client.responses.create(
                            model=model,
                            input=[{"role": "user", "content": codegen_prompt}],
                            reasoning={"effort": "medium", "summary": "auto"},
                            max_output_tokens=16000,
                            store=False,
                        ),
                        timeout=120,
                    )
                    tracker.record(code_resp, label=f"officeqa-codegen-{attempt}")
                    code_raw = code_resp.output_text or ""
                else:
                    code_resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": codegen_prompt},
                            ],
                            temperature=0.1 * attempt,
                            max_tokens=4096,
                        ),
                        timeout=120,
                    )
                    tracker.record(code_resp, label=f"officeqa-codegen-{attempt}")
                    code_raw = code_resp.choices[0].message.content or ""

                # Strip markdown code fences if present
                code_raw = re.sub(r"^```(?:python)?\s*\n?", "", code_raw.strip())
                code_raw = re.sub(r"\n?```\s*$", "", code_raw)
                # Strip redundant imports (re, pd, math, etc. already in namespace)
                code_raw = re.sub(
                    r"^(?:import|from)\s+(?:re|math|statistics|pandas)\b.*\n?",
                    "", code_raw, flags=re.MULTILINE,
                )

                logger.info("Code-gen attempt %d:\n%s", attempt, code_raw[:500])

                if on_status:
                    await on_status("Executing analysis code...")
                exec_result = _exec_pandas_code(code_raw, tables_text)
                logger.info("Code-gen result: %s", exec_result[:200])

                if exec_result.startswith("EXEC_ERROR"):
                    # Feed error back for retry
                    logger.warning(
                        "Code execution error (attempt %d): %s",
                        attempt, exec_result,
                    )
                    codegen_prompt = (
                        f"{codegen_prompt}\n\n"
                        f"Your previous code produced this error:\n{exec_result}\n\n"
                        "Fix the code and try again. Write ONLY executable Python code."
                    )
                elif _is_invalid_answer(exec_result):
                    # nan, inf, empty, or other degenerate results -- retry
                    logger.warning(
                        "Code-gen produced invalid answer (attempt %d): %s",
                        attempt, exec_result,
                    )
                    codegen_prompt = (
                        f"{codegen_prompt}\n\n"
                        f"Your previous code printed '{exec_result}' which is not a valid numeric answer. "
                        "Check that:\n"
                        "1. You are reading the correct table and columns\n"
                        "2. Values are not NaN (use .dropna() or check with pd.notna())\n"
                        "3. You are summing ALL relevant rows/columns, not just a subset\n"
                        "4. The final print() outputs a concrete number\n\n"
                        "Fix the code and try again. Write ONLY executable Python code."
                    )
                else:
                    codegen_result = exec_result
                    break
            except Exception as exc:
                logger.warning("Code-gen LLM call failed (attempt %d): %s", attempt, exc)

    # 7 ── Format result
    if codegen_result:
        result = (
            f"<REASONING>\nCode-gen computed answer from parsed tables.\n"
            f"</REASONING>\n"
            f"<FINAL_ANSWER>\n{codegen_result}\n</FINAL_ANSWER>"
        )
        logger.info("OfficeQA code-gen result: %s", codegen_result[:200])
        return result

    # 8 ── Fallback: text-based LLM reasoning (original approach)
    logger.warning("Code-gen failed or no tables found, falling back to text reasoning")
    if on_status:
        await on_status("Falling back to text reasoning...")
    excerpts = _format_excerpts(chunks)
    final_answer = await _llm_answer(
        client, model, input_text, excerpts,
        label="officeqa-fallback", effort="medium",
    )
    result = _normalize_answer(final_answer)
    logger.info("OfficeQA fallback result (%d chars): %.200s...", len(result), result)
    return result


# ── Helpers ───────────────────────────────────────────────────────────


def _format_excerpts(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"--- Excerpt {i} ({chunk['source']}) ---\n{chunk['text']}")
    return "\n\n".join(parts)


def _format_catalog(
    scored_tables: list[tuple[int, _TableMeta]],
) -> str:
    """Format scored table metadata catalog for LLM table selection."""
    lines = []
    for score, tm in scored_tables:
        cols_short = tm.columns[:120] + "..." if len(tm.columns) > 120 else tm.columns
        type_tag = f" [{tm.table_type}]" if tm.table_type else ""
        lines.append(
            f"Relevance: {score}/100{type_tag}\n"
            f"  ID: {tm.table_id}\n"
            f"  Title: {tm.title}\n"
            f"  Source: {tm.source} ({tm.file_year}-{tm.file_month:02d})\n"
            f"  Units: {tm.units or 'not specified'}\n"
            f"  Columns: {cols_short or 'N/A'}\n"
            f"  Years: {tm.year_min}-{tm.year_max}"
        )
    return "\n".join(lines)


async def _llm_answer(
    client: AsyncOpenAI,
    model: str,
    question: str,
    excerpts: str,
    *,
    label: str = "officeqa",
    effort: str = "medium",
) -> str:
    """Call the LLM to answer a question given retrieved excerpts."""
    is_reasoning = any(model.startswith(p) for p in _REASONING_PREFIXES)
    user_content = (
        f"Question: {question}\n\n"
        f"Retrieved excerpts:\n{excerpts}"
    )
    try:
        if is_reasoning:
            response = await asyncio.wait_for(
                client.responses.create(
                    model=model,
                    instructions=SYSTEM_PROMPT,
                    input=[{"role": "user", "content": user_content}],
                    reasoning={"effort": effort, "summary": "auto"},
                    max_output_tokens=16_000,
                    store=False,
                ),
                timeout=120,
            )
            tracker.record(response, label=label)
            return response.output_text or ""
        else:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                ),
                timeout=120,
            )
            tracker.record(response, label=label)
            return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("LLM call failed (%s): %s", label, exc)
        return ""


def _normalize_answer(text: str) -> str:
    if "<FINAL_ANSWER>" in text:
        return text
    text = text.strip()
    return (
        f"<REASONING>\n{text}\n</REASONING>\n"
        "<FINAL_ANSWER>\nUnable to determine\n</FINAL_ANSWER>"
    )
