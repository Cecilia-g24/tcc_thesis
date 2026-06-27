"""
Pre-processing pipeline: merge → translate → export to LIWC-ready CSVs.

Reads raw data from data_raw/, writes two CSV files to
data_clean/01_csvssforliwcmanualinput/:
  - full_dataset_de.csv  (German original)
  - full_dataset_en.csv  (German + English translation)

Usage:
    python pre_processor.py
    python pre_processor.py --data-raw path/to/data_raw --output path/to/out
    python pre_processor.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Iterable | None = None, *args: Any, **kwargs: Any) -> Iterable:  # type: ignore[misc]
        if iterable is None:
            return []
        return iterable


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_paths() -> dict[str, str]:
    config_path = Path(__file__).resolve().parent / "configs" / "paths.json"
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)

_PATHS: dict[str, str] = _load_paths()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_LINE_RE = re.compile(r"^\s*###\s*(.*?)\s*###\s*$")
PARTICIPANT_ID_RE = re.compile(r"^AU\d{2}\.\d{6}(?:\.txt)?$", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'À-ɏ])")
BLANKLINE_RE = re.compile(r"\n\s*\n+")

EXPECTED_RATERS = ("Rater1", "Rater2", "Rater3")
VALID_RATING_VALUES = {0, 1, 2, 3, 4}

# Maps substrings in transcript filenames (lowercased) to canonical dimensions.
# Checked in order — more specific keywords first.
_TRANSCRIPT_KEYWORDS: list[tuple[str, str]] = [
    ("ambiguitätstoleranz", "d5_ambiguity_tolerance"),
    ("ambiguität",          "d5_ambiguity_tolerance"),
    ("familiensystem",      "d4_family_system"),
    ("familie",             "d4_family_system"),
    ("krankheitsvorstellungen", "d1_illness_beliefs"),
    ("krankheit",           "d1_illness_beliefs"),
    ("kulturelle",          "d3_cultural_factors"),
    ("kultur",              "d3_cultural_factors"),
    ("nichtwissen",         "d2_lack_of_knowledge"),
]

# Maps digit in DimensionN.xlsx filename to canonical dimension.
_RATING_DIM_MAP: dict[str, str] = {
    "1": "d1_illness_beliefs",
    "2": "d2_lack_of_knowledge",
    "3": "d3_cultural_factors",
    "4": "d4_family_system",
    "5": "d5_ambiguity_tolerance",
}

_DIMENSION_NUMBER_RE = re.compile(r"dimension(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_transcript_files(transcripts_dir: Path) -> dict[str, list[Path]]:
    """Scan transcripts_dir recursively and group .txt files by dimension."""
    result: dict[str, list[Path]] = {}
    for path in sorted(transcripts_dir.rglob("*.txt")):
        stem = path.stem.lower()
        dim = next((d for kw, d in _TRANSCRIPT_KEYWORDS if kw in stem), None)
        if dim is None:
            print(f"[WARNING] Could not determine dimension for transcript: {path.name} — skipped")
            continue
        result.setdefault(dim, []).append(path)
    return result


def discover_rating_files(ratings_dir: Path) -> dict[str, Path]:
    """Scan ratings_dir for DimensionN.xlsx files and map them to dimensions."""
    result: dict[str, Path] = {}
    for path in sorted(ratings_dir.glob("*.xlsx")):
        match = _DIMENSION_NUMBER_RE.search(path.stem)
        if match is None:
            continue
        dim = _RATING_DIM_MAP.get(match.group(1))
        if dim is None:
            print(f"[WARNING] Unknown dimension number in: {path.name} — skipped")
            continue
        result[dim] = path
    return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_col(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def _normalize_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\.txt$", "", text, flags=re.IGNORECASE).rstrip(".")
    return text or None


def _parse_rating(value: Any) -> int | float | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return str(value).strip()


def _is_valid_rating(value: Any) -> bool:
    return isinstance(value, (int, float)) and value in VALID_RATING_VALUES


# ---------------------------------------------------------------------------
# Step 1 — Merge human ratings
# ---------------------------------------------------------------------------

def _resolve_rating_columns(df: pd.DataFrame, file_path: Path) -> tuple[str, dict[str, str]]:
    lookup = {_normalize_col(col): col for col in df.columns}
    id_col = None
    for candidate in ("audio", "id", "filename", "file", "transcriptid", "transcript"):
        if candidate in lookup:
            id_col = lookup[candidate]
            break
    if id_col is None:
        raise ValueError(f"No ID/audio column found in {file_path}. Columns: {list(df.columns)}")
    rater_cols: dict[str, str] = {}
    for rater in EXPECTED_RATERS:
        key = _normalize_col(rater)
        if key not in lookup:
            raise ValueError(f"Missing {rater} in {file_path}. Columns: {list(df.columns)}")
        rater_cols[rater] = lookup[key]
    return id_col, rater_cols


def _load_rating_file(file_path: Path, dimension: str) -> list[dict[str, Any]]:
    df = pd.read_excel(file_path, sheet_name=0, dtype=object).dropna(how="all").copy()
    df.columns = [str(col).strip() for col in df.columns]
    id_col, rater_cols = _resolve_rating_columns(df, file_path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        norm_id = _normalize_id(row[id_col])
        r1 = _parse_rating(row[rater_cols["Rater1"]])
        r2 = _parse_rating(row[rater_cols["Rater2"]])
        r3 = _parse_rating(row[rater_cols["Rater3"]])
        all_valid = all(_is_valid_rating(v) for v in (r1, r2, r3))
        average_score = round((float(r1) + float(r2) + float(r3)) / 3, 3) if all_valid else None
        rows.append({
            "id": norm_id,
            "dimension": dimension,
            "rater_one": r1,
            "rater_two": r2,
            "rater_three": r3,
            "average_score": average_score,
        })
    return rows


def merge_human_ratings(rating_files: dict[str, Path]) -> list[dict[str, Any]]:
    """Load and merge all human rating Excel files into a flat list of records."""
    rows: list[dict[str, Any]] = []
    for dimension, file_path in sorted(rating_files.items()):
        if not file_path.exists():
            print(f"[WARNING] Rating file not found: {file_path}")
            continue
        rows.extend(_load_rating_file(file_path, dimension))
    rows.sort(key=lambda r: (r["dimension"], r["id"] or ""))
    return rows


# ---------------------------------------------------------------------------
# Step 1 — Merge transcripts
# ---------------------------------------------------------------------------

def _extract_transcript_records(text: str, dimension: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_id: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        line_clean = line.rstrip("\r\n")
        match = HEADER_LINE_RE.match(line_clean)
        if match:
            header = match.group(1).strip()
            if PARTICIPANT_ID_RE.match(header):
                if current_id is not None:
                    records.append({
                        "id": current_id,
                        "dimension": dimension,
                        "text": "".join(current_lines).strip(),
                    })
                current_id = _normalize_id(header)
                current_lines = []
            continue
        if current_id is not None:
            current_lines.append(line)
    if current_id is not None:
        records.append({
            "id": current_id,
            "dimension": dimension,
            "text": "".join(current_lines).strip(),
        })
    return records


def merge_transcripts(transcript_files: dict[str, list[Path]]) -> list[dict[str, Any]]:
    """Load and merge all transcript text files into a flat list of records."""
    rows: list[dict[str, Any]] = []
    for dimension, file_paths in sorted(transcript_files.items()):
        for file_path in file_paths:
            if not file_path.exists():
                print(f"[WARNING] Transcript file not found: {file_path}")
                continue
            text = file_path.read_text(encoding="utf-8-sig")
            rows.extend(_extract_transcript_records(text, dimension))
    rows.sort(key=lambda r: (r["dimension"], r["id"] or ""))
    return rows


# ---------------------------------------------------------------------------
# Step 1 — Combine into full dataset
# ---------------------------------------------------------------------------

def merge_full_dataset(
    rating_rows: list[dict[str, Any]],
    transcript_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join rating and transcript rows on (id, dimension)."""
    rating_index = {(r["id"], r["dimension"]): r for r in rating_rows}
    transcript_index = {(r["id"], r["dimension"]): r for r in transcript_rows}

    rating_keys = set(rating_index)
    transcript_keys = set(transcript_index)

    only_ratings = rating_keys - transcript_keys
    only_transcripts = transcript_keys - rating_keys
    if only_ratings:
        print(f"[WARNING] {len(only_ratings)} rating(s) have no matching transcript — excluded")
    if only_transcripts:
        print(f"[WARNING] {len(only_transcripts)} transcript(s) have no matching rating — excluded")

    matched = sorted(rating_keys & transcript_keys, key=lambda k: (k[1], k[0] or ""))
    full_rows: list[dict[str, Any]] = []
    for key in matched:
        rating = rating_index[key]
        transcript = transcript_index[key]
        full_rows.append({
            "id": rating["id"],
            "dimension": rating["dimension"],
            "text": transcript["text"],
            "rater_one": rating["rater_one"],
            "rater_two": rating["rater_two"],
            "rater_three": rating["rater_three"],
            "average_score": rating["average_score"],
        })
    return full_rows


def merge(data_raw_dir: Path) -> list[dict[str, Any]]:
    """Run the full merge step and return the combined dataset."""
    transcripts_dir = data_raw_dir / _PATHS["transcripts_subdir"]
    ratings_dir = data_raw_dir / _PATHS["ratings_subdir"]

    transcript_files = discover_transcript_files(transcripts_dir)
    rating_files = discover_rating_files(ratings_dir)

    print(f"Found {sum(len(v) for v in transcript_files.values())} transcript files "
          f"across {len(transcript_files)} dimensions")
    print(f"Found {len(rating_files)} rating files")

    rating_rows = merge_human_ratings(rating_files)
    transcript_rows = merge_transcripts(transcript_files)
    full_rows = merge_full_dataset(rating_rows, transcript_rows)

    print(f"Merged dataset: {len(full_rows)} records")
    return full_rows


# ---------------------------------------------------------------------------
# Step 2 — Translate
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    paragraphs: list[str] = []
    for para in BLANKLINE_RE.split(text):
        lines = [line.strip() for line in para.split("\n") if line.strip()]
        if lines:
            paragraphs.append(" ".join(lines))
    return "\n\n".join(paragraphs)


def _chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]:
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), limit):
                chunks.append(sentence[start: start + limit].strip())
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _translate_chunk(chunk: str, translator: Any, retries: int, wait_seconds: float) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = translator.translate(chunk)
            return result if result is not None else ""
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(wait_seconds * attempt)
    raise RuntimeError(f"Translation failed after {retries} attempts: {last_error}")


def _translate_text(
    text: str,
    translator: Any,
    chunk_limit: int = 4000,
    retries: int = 3,
    wait_seconds: float = 1.0,
) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    translated_paras: list[str] = []
    for para in normalized.split("\n\n"):
        parts = [
            _translate_chunk(chunk, translator, retries, wait_seconds).strip()
            for chunk in _chunk_text(para, chunk_limit)
            if chunk.strip()
        ]
        translated_paras.append(" ".join(p for p in parts if p))
    return "\n\n".join(p for p in translated_paras if p.strip())


def translate(
    full_rows: list[dict[str, Any]],
    chunk_limit: int = 4000,
    retries: int = 3,
    wait_seconds: float = 1.0,
    sleep_between_records: float = 0.0,
) -> list[dict[str, Any]]:
    """Translate the ``text`` field of each record from German to English."""
    if GoogleTranslator is None:
        raise RuntimeError(
            "deep-translator is not installed.\n"
            "Install it with:  pip install deep-translator"
        )

    translator = GoogleTranslator(source="de", target="en")
    output_rows: list[dict[str, Any]] = []
    n_translated = n_empty = n_failed = 0

    with tqdm(total=len(full_rows), desc="Translating", unit="record") as bar:
        for idx, record in enumerate(full_rows):
            record_id = record.get("id", f"row_{idx}")
            text_de = record.get("text", "")
            text_en = ""
            if text_de.strip():
                try:
                    text_en = _translate_text(text_de, translator, chunk_limit, retries, wait_seconds)
                    n_translated += 1
                except Exception as exc:  # noqa: BLE001
                    n_failed += 1
                    print(f"[WARNING] Translation failed for {record_id}: {exc}")
            else:
                n_empty += 1

            output_rows.append({
                "id": record["id"],
                "dimension": record["dimension"],
                "text": text_de,
                "text_en": text_en,
                "rater_one": record["rater_one"],
                "rater_two": record["rater_two"],
                "rater_three": record["rater_three"],
                "average_score": record["average_score"],
            })

            if hasattr(bar, "set_postfix"):
                bar.set_postfix(id=record_id, ok=n_translated, fail=n_failed, refresh=False)
            if hasattr(bar, "update"):
                bar.update(1)

            if sleep_between_records > 0 and idx < len(full_rows) - 1:
                time.sleep(sleep_between_records)

    print(f"Translation complete — translated: {n_translated}, empty: {n_empty}, failed: {n_failed}")
    return output_rows


# ---------------------------------------------------------------------------
# Step 3 — Export CSVs
# ---------------------------------------------------------------------------

def _collapse_newlines(series: pd.Series) -> pd.Series:
    """Replace newlines with a single space so LIWC sees one text per row."""
    return series.str.replace(r"\r?\n", " ", regex=True).str.strip()


def export_csvs(
    full_rows_de: list[dict[str, Any]],
    full_rows_en: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write LIWC-ready CSVs for the German and English datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_de = output_dir / _PATHS["output_de_csv"]
    out_en = output_dir / _PATHS["output_en_csv"]

    df_de = pd.DataFrame(full_rows_de)
    df_de["text"] = _collapse_newlines(df_de["text"])
    df_de = df_de[["id", "dimension", "text", "rater_one", "rater_two", "rater_three", "average_score"]]

    df_en = pd.DataFrame(full_rows_en)
    df_en["text"] = _collapse_newlines(df_en["text"])
    df_en["text_en"] = _collapse_newlines(df_en["text_en"])
    df_en = df_en[["id", "dimension", "text", "text_en", "rater_one", "rater_two", "rater_three", "average_score"]]

    df_de.to_csv(out_de, index=False, encoding="utf-8-sig")
    df_en.to_csv(out_en, index=False, encoding="utf-8-sig")

    print(f"DE CSV → {out_de}  ({len(df_de)} rows)")
    print(f"EN CSV → {out_en}  ({len(df_en)} rows)")
    return out_de, out_en


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    data_raw_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    sleep_between_records: float = 0.0,
) -> None:
    """Run merge → translate → export from data_raw_dir to output_dir."""
    data_clean_dir = output_dir.parent
    _ensure_output_folders(data_clean_dir)

    print(f"\nData source: {data_raw_dir}")
    print(f"Output dir:  {output_dir}\n")

    full_rows_de = merge(data_raw_dir)

    if dry_run:
        non_empty = sum(1 for r in full_rows_de if str(r.get("text", "")).strip())
        print(f"\n[dry-run] {non_empty} records would be translated — no API calls made.")
        return

    full_rows_en = translate(full_rows_de, sleep_between_records=sleep_between_records)
    export_csvs(full_rows_de, full_rows_en, output_dir)
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_data_raw() -> Path:
    return Path(__file__).resolve().parent / _PATHS["data_raw"]


def _default_output() -> Path:
    return Path(__file__).resolve().parent / _PATHS["data_clean"] / _PATHS["output_csvs_subdir"]


def _ensure_output_folders(data_clean_dir: Path) -> None:
    """Create expected output folders under data_clean/ if they don't exist."""
    (data_clean_dir / _PATHS["liwc_results_subdir"]).mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge, translate, and export transcript data to LIWC-ready CSVs."
    )
    parser.add_argument(
        "--data-raw",
        type=Path,
        default=_default_data_raw(),
        help="Path to the data_raw folder (default: data/data_raw/ next to this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output(),
        help="Output directory for CSV files (default: data/data_clean/01_csvssforliwcmanualinput/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the merge step only; skip translation and file writing.",
    )
    parser.add_argument(
        "--sleep-between-records",
        type=float,
        default=0.0,
        help="Seconds to sleep between translation requests (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        data_raw_dir=args.data_raw.resolve(),
        output_dir=args.output.resolve(),
        dry_run=args.dry_run,
        sleep_between_records=args.sleep_between_records,
    )
