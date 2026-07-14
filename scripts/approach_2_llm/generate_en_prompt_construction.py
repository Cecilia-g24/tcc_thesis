"""
One-off generator: produce en_prompt_construction.py by machine-translating
de_prompt_construction.py's German content into English.

Uses the same translation engine as pre_processor.py's transcript translation
(deep_translator.GoogleTranslator, source="de", target="en"). This script is a
build-time tool only: the generated en_prompt_construction.py does not import
deep_translator and makes no network calls at LLM-assessment runtime -- it is a
plain static module, architecturally identical to de_prompt_construction.py.

Everything that ends up inside the rendered LLM prompt (dimension name,
instruction, criteria, note, checklist, anchor examples, the shared task-framing
and scoring-rule blocks, and the section labels baked into the section-builder
functions) is translated. Developer-facing text that is already English in
de_prompt_construction.py (docstrings, comments, argparse help, error messages)
is carried over/adapted directly, not machine-translated.

Run once, or whenever de_prompt_construction.py's content changes:
    python generate_en_prompt_construction.py

Output:
    scripts/approach_2_llm/en_prompt_construction.py

IMPORTANT: review the generated file before using it for LLM assessments.
Machine translation of clinical/therapy terminology and the quoted anchor
example utterances can be imprecise. Treat the reviewed, committed file as
frozen -- re-run this generator (and re-review) only if the German source
content changes.
"""

from __future__ import annotations

import time
from pathlib import Path

from deep_translator import GoogleTranslator
from tqdm import tqdm

import de_prompt_construction as de

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "en_prompt_construction.py"

TRANSLATOR = GoogleTranslator(source="de", target="en")
_CACHE: dict[str, str] = {}
_call_count = 0
_progress_bar: tqdm | None = None


def t(text: str, retries: int = 3, wait_seconds: float = 1.5) -> str:
    """Translate one short string DE->EN, memoized, with simple retry."""
    global _call_count
    text = text.strip()
    if not text:
        return ""
    if text in _CACHE:
        result = _CACHE[text]
        if _progress_bar is not None:
            _progress_bar.update(1)
        return result
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = (TRANSLATOR.translate(text) or "").strip()
            _CACHE[text] = result
            _call_count += 1
            if _progress_bar is not None:
                _progress_bar.update(1)
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(wait_seconds * attempt)
    raise RuntimeError(f"Translation failed after {retries} attempts for {text!r}: {last_error}")


def t_bulleted_block(block: str) -> str:
    """Translate a block containing an optional lead-in line plus '- ...' bullet
    lines, one line at a time, preserving line structure and bullet markers exactly."""
    out_lines: list[str] = []
    for line in block.split("\n"):
        if not line.strip():
            out_lines.append("")
        elif line.startswith("- "):
            out_lines.append(f"- {t(line[2:])}")
        else:
            out_lines.append(t(line))
    return "\n".join(out_lines)


def t_paragraphs(block: str) -> str:
    """Translate blank-line-separated paragraphs, preserving paragraph breaks."""
    paras = block.strip("\n").split("\n\n")
    return "\n\n".join(t(p) for p in paras)


# ---------------------------------------------------------------------------
# Translate DIMENSIONS and VARIANTS content (structure/toggles unchanged;
# codes and ids unchanged since they are shared keys across languages)
# ---------------------------------------------------------------------------

def translate_dimension(spec: de.DimensionSpec) -> de.DimensionSpec:
    return de.DimensionSpec(
        code=spec.code,
        name=t(spec.name),
        instruction=t(spec.instruction),
        criteria=t_bulleted_block(spec.criteria),
        note=t(spec.note),
        checklist=[t(item) for item in spec.checklist],
        anchor_examples={
            score: [t(ex) for ex in examples]
            for score, examples in spec.anchor_examples.items()
        },
    )


def translate_variant(spec: de.VariantSpec) -> de.VariantSpec:
    return de.VariantSpec(
        id=spec.id,
        name=t(spec.name),
        why_chosen=t(spec.why_chosen),
        include_anchors=spec.include_anchors,
        include_rationale=spec.include_rationale,
        rationale_before_score=spec.rationale_before_score,
        criteria_format=spec.criteria_format,
        include_criteria=spec.include_criteria,
        include_note=spec.include_note,
        task_framing=spec.task_framing,
        scoring_rules=spec.scoring_rules,
    )


# ---------------------------------------------------------------------------
# Section-builder label strings hardcoded in de_prompt_construction.py's
# function bodies (not exposed as module-level constants there, so they are
# reproduced here verbatim before translation).
# ---------------------------------------------------------------------------

LABELS_DE: dict[str, str] = {
    "target": "Zieldimension:",
    "instruction": "Instruktion:",
    "criteria_bullets": "Bewertungskriterien:",
    "criteria_checklist_header": (
        "Bewertungskriterien (Checklistenform; dieselben Leistungsmerkmale des "
        "Manuals, umstrukturiert):"
    ),
    "criteria_checklist_intro_prefix": (
        "Nutzen Sie die folgende Checkliste, um zu beurteilen, ob die Reaktion "
        "dieselben im Manual beschriebenen Kriterien erfüllt:"
    ),
    "criteria_checklist_guidance": (
        "Nutzen Sie die Checkliste zur Steuerung der Aufmerksamkeit, zählen Sie die "
        "Punkte aber nicht mechanisch ab. Vergeben Sie die abschließende Bewertung "
        "von 0-4 ganzheitlich gemäß dem Ratingmanual."
    ),
    "note": "Wichtiger dimensionsspezifischer Hinweis:",
    "anchors": "Ankerbeispiele aus dem Ratingmanual:",
    "anchor_score_prefix": "Beispiele für Bewertung",
    "scoring_general": "Allgemeine Bewertungsregeln:",
    "scoring_minimal": "Bewertungsskala:",
    "transcript": "Transkript:",
    "output_no_rationale": (
        'Geben Sie ausschließlich gültiges JSON mit genau diesem Schlüssel zurück. '
        'Der Wert von "score" muss eine Ganzzahl 0, 1, 2, 3 oder 4 sein:'
    ),
    "output_rationale_before_lead": (
        "Geben Sie zunächst eine kurze Begründung an, die sich auf konkrete Belege "
        "aus dem Transkript stützt. Vergeben Sie anschließend die abschließende "
        "Bewertung, die mit dieser Begründung übereinstimmt."
    ),
    "output_rationale_before_instr": (
        'Geben Sie ausschließlich gültiges JSON mit genau diesen Schlüsseln in dieser '
        'Reihenfolge zurück. Der Wert von "score" muss eine Ganzzahl 0, 1, 2, 3 oder 4 sein:'
    ),
    "output_rationale_after_instr": (
        'Geben Sie ausschließlich gültiges JSON mit genau diesen Schlüsseln zurück. '
        'Der Wert von "score" muss eine Ganzzahl 0, 1, 2, 3 oder 4 sein:'
    ),
}


def count_translatable_units() -> int:
    """Count how many individual t() calls main() will make, matching the exact
    line/paragraph/item splitting used by translate_dimension/translate_variant/
    t_bulleted_block/t_paragraphs, so the tqdm progress bar total is accurate."""
    count = 0
    for spec in de.DIMENSIONS.values():
        count += 1  # name
        count += 1  # instruction
        count += len([line for line in spec.criteria.split("\n") if line.strip()])
        count += 1  # note
        count += len(spec.checklist)
        count += sum(len(examples) for examples in spec.anchor_examples.values())
    for _ in de.VARIANTS.values():
        count += 2  # name, why_chosen
    count += len([line for line in de.GENERAL_SCORING_RULES.split("\n") if line.strip()])
    count += len([p for p in de.STANDARD_TASK_FRAMING.strip("\n").split("\n\n") if p.strip()])
    count += len([p for p in de.MINIMAL_TASK_FRAMING.strip("\n").split("\n\n") if p.strip()])
    count += len([line for line in de.MINIMAL_SCORING_RULES.split("\n") if line.strip()])
    count += len(LABELS_DE)
    return count


# ---------------------------------------------------------------------------
# Render en_prompt_construction.py source text
# ---------------------------------------------------------------------------

MODULE_DOCSTRING = '''"""
English-language LLM-as-judge prompt templates for transcultural competence ratings.

This is the English counterpart of de_prompt_construction.py. Its content (dimension
name, instruction, criteria, note, checklist, anchor examples, the shared task-framing
and scoring-rule text, and section labels) was produced by machine-translating
de_prompt_construction.py's German content from German to English, using the same
translation engine used for the therapy transcripts in pre_processor.py
(deep_translator.GoogleTranslator, source="de", target="en"). Generation is a one-time,
human-reviewed step performed by generate_en_prompt_construction.py -- this file itself
has no translation/network dependency and is treated as frozen, reviewed content.

The section-building architecture mirrors de_prompt_construction.py exactly (same
DimensionSpec / VariantSpec / build_prompt design) so the two modules stay easy to
compare; only the language and the translated wording differ. One deliberate deviation:
_section_criteria() below splits the criteria text on its first newline rather than on
a German marker phrase, since a marker phrase is not guaranteed to survive translation
unchanged -- the first-line-is-the-intro-sentence structure is language-independent and
behaves identically to de_prompt_construction.py's marker-string split for this data.

The dimension code passed to build_prompt must be one of:
    d1_illness_beliefs
    d2_lack_of_knowledge
    d3_cultural_factors
    d4_family_system
    d5_ambiguity_tolerance

The variant_id passed to build_prompt must be one of:
    V1_full_manual_baseline
    V2_no_anchors
    V3_no_rationale
    V4_evidence_before_score
    V5_structured_checklist
    V6_minimal_natural

The output JSON keys ("score", "brief_rationale") are kept in English across both the
German and English prompt variants, so downstream parsing code (e.g.
llm_assessment_api.py's valid_integer_score(parsed.get("score"))) works unchanged
regardless of prompt language.

Usage (from an assessment script):
    from en_prompt_construction import build_prompt

    prompt = build_prompt(
        dimension_code="d1_illness_beliefs",
        transcript=transcript_text,
        variant_id="V1_full_manual_baseline",
    )
"""'''


def render_dict_block(name: str, type_hint: str, items: dict[str, object]) -> str:
    lines = [f"{name}: {type_hint} = {{"]
    for key, value in items.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    return "\n".join(lines)


def render_file(
    dimensions_en: dict[str, de.DimensionSpec],
    variants_en: dict[str, de.VariantSpec],
    general_scoring_rules_en: str,
    standard_task_framing_en: str,
    minimal_task_framing_en: str,
    minimal_scoring_rules_en: str,
    labels_en: dict[str, str],
) -> str:
    parts: list[str] = []

    parts.append(
        "### to preview the 6 prompt variants for every dimension, run this script with the\n"
        "### --display flag. The prompts will be saved as .txt files in\n"
        "### data/assets/en_assets/en_prompts/<dimension_code>/.\n"
        "# python en_prompt_construction.py --display"
    )

    parts.append(MODULE_DOCSTRING)

    parts.append(
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "from typing import Literal"
    )

    parts.append(
        "# ---------------------------------------------------------------------------\n"
        "# Shared rating information from the rating manual (machine-translated)\n"
        "# ---------------------------------------------------------------------------"
    )

    parts.append(f"GENERAL_SCORING_RULES = {general_scoring_rules_en!r}")
    parts.append(f"STANDARD_TASK_FRAMING = {standard_task_framing_en!r}")
    parts.append(f"MINIMAL_TASK_FRAMING = {minimal_task_framing_en!r}")
    parts.append(f"MINIMAL_SCORING_RULES = {minimal_scoring_rules_en!r}")

    parts.append(
        "# ---------------------------------------------------------------------------\n"
        "# Section labels used by the section-builder functions below (machine-translated)\n"
        "# ---------------------------------------------------------------------------"
    )
    for key, value in labels_en.items():
        parts.append(f"LABEL_{key.upper()} = {value!r}")

    parts.append(
        "@dataclass(frozen=True)\n"
        "class DimensionSpec:\n"
        "    code: str\n"
        "    name: str\n"
        "    instruction: str\n"
        "    criteria: str\n"
        "    note: str\n"
        "    checklist: list[str]\n"
        "    anchor_examples: dict[int, list[str]]"
    )

    parts.append(render_dict_block("DIMENSIONS", "dict[str, DimensionSpec]", dimensions_en))

    parts.append(
        "CriteriaFormat = Literal[\"bullets\", \"checklist\"]\n"
        "TaskFraming = Literal[\"standard\", \"minimal\"]\n"
        "ScoringRules = Literal[\"full\", \"minimal\"]\n\n\n"
        "@dataclass(frozen=True)\n"
        "class VariantSpec:\n"
        "    id: str\n"
        "    name: str\n"
        "    why_chosen: str\n"
        "    include_anchors: bool\n"
        "    include_rationale: bool\n"
        "    rationale_before_score: bool = False\n"
        "    criteria_format: CriteriaFormat = \"bullets\"\n"
        "    include_criteria: bool = True\n"
        "    include_note: bool = True\n"
        "    task_framing: TaskFraming = \"standard\"\n"
        "    scoring_rules: ScoringRules = \"full\""
    )

    parts.append(render_dict_block("VARIANTS", "dict[str, VariantSpec]", variants_en))

    parts.append(
        "def format_anchor_examples(anchor_examples: dict[int, list[str]]) -> str:\n"
        '    """Format score-level anchor examples for insertion into a prompt."""\n'
        "    lines: list[str] = []\n"
        "    for score in sorted(anchor_examples.keys(), reverse=True):\n"
        '        lines.append(f"{LABEL_ANCHOR_SCORE_PREFIX} {score}:")\n'
        "        for ex in anchor_examples[score]:\n"
        '            lines.append(f"- {ex}")\n'
        '    return "\\n".join(lines)\n\n\n'
        "def format_checklist(checklist: list[str]) -> str:\n"
        '    """Format the checklist as a numbered list."""\n'
        '    return "\\n".join(f"{i}. {item}" for i, item in enumerate(checklist, start=1))'
    )

    parts.append(
        "def _section_task_framing(variant: VariantSpec) -> str:\n"
        '    if variant.task_framing == "minimal":\n'
        "        return MINIMAL_TASK_FRAMING.strip()\n"
        "    return STANDARD_TASK_FRAMING.strip()\n\n\n"
        "def _section_target(spec: DimensionSpec) -> str:\n"
        '    return f"{LABEL_TARGET}\\n{spec.name}"\n\n\n'
        "def _section_instruction(spec: DimensionSpec) -> str:\n"
        '    return f"{LABEL_INSTRUCTION}\\n{spec.instruction}"\n\n\n'
        "def _section_criteria(spec: DimensionSpec, variant: VariantSpec) -> str:\n"
        "    if not variant.include_criteria:\n"
        '        return ""\n'
        '    if variant.criteria_format == "checklist":\n'
        "        # Language-independent equivalent of de_prompt_construction.py's\n"
        "        # marker-string split: the criteria text's first line is always the\n"
        "        # intro sentence, everything after is bullet lines.\n"
        '        criteria_intro = spec.criteria.split("\\n", 1)[0].strip()\n'
        "        return (\n"
        '            f"{LABEL_CRITERIA_CHECKLIST_HEADER}\\n"\n'
        '            f"{LABEL_CRITERIA_CHECKLIST_INTRO_PREFIX} {criteria_intro}\\n"\n'
        '            f"{LABEL_CRITERIA_CHECKLIST_GUIDANCE}\\n\\n"\n'
        '            f"{format_checklist(spec.checklist)}"\n'
        "        )\n"
        '    return f"{LABEL_CRITERIA_BULLETS}\\n{spec.criteria}"\n\n\n'
        "def _section_note(spec: DimensionSpec, variant: VariantSpec) -> str:\n"
        "    if not variant.include_note:\n"
        '        return ""\n'
        '    return f"{LABEL_NOTE}\\n{spec.note}"\n\n\n'
        "def _section_anchors(spec: DimensionSpec, variant: VariantSpec) -> str:\n"
        "    if not variant.include_anchors:\n"
        '        return ""\n'
        '    return f"{LABEL_ANCHORS}\\n{format_anchor_examples(spec.anchor_examples)}"\n\n\n'
        "def _section_scoring_rules(variant: VariantSpec) -> str:\n"
        '    if variant.scoring_rules == "minimal":\n'
        '        return f"{LABEL_SCORING_MINIMAL}\\n{MINIMAL_SCORING_RULES.strip()}"\n'
        '    return f"{LABEL_SCORING_GENERAL}\\n{GENERAL_SCORING_RULES.strip()}"\n\n\n'
        "def _section_transcript(transcript: str) -> str:\n"
        '    return f"{LABEL_TRANSCRIPT}\\n{transcript}"\n\n\n'
        "def _section_output(variant: VariantSpec) -> str:\n"
        "    if not variant.include_rationale:\n"
        "        return (\n"
        '            f"{LABEL_OUTPUT_NO_RATIONALE}\\n"\n'
        '            "{\\n"\n'
        '            \'  "score": 0\\n\'\n'
        '            "}"\n'
        "        )\n"
        "    if variant.rationale_before_score:\n"
        "        return (\n"
        '            f"{LABEL_OUTPUT_RATIONALE_BEFORE_LEAD}\\n\\n"\n'
        '            f"{LABEL_OUTPUT_RATIONALE_BEFORE_INSTR}\\n"\n'
        '            "{\\n"\n'
        '            \'  "brief_rationale": "...",\\n\'\n'
        '            \'  "score": 0\\n\'\n'
        '            "}"\n'
        "        )\n"
        "    return (\n"
        '        f"{LABEL_OUTPUT_RATIONALE_AFTER_INSTR}\\n"\n'
        '        "{\\n"\n'
        '        \'  "score": 0,\\n\'\n'
        '        \'  "brief_rationale": "..."\\n\'\n'
        '        "}"\n'
        "    )"
    )

    parts.append(
        "def build_prompt(dimension_code: str, transcript: str, variant_id: str) -> str:\n"
        '    """Construct one rendered English prompt from plain variables.\n\n'
        "    Args:\n"
        '        dimension_code: One of the keys in DIMENSIONS, e.g. "d1_illness_beliefs".\n'
        "        transcript: The participant's transcript text to be rated.\n"
        '        variant_id: One of the keys in VARIANTS, e.g. "V1_full_manual_baseline".\n'
        '    """\n'
        "    if dimension_code not in DIMENSIONS:\n"
        '        valid = ", ".join(DIMENSIONS)\n'
        "        raise ValueError(f\"Unknown dimension code {dimension_code!r}. Valid dimension codes: {valid}\")\n"
        "    if variant_id not in VARIANTS:\n"
        '        valid = ", ".join(VARIANTS)\n'
        "        raise ValueError(f\"Unknown prompt variant {variant_id!r}. Valid variants: {valid}\")\n\n"
        "    spec = DIMENSIONS[dimension_code]\n"
        "    variant = VARIANTS[variant_id]\n"
        '    transcript_text = "" if transcript is None else str(transcript).strip()\n\n'
        "    sections = [\n"
        "        _section_task_framing(variant),\n"
        "        _section_target(spec),\n"
        "        _section_instruction(spec),\n"
        "        _section_criteria(spec, variant),\n"
        "        _section_note(spec, variant),\n"
        "        _section_anchors(spec, variant),\n"
        "        _section_scoring_rules(variant),\n"
        "        _section_transcript(transcript_text),\n"
        "        _section_output(variant),\n"
        "    ]\n"
        '    return "\\n\\n".join(section for section in sections if section).strip()'
    )

    parts.append(
        "REPO_ROOT = Path(__file__).resolve().parents[2]\n"
        'DEFAULT_PROMPTS_DIR = REPO_ROOT / "data" / "assets" / "en_assets" / "en_prompts"'
    )

    parts.append(
        "def display_variants_for_dimension(dimension_code: str, output_dir: Path = DEFAULT_PROMPTS_DIR) -> list[Path]:\n"
        '    """Render all prompt variants for one dimension and write each to its own .txt file.\n\n'
        "    The transcript field is left as a literal placeholder since it is supplied\n"
        "    per-call by whatever script calls build_prompt().\n\n"
        "    Args:\n"
        '        dimension_code: One of the keys in DIMENSIONS, e.g. "d1_illness_beliefs".\n'
        "        output_dir: Directory where the variant text files are written, under\n"
        "            a subfolder named after the dimension code.\n"
        '    """\n'
        "    if dimension_code not in DIMENSIONS:\n"
        '        valid = ", ".join(DIMENSIONS)\n'
        "        raise ValueError(f\"Unknown dimension code {dimension_code!r}. Valid dimension codes: {valid}\")\n\n"
        "    dimension_dir = output_dir / dimension_code\n"
        "    dimension_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "    written: list[Path] = []\n"
        "    for variant_id in VARIANTS:\n"
        '        prompt = build_prompt(dimension_code, transcript="{transcript}", variant_id=variant_id)\n'
        '        file_path = dimension_dir / f"{variant_id}.txt"\n'
        '        file_path.write_text(prompt, encoding="utf-8")\n'
        "        written.append(file_path)\n\n"
        "    return written\n\n\n"
        "def display_variants_for_all_dimensions(output_dir: Path = DEFAULT_PROMPTS_DIR) -> list[Path]:\n"
        '    """Render all prompt variants for every dimension and write each to its own .txt file."""\n'
        "    written: list[Path] = []\n"
        "    for dimension_code in DIMENSIONS:\n"
        "        written.extend(display_variants_for_dimension(dimension_code, output_dir))\n"
        "    return written"
    )

    parts.append(
        "def parse_args() -> argparse.Namespace:\n"
        '    parser = argparse.ArgumentParser(description="English LLM-as-judge prompt construction utilities.")\n'
        "    parser.add_argument(\n"
        '        "--display",\n'
        '        action="store_true",\n'
        '        help="Render prompt variants for every dimension and save them as .txt files "\n'
        '        "under data/assets/en_assets/en_prompts/<dimension_code>/.",\n'
        "    )\n"
        "    return parser.parse_args()"
    )

    parts.append(
        'if __name__ == "__main__":\n'
        "    args = parse_args()\n"
        "    if args.display:\n"
        "        for path in display_variants_for_all_dimensions():\n"
        '            print(f"Saved: {path}")\n'
        "    else:\n"
        '        print("Nothing to do. Pass --display to render the prompt variants for a chosen dimension.")'
    )

    return "\n\n\n".join(parts) + "\n"


def main() -> None:
    global _progress_bar
    total_units = count_translatable_units()
    _progress_bar = tqdm(total=total_units, desc="Translating de -> en", unit="item")

    dimensions_en = {code: translate_dimension(spec) for code, spec in de.DIMENSIONS.items()}
    variants_en = {vid: translate_variant(spec) for vid, spec in de.VARIANTS.items()}
    general_scoring_rules_en = t_bulleted_block(de.GENERAL_SCORING_RULES)
    standard_task_framing_en = t_paragraphs(de.STANDARD_TASK_FRAMING)
    minimal_task_framing_en = t_paragraphs(de.MINIMAL_TASK_FRAMING)
    minimal_scoring_rules_en = t_bulleted_block(de.MINIMAL_SCORING_RULES)
    labels_en = {key: t(value) for key, value in LABELS_DE.items()}

    _progress_bar.close()
    print(f"Items processed: {total_units} | actual translation API calls: {_call_count} (rest deduped via cache)")

    source = render_file(
        dimensions_en=dimensions_en,
        variants_en=variants_en,
        general_scoring_rules_en=general_scoring_rules_en,
        standard_task_framing_en=standard_task_framing_en,
        minimal_task_framing_en=minimal_task_framing_en,
        minimal_scoring_rules_en=minimal_scoring_rules_en,
        labels_en=labels_en,
    )
    OUTPUT_PATH.write_text(source, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print("Review the generated file before using it for LLM assessments.")


if __name__ == "__main__":
    main()
