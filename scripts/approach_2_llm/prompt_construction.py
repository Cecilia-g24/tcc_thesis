### to preview the 5 prompt variants for one dimension, run this script with the --display
### flag and enter a dimension code when prompted. The prompts will be saved as .txt files
### in data/assets/prompts/<dimension_code>/.
# python prompt_construction.py --display

"""
LLM-as-judge prompt templates for transcultural competence ratings.

This module only constructs prompts. It does not call any LLM API and does not
perform the actual assessment.

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

Each variant is defined as a single-factor change relative to V1_full_manual_baseline
(see VariantSpec / VARIANTS below), not as an independently written template. This keeps
the ablation clean: rendering V1 and any other variant for the same dimension differs
in exactly the one toggle that variant is meant to test, and nowhere else.

Usage (from an assessment script):
    from prompt_construction import build_prompt

    prompt = build_prompt(
        dimension_code="d1_illness_beliefs",
        transcript=transcript_text,
        variant_id="V1_full_manual_baseline",
    )
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Shared rating information from the rating manual
# ---------------------------------------------------------------------------

GENERAL_SCORING_RULES = """\
- Use exactly one integer score from 0 to 4.
- Valid scores are only 0, 1, 2, 3, and 4. Do not use decimals.
- 0 = inadequate implementation.
- 1 = very limited implementation.
- 2 = task fulfilled.
- 3 = good implementation.
- 4 = excellent implementation.
- A score of 2 means the task is fulfilled.
- A high score does not require a perfect response.
- A high score does not require every performance criterion to be met.
- Even a few well-executed performance criteria can be sufficient for a high score.
- Deduct points if the task is only partially fulfilled or not fulfilled.
- Also consider basic communication quality: the response should be appropriate and easy for the patient to understand.
- Deduct points for unclear, ambiguous, leading, lecturing, derogatory, incomprehensible, very long/convoluted responses, or many questions in succession.
- Treat transcription errors leniently if the meaning is still understandable.
- If the transcript content is incomprehensible, assign a low score.
"""

COMMON_TASK_FRAMING = """\
You are acting as a trained rater for a behavioral test of transcultural competence in psychotherapy.

The participant watched a short video clip of a challenging transcultural psychotherapy situation and responded as the therapist.
You will receive a transcript of the participant's response and the rating criteria for one target dimension.

Important constraints:
- Base your evaluation only on the transcript and the rating manual criteria provided in this prompt.
- Do not use the human rater scores, even if they are available elsewhere in the dataset.
- Do not infer or invent the original video clip.
- Do not assume patient background information that is not included in the transcript.
- Rate only the target dimension specified below.
"""


@dataclass(frozen=True)
class DimensionSpec:
    code: str
    name: str
    instruction: str
    criteria: str
    note: str
    checklist: list[str]
    anchor_examples: dict[int, list[str]]


DIMENSIONS: dict[str, DimensionSpec] = {
    "d1_illness_beliefs": DimensionSpec(
        code="d1_illness_beliefs",
        name="Inquiring about subjective ideas about illness and healing",
        instruction="Explore how the patients explain their problems to themselves.",
        criteria=(
            "Explore the patient's subjective explanatory model and/or concepts of healing; validate the patient's perspective; "
            "and flexibly adapt the therapeutic style accordingly. Manual performance criteria include:\n"
            "- Summarizing the patient's explanatory model in the therapist's own words.\n"
            "- Asking about the patient's explanatory model, such as how the patient explains the symptoms or what the patient believes caused the problem.\n"
            "- Asking about the patient's ideas about healing or what the patient thinks could help.\n"
            "- Clarifying the patient's expectations for therapy or how the therapist can support the patient.\n"
            "- Defining the therapist's role when appropriate.\n"
            "- Addressing obvious reservations or concerns from the patient.\n"
            "- Showing understanding and sensitivity to the patient's doubts or objections.\n"
            "- Validating the patient's openness in expressing doubts or concerns.\n"
            "- Validating the patient's explanatory model."
        ),
        note=(
            "Neither symptom exploration alone nor validation alone is sufficient. "
            "The patient's understanding of illness and/or healing should be explicitly explored."
        ),
        checklist=[
            "Does the response summarize the patient's explanatory model in the therapist's own words?",
            "Does the response ask about the patient's explanatory model, such as how the patient explains the symptoms or what the patient believes caused the problem?",
            "Does the response ask about the patient's ideas about healing or what the patient thinks could help?",
            "Does the response clarify the patient's expectations for therapy or how the therapist can support the patient?",
            "Does the response define the therapist's role when appropriate?",
            "Does the response address obvious reservations or concerns from the patient?",
            "Does the response show understanding and sensitivity to the patient's doubts or objections?",
            "Does the response validate the patient's openness in expressing doubts or concerns?",
            "Does the response validate the patient's explanatory model?",
        ],
        anchor_examples={
            4: [
                "Yes, I understand you realize there might be a connection between the estrangement from your family and the symptoms you're describing. How do you think these things could be related?",
                "They doubt that therapy can help them because they believe only God can help them. I can imagine it's frustrating for them to sit in therapy and question what it's all supposed to achieve. I wonder what their expectations are and how I can still help them.",
            ],
            3: [
                "You say your escape could be a punishment from God, or the symptoms that have arisen as a result. What do you mean by that? What would be God's punishment? What would He be punishing you for?",
                "That means you feel guilty, right? What exactly do you think you were punished for?",
            ],
            2: [
                "Putting myself in your shoes, I can imagine it must be quite exhausting to constantly go into more detail and explain even more. However, I'm just the specialist with the expertise. You are your own expert, and only you can answer certain questions about yourself.",
                "I understand your frustration, but unfortunately I can't read your mind. So we need to work together to figure out how we can help you.",
            ],
            1: [
                "I can well understand that this makes you feel very burdened and that leaving your parents in this situation is an unpleasant feeling for you.",
                "You have the impression that you can't get yourself out of this on your own, right? You think you need outside support.",
            ],
            0: [
                "I think you're seeing a connection where there might not be one. Physical ailments don't just arise simply because someone has no contact with their family.",
                "It sounds like you're getting in your own way. You need to accept that it's over. If you only focus on your loneliness, nothing will change.",
            ],
        },
    ),
    "d2_lack_of_knowledge": DimensionSpec(
        code="d2_lack_of_knowledge",
        name="Proactively dealing with not knowing",
        instruction="Encourage the patients to explain their statement in more detail.",
        criteria=(
            "Approach the patient openly, reveal gaps in the therapist's own knowledge, and ask questions to better understand the patient's statements. "
            "Manual performance criteria include:\n"
            "- Clarifying imprecise patient statements, for example by asking what exactly the patient means.\n"
            "- Openly admitting the therapist's own lack of knowledge when relevant.\n"
            "- Openly acknowledging the therapist's own uncertainty when relevant.\n"
            "- Emphasizing the patient's expertise about their own experience or meaning.\n"
            "- Checking the therapist's understanding by asking clarifying questions.\n"
            "- Appropriately encouraging the patient to take an active role, for example by giving another example or explaining the statement further."
        ),
        note=(
            "This dimension concerns specific follow-up questions regarding patient statements, "
            "not general exploration of complaints."
        ),
        checklist=[
            "Does the response clarify imprecise patient statements, for example by asking what exactly the patient means?",
            "Does the response openly admit the therapist's own lack of knowledge when relevant?",
            "Does the response openly acknowledge the therapist's own uncertainty when relevant?",
            "Does the response emphasize the patient's expertise about their own experience or meaning?",
            "Does the response check the therapist's understanding by asking clarifying questions?",
            "Does the response appropriately encourage the patient to take an active role, for example by giving another example or explaining the statement further?",
        ],
        anchor_examples={
            4: [
                "You said you can't talk to your family about the therapy here. What exactly do you mean by that? What is your family's attitude towards psychotherapy?",
                "You feel that your community might speak ill of you and view you as weak if they find out about your daughter's husband. When you say 'weak man,' what exactly do you mean by that?",
            ],
            3: [
                "I'm just curious how things generally are in your culture. Are men who don't adhere to the prevailing traditions considered weak? Or how should I understand that?",
                "It sounds like you're putting a lot of pressure on yourself regarding the language. Unfortunately, I don't know what the best course of action is for you right now. Perhaps we could discuss together what would be best for you at the moment. To do that, I'd like to understand your situation a little better.",
            ],
            2: [
                "How did you come to the conclusion that I have so little understanding or am unable to grasp your situation and give you appropriate advice? Consider what I might have said incorrectly, or are there any indications that lead you to this conclusion?",
                "You see, I would like to understand exactly what this means for you and your family, in order to better understand the situation and then, of course, to be happy to help you further.",
            ],
            1: [
                "I can well imagine that it was very difficult for you to leave your country and start completely anew in a foreign one. I would like to know what motivated you to leave your country back then. There are, of course, many factors that play a role in such a decision.",
                "I can completely understand these concerns. I can tell you that therapy can be very helpful for the problems you described, namely feeling stressed or tense.",
            ],
            0: [
                "Okay, so that means it bothers you that your daughter is going her own way and has her own will. And there's nothing you can do about it.",
                "I have been working as a therapist for many years and can help you, even though I don't come from your culture. That's irrelevant.",
            ],
        },
    ),
    "d3_cultural_factors": DimensionSpec(
        code="d3_cultural_factors",
        name="Consideration of cultural and contextual factors",
        instruction="Discuss with the patients whether their culture or external circumstances, such as migration or discrimination, could influence their symptoms or therapy.",
        criteria=(
            "Suggest whether cultural or contextual factors such as migration, language, discrimination, religion, or external circumstances could influence "
            "the patient's complaints, cooperation, or therapy. Manual performance criteria include:\n"
            "- Asking whether cultural or contextual conditions, such as migration, language, or discrimination, could predispose to, trigger, or maintain the complaints.\n"
            "- Openly addressing differences between patient and therapist as a possible influence on therapy.\n"
            "- Using culture or belief as a resource or protective factor when relevant.\n"
            "- Asking about stress or internal conflicts resulting from cultural differences.\n"
            "- Developing hypotheses together with the patient about whether culture, belief, migration, language, discrimination, religion, or other external circumstances influence complaints, cooperation, or therapy."
        ),
        note=(
            "The goal is to understand the personal significance of cultural and contextual influences as possible stressors or protective factors and to integrate them constructively. "
            "This dimension requires proactive suggestions rather than only general exploration."
        ),
        checklist=[
            "Does the response ask whether cultural or contextual conditions, such as migration, language, or discrimination, could predispose to, trigger, or maintain the complaints?",
            "Does the response openly address differences between patient and therapist as a possible influence on therapy?",
            "Does the response use culture or belief as a resource or protective factor when relevant?",
            "Does the response ask about stress or internal conflicts resulting from cultural differences?",
            "Does the response develop hypotheses together with the patient about whether culture, belief, migration, language, discrimination, religion, or other external circumstances influence complaints, cooperation, or therapy?",
        ],
        anchor_examples={
            4: [
                "First of all, I'm sorry to hear that you've experienced this. The idea that you shouldn't study and pursue a leadership position could therefore be related to your experiences of racism. I wonder if these experiences have weakened your self-confidence.",
                "It sounds like you're in a difficult situation, caught between two worlds. I wonder if 'two worlds' also means you're caught between two cultures. Could it be that the differing expectations and demands of these two cultures are taking their toll on you?",
            ],
            3: [
                "Do you experience a connection between your origins and the feeling of never fully belonging?",
                "Your ideas about who belongs at university and who doesn't might also have a cultural basis. This could be re-examined, exploring how these ideas developed in your life, at which stages this occurred, and what role your migration background plays.",
            ],
            2: [
                "I'm sorry to hear that they feel like failures. It's understandable, considering the migration process and the added challenge of learning the language; I can imagine that's very difficult. But I also think migration plays a role here.",
                "That sounds like a huge responsibility. I'd be interested to know how you developed the conviction that you are obligated to provide for your family. Is that something you know from your own family?",
            ],
            1: [
                "Okay, so you have an idea why they might feel excluded?",
                "Why do you think that people like you with a migration background don't belong at university? If you look at the university itself, you'll see that a fairly large proportion of students have a migration background. How come you think that way?",
            ],
            0: [
                "But if you have friends, then you're not really excluded. Maybe you're imagining it or simply expecting too much. People in Germany are just a bit more reserved. That has nothing to do with where you come from.",
                "Perhaps you simply need to learn not to take everything so personally; otherwise it will be difficult to cope with everyday life.",
            ],
        },
    ),
    "d4_family_system": DimensionSpec(
        code="d4_family_system",
        name="Inclusion of the family system",
        instruction="Explore the described situation and encourage the patient to change her perspective.",
        criteria=(
            "Explore the patient's complaints within the family context and work collaboratively toward possible solutions without giving advice. "
            "Manual performance criteria include:\n"
            "- Validating the patient's difficult situation.\n"
            "- Validating the patient's feeling of obligation or guilt toward the family.\n"
            "- Validating the importance of family for the patient.\n"
            "- Asking about the family's approach to the problem or how the family usually deals with such problems.\n"
            "- Encouraging the patient to independently generate possible solutions in their environment.\n"
            "- Inviting perspective-taking from relevant people in the patient's family or social environment.\n"
            "- Identifying relevant reference persons in the family or social environment.\n"
            "- Using circular questions about how relevant family members or other people might feel, think, or react."
        ),
        note=(
            "The patient's environment should be included. This dimension is not mainly about the patient's own individual perspective."
        ),
        checklist=[
            "Does the response validate the patient's difficult situation?",
            "Does the response validate the patient's feeling of obligation or guilt toward the family?",
            "Does the response validate the importance of family for the patient?",
            "Does the response ask about the family's approach to the problem or how the family usually deals with such problems?",
            "Does the response encourage the patient to independently generate possible solutions in their environment?",
            "Does the response invite perspective-taking from relevant people in the patient's family or social environment?",
            "Does the response identify relevant reference persons in the family or social environment?",
            "Does the response use circular questions about how relevant family members or other people might feel, think, or react?",
        ],
        anchor_examples={
            4: [
                "Given the sacrifices your parents made, it's understandable that you feel bound to your family's values. This makes it very difficult for you to find your own way. How do you think your family would react if they knew about your worries?",
                "It sounds like you're in a pretty difficult situation right now, putting your own needs aside for the sake of your family. How do other women in your circle manage being a good wife?",
            ],
            3: [
                "I understand that you are very upset right now. Did you speak to your mother about it in that situation?",
                "Do you think you need to protect your family from your decisions?",
            ],
            2: [
                "I understand that you have a desire for freedom and would like to decide for yourself how you live your life, but at the same time you're worried about losing contact with your family if you don't meet their expectations. Have I understood that correctly?",
                "Okay, so that means they don't feel seen by their family?",
            ],
            1: [
                "I can well understand that you're reconsidering the consequences of your move for your family. However, we should also consider the possibilities of looking at things from a different perspective and then making an independent decision about which direction things might take.",
                "At this moment, it's important to look inward and consider what feelings your husband's expectations evoke in you. Of course, we also need to examine whether you can and want to always meet these expectations, and how willing you are to change.",
            ],
            0: [
                "You really need to sort this out now. Either you stand by your partner or you stop stringing everyone along. Secrecy gets you nowhere.",
                "You're an adult, and what your family thinks shouldn't bother you so much. At some point, you have to live your own life.",
            ],
        },
    ),
    "d5_ambiguity_tolerance": DimensionSpec(
        code="d5_ambiguity_tolerance",
        name="Ambiguity tolerance",
        instruction="Invite the patients to a dialogue about the aforementioned viewpoint.",
        criteria=(
            "Validate that the topic is important to the patient, show openness to the patient's perspective, and explore the perspective without judgment. "
            "Manual performance criteria include:\n"
            "- Validating the importance or significance of the topic for the patient.\n"
            "- Validating the patient's openness.\n"
            "- Highlighting strengths or resources expressed in the patient's statement.\n"
            "- Summarizing and reflecting back the patient's viewpoint.\n"
            "- Exploring the mentioned perspective, for example by asking what the patient means.\n"
            "- Summarizing the patient's statement in the therapist's own words.\n"
            "- Avoiding judgments of the patient's viewpoint as right or wrong.\n"
            "- Addressing contradictory statements when relevant.\n"
            "- Respectfully setting the therapist's own boundaries when needed."
        ),
        note=(
            "The focus is on the patient's perspective. This is distinct from the family-system dimension."
        ),
        checklist=[
            "Does the response validate the importance or significance of the topic for the patient?",
            "Does the response validate the patient's openness?",
            "Does the response highlight strengths or resources expressed in the patient's statement?",
            "Does the response summarize and reflect back the patient's viewpoint?",
            "Does the response explore the mentioned perspective, for example by asking what the patient means?",
            "Does the response summarize the patient's statement in the therapist's own words?",
            "Does the response avoid judging the patient's viewpoint as right or wrong?",
            "Does the response address contradictory statements when relevant?",
            "Does the response respectfully set the therapist's own boundaries when needed?",
        ],
        anchor_examples={
            4: [
                "I can well understand that tradition and family are very important to you. Why, then, would you describe fellow students who are now atheists as disrespectful? Could you explain that in more detail?",
                "Yes, I gather that religion is very important to you. It's difficult for you to then be confronted with other people for whom it isn't so important. What exactly is it about your religion that you value so much and that is particularly important to you?",
            ],
            3: [
                "Could you explain that in more detail? What exactly do you find disrespectful about it?",
                "To me, it sounds like she's quite torn and engaged in an inner struggle. I'd like to take a closer look at both sides of her voice with her.",
            ],
            2: [
                "If you're saying that you need to positively influence the angels for your problems to disappear, is there also an opposite? I'd like to discuss with you in more detail how important it can be to take action for yourself and help yourself.",
                "Now that we are talking here together, we can try to consider whether there are any other factors that might be contributing to her mental health problems and whether there are any other factors that could help her feel better.",
            ],
            1: [
                "Okay, so you've already made it very clear where you think this is coming from. Have you considered whether there might be another trigger or other cause for these symptoms?",
                "So, your explanation is that you are cursed; could there also be another explanation for the situation you are in, for how you are feeling right now?",
            ],
            0: [
                "If you believe you are surrounded by a curse, then don't consider all the facts. We should also consider other facts.",
                "A curse? That's just superstition. Your symptoms have medical causes. You should accept that.",
            ],
        },
    ),
}


# ---------------------------------------------------------------------------
# Five prompt variants, defined as single-factor toggles relative to
# V1_full_manual_baseline (see the module docstring). A shared rendering
# function (build_prompt) assembles sections in a fixed order and includes
# or reorders a section only according to these toggles, so that comparing
# V1's rendering to any other variant's rendering isolates exactly one change.
# ---------------------------------------------------------------------------

CriteriaFormat = Literal["bullets", "checklist"]


@dataclass(frozen=True)
class VariantSpec:
    id: str
    name: str
    why_chosen: str
    include_anchors: bool
    include_rationale: bool
    rationale_before_score: bool = False
    criteria_format: CriteriaFormat = "bullets"


VARIANTS: dict[str, VariantSpec] = {
    "V1_full_manual_baseline": VariantSpec(
        id="V1_full_manual_baseline",
        name="Full manual baseline",
        why_chosen="Human-rater-equivalent baseline",
        include_anchors=True,
        include_rationale=True,
        rationale_before_score=False,
        criteria_format="bullets",
    ),
    "V2_no_anchors": VariantSpec(
        id="V2_no_anchors",
        name="Manual without anchor examples",
        why_chosen="Anchor examples help?",
        include_anchors=False,
        include_rationale=True,
        rationale_before_score=False,
        criteria_format="bullets",
    ),
    "V3_no_rationale": VariantSpec(
        id="V3_no_rationale",
        name="Manual without rationale output",
        why_chosen="Does requiring an explicit rationale affect scoring?",
        include_anchors=True,
        include_rationale=False,
        rationale_before_score=False,
        criteria_format="bullets",
    ),
    "V4_evidence_before_score": VariantSpec(
        id="V4_evidence_before_score",
        name="Rationale before score",
        why_chosen="Output format helps? (score then justification vs. reasoning then score)",
        include_anchors=True,
        include_rationale=True,
        rationale_before_score=True,
        criteria_format="bullets",
    ),
    "V5_structured_checklist": VariantSpec(
        id="V5_structured_checklist",
        name="Performance criteria restructured as checklist",
        why_chosen="Decomposition helps?",
        include_anchors=True,
        include_rationale=True,
        rationale_before_score=False,
        criteria_format="checklist",
    ),
}


# ---------------------------------------------------------------------------
# Formatting helpers for dimension-specific content
# ---------------------------------------------------------------------------

def format_anchor_examples(anchor_examples: dict[int, list[str]]) -> str:
    """Format score-level anchor examples for insertion into a prompt."""
    lines: list[str] = []
    for score in sorted(anchor_examples.keys(), reverse=True):
        lines.append(f"Score {score} examples:")
        for ex in anchor_examples[score]:
            lines.append(f"- {ex}")
    return "\n".join(lines)


def format_checklist(checklist: list[str]) -> str:
    """Format the checklist as a numbered list."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(checklist, start=1))


# ---------------------------------------------------------------------------
# Section builders
#
# Each function returns one section of the prompt (heading + content) or an
# empty string if the section is toggled off for the given variant. build_prompt
# joins the non-empty sections with blank lines, so a variant's rendering is
# the baseline's rendering with exactly its declared toggles applied.
# ---------------------------------------------------------------------------

def _section_target(spec: DimensionSpec) -> str:
    return f"Target dimension:\n{spec.name}"


def _section_instruction(spec: DimensionSpec) -> str:
    return f"Instruction:\n{spec.instruction}"


def _section_criteria(spec: DimensionSpec, variant: VariantSpec) -> str:
    if variant.criteria_format == "checklist":
        criteria_intro = spec.criteria.split("Manual performance criteria include:", 1)[0].strip()
        return (
            "Rating criteria (checklist form; same manual performance criteria, restructured):\n"
            f"Use the following checklist to assess whether the response meets the same criteria described in the manual: {criteria_intro}\n"
            "Use the checklist to guide attention, but do not mechanically count items. Assign the final 0-4 score holistically according to the rating manual.\n\n"
            f"{format_checklist(spec.checklist)}"
        )
    return f"Rating criteria:\n{spec.criteria}"


def _section_note(spec: DimensionSpec) -> str:
    return f"Important dimension-specific note:\n{spec.note}"


def _section_anchors(spec: DimensionSpec, variant: VariantSpec) -> str:
    if not variant.include_anchors:
        return ""
    return f"Anchor examples from the rating manual:\n{format_anchor_examples(spec.anchor_examples)}"


def _section_scoring_rules() -> str:
    return f"General scoring rules:\n{GENERAL_SCORING_RULES.strip()}"


def _section_transcript(transcript: str) -> str:
    return f"Transcript:\n{transcript}"


def _section_output(variant: VariantSpec) -> str:
    if not variant.include_rationale:
        return (
            "Return only valid JSON with exactly this key. "
            "The value of \"score\" must be an integer 0, 1, 2, 3, or 4:\n"
            "{\n"
            '  "score": 0\n'
            "}"
        )
    if variant.rationale_before_score:
        return (
            "First provide a brief rationale that cites specific evidence from the transcript. "
            "Then assign the final score consistent with that rationale.\n\n"
            "Return only valid JSON with exactly these keys, in this order. "
            "The value of \"score\" must be an integer 0, 1, 2, 3, or 4:\n"
            "{\n"
            '  "brief_rationale": "...",\n'
            '  "score": 0\n'
            "}"
        )
    return (
        "Return only valid JSON with exactly these keys. "
        "The value of \"score\" must be an integer 0, 1, 2, 3, or 4:\n"
        "{\n"
        '  "score": 0,\n'
        '  "brief_rationale": "..."\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(dimension_code: str, transcript: str, variant_id: str) -> str:
    """Construct one rendered prompt from plain variables.

    Args:
        dimension_code: One of the keys in DIMENSIONS, e.g. "d1_illness_beliefs".
        transcript: The participant's transcript text to be rated.
        variant_id: One of the keys in VARIANTS, e.g. "V1_full_manual_baseline".
    """
    if dimension_code not in DIMENSIONS:
        valid = ", ".join(DIMENSIONS)
        raise ValueError(f"Unknown dimension code {dimension_code!r}. Valid dimension codes: {valid}")
    if variant_id not in VARIANTS:
        valid = ", ".join(VARIANTS)
        raise ValueError(f"Unknown prompt variant {variant_id!r}. Valid variants: {valid}")

    spec = DIMENSIONS[dimension_code]
    variant = VARIANTS[variant_id]
    transcript_text = "" if transcript is None else str(transcript).strip()

    sections = [
        COMMON_TASK_FRAMING.strip(),
        _section_target(spec),
        _section_instruction(spec),
        _section_criteria(spec, variant),
        _section_note(spec),
        _section_anchors(spec, variant),
        _section_scoring_rules(),
        _section_transcript(transcript_text),
        _section_output(variant),
    ]
    return "\n\n".join(section for section in sections if section).strip()


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data" / "assets" / "prompts"


def display_variants_for_dimension(dimension_code: str, output_dir: Path = DEFAULT_PROMPTS_DIR) -> list[Path]:
    """Render all 5 prompt variants for one dimension and write each to its own .txt file.

    The transcript field is left as a literal placeholder since it is supplied
    per-call by whatever script calls build_prompt().

    Args:
        dimension_code: One of the keys in DIMENSIONS, e.g. "d1_illness_beliefs".
        output_dir: Directory where the 5 variant text files are written, under
            a subfolder named after the dimension code.
    """
    if dimension_code not in DIMENSIONS:
        valid = ", ".join(DIMENSIONS)
        raise ValueError(f"Unknown dimension code {dimension_code!r}. Valid dimension codes: {valid}")

    dimension_dir = output_dir / dimension_code
    dimension_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for variant_id in VARIANTS:
        prompt = build_prompt(dimension_code, transcript="{transcript}", variant_id=variant_id)
        file_path = dimension_dir / f"{variant_id}.txt"
        file_path.write_text(prompt, encoding="utf-8")
        written.append(file_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge prompt construction utilities.")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Prompt for a dimension code, then render its 5 prompt variants and save "
        "them as .txt files under data/assets/prompts/<dimension_code>/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.display:
        valid_codes = ", ".join(DIMENSIONS)
        dimension_code = input(f"Enter dimension code ({valid_codes}): ").strip()
        for path in display_variants_for_dimension(dimension_code):
            print(f"Saved: {path}")
    else:
        print("Nothing to do. Pass --display to render the 5 prompt variants for a chosen dimension.")
