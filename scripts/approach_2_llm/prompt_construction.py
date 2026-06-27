### to display all 5 prompt variants, run this script with the --display flag. The prompts will be saved as .txt files in data/assets/prompts.
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

Usage (from an assessment script):
    from prompt_construction import build_prompt

    prompt = build_prompt(
        dimension_code="d1_illness_beliefs",
        transcript=transcript_text,
        variant_id="v1_manual_rubric",
    )
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


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
            "Explore the patient's subjective explanatory model and/or concepts of healing; "
            "validate the patient's perspective; and flexibly adapt the therapeutic style. "
            "Relevant behaviors include summarizing the patient's explanatory model, asking how the patient explains the symptoms, "
            "asking about healing ideas, clarifying expectations for therapy, defining the therapist role when appropriate, "
            "addressing reservations or concerns, and showing understanding and sensitivity to doubts or objections."
        ),
        note=(
            "Neither symptom exploration alone nor validation alone is sufficient. "
            "The patient's understanding of illness and/or healing should be explicitly explored."
        ),
        checklist=[
            "Does the response explicitly explore the patient's subjective explanation of illness or healing?",
            "Does it validate or respectfully reflect the patient's perspective?",
            "Does it avoid dismissing the patient's explanatory model?",
            "Does it clarify expectations, doubts, or concerns about therapy when relevant?",
            "Is the response understandable, non-lecturing, and clinically appropriate?",
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
            "Approach the patient openly; reveal gaps in one's own knowledge; ask questions to better understand the patient's statements; "
            "clarify imprecise statements; openly admit lack of knowledge or uncertainty; emphasize the patient's expertise; "
            "check understanding with clarifying questions; and encourage the patient to take an active role in explaining the statement."
        ),
        note=(
            "This dimension concerns specific follow-up questions regarding patient statements, "
            "not general exploration of complaints."
        ),
        checklist=[
            "Does the response ask a specific follow-up question about the patient's statement?",
            "Does it show openness, curiosity, or willingness to learn from the patient?",
            "Does it acknowledge uncertainty or avoid pretending to already know?",
            "Does it encourage the patient to explain the statement in more detail?",
            "Does it avoid dismissing, lecturing, judging, or giving premature advice?",
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
            "Suggest whether cultural or contextual factors such as migration, language, discrimination, religion, or external circumstances "
            "could influence the patient's complaints, cooperation, or therapy. Relevant behaviors include asking whether cultural/contextual conditions "
            "could predispose to, trigger, or maintain complaints; openly addressing differences between patient and therapist as potential influences; "
            "using culture or belief as a resource/protective factor; asking about stress or internal conflicts from cultural differences; "
            "and developing hypotheses together with the patient."
        ),
        note=(
            "The goal is to understand the personal significance of cultural and contextual influences as possible stressors or protective factors and to integrate them constructively. "
            "This dimension requires proactive suggestions rather than only general exploration."
        ),
        checklist=[
            "Does the response explicitly mention a cultural, contextual, migration-related, language-related, religious, or discrimination-related factor?",
            "Does it suggest, without forcing, that this factor could influence symptoms, distress, cooperation, or therapy?",
            "Does it invite the patient to reflect on the personal meaning of this factor?",
            "Does it avoid leading, stereotyping, or overgeneralizing based on culture?",
            "Is the response validating and clinically appropriate?",
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
            "Relevant behaviors include validating the difficult situation, validating obligation or guilt toward the family, validating the importance of family, "
            "asking how the family deals with the problem, encouraging the patient to generate possible solutions in their environment, "
            "taking the perspective of relevant people, identifying important reference persons, and using circular questions."
        ),
        note=(
            "The patient's environment should be included. This dimension is not mainly about the patient's own individual perspective."
        ),
        checklist=[
            "Does the response include the family or social environment in understanding the problem?",
            "Does it validate the patient's family-related obligation, guilt, loyalty, conflict, or burden?",
            "Does it ask about family members' perspectives, reactions, or ways of dealing with the problem?",
            "Does it encourage collaborative reflection or possible solutions without giving direct advice?",
            "Does it avoid dismissing the family's importance or pushing the patient toward a simplistic decision?",
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
            "Relevant behaviors include validating the topic's importance, validating the patient's openness, highlighting strengths or resources, "
            "summarizing and reflecting the patient's viewpoint, exploring what the patient means, avoiding judgments of right or wrong, "
            "addressing contradictory statements, and respectfully setting boundaries when needed."
        ),
        note=(
            "The focus is on the patient's perspective. This is distinct from the family-system dimension."
        ),
        checklist=[
            "Does the response validate that the patient's viewpoint is important or meaningful to them?",
            "Does it explore the viewpoint without judging it as right or wrong?",
            "Does it reflect or summarize the patient's perspective accurately?",
            "Does it invite further dialogue or clarification about the viewpoint?",
            "Does it avoid dismissive, corrective, moralizing, or overly certain statements?",
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
# Five prompt variants
#
# Templates use Python str.format() placeholders, e.g. {transcript}.
# Literal JSON braces in the output schema are escaped as {{ }}.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "v1_manual_rubric": {
        "name": "Manual rubric",
        "template": """\
{common_task_framing}

Target dimension:
{dimension_name}

Instruction:
{dimension_instruction}

Rating criteria:
{dimension_criteria}

Important dimension-specific note:
{dimension_note}

General scoring rules:
{general_scoring_rules}

Transcript:
{transcript}

Return only valid JSON with exactly these keys. The value of "score" must be an integer 0, 1, 2, 3, or 4:
{{
  "score": 0,
  "brief_rationale": "...",
  "deduction_flags": []
}}
""",
    },
    "v2_manual_and_anchor_examples": {
        "name": "Manual rubric + anchor examples",
        "template": """\
{common_task_framing}

Target dimension:
{dimension_name}

Instruction:
{dimension_instruction}

Rating criteria:
{dimension_criteria}

Important dimension-specific note:
{dimension_note}

General scoring rules:
{general_scoring_rules}

Anchor examples from the rating manual:
{anchor_examples}

Transcript:
{transcript}

Evaluate the transcript by comparing it with the rating criteria and anchor examples.

Return only valid JSON with exactly these keys. The value of "score" must be an integer 0, 1, 2, 3, or 4:
{{
  "score": 0,
  "brief_rationale": "...",
  "deduction_flags": []
}}
""",
    },
    "v3_checklist": {
        "name": "Checklist-based structured rating",
        "template": """\
{common_task_framing}

Target dimension:
{dimension_name}

Instruction:
{dimension_instruction}

Rating criteria:
{dimension_criteria}

Important dimension-specific note:
{dimension_note}

Use this checklist before assigning the score:
{checklist}

General scoring rules:
{general_scoring_rules}

Transcript:
{transcript}

Return only valid JSON with exactly these keys. The value of "score" must be an integer 0, 1, 2, 3, or 4:
{{
  "checklist": {{
    "1": "yes/partly/no",
    "2": "yes/partly/no",
    "3": "yes/partly/no",
    "4": "yes/partly/no",
    "5": "yes/partly/no"
  }},
  "score": 0,
  "brief_rationale": "...",
  "deduction_flags": []
}}
""",
    },
    "v4_feedback_before_score": {
        "name": "Feedback before score",
        "template": """\
You are a fair and consistent evaluator of therapist responses in a behavioral test of transcultural competence.

Base your evaluation strictly on the transcript and the target-dimension rubric below.
Do not infer missing patient background or the hidden video prompt.
Do not evaluate dimensions other than the target dimension.

Target dimension:
{dimension_name}

Instruction:
{dimension_instruction}

Rubric and criteria:
{dimension_criteria}

Important dimension-specific note:
{dimension_note}

General scoring rules:
{general_scoring_rules}

Transcript:
{transcript}

First provide concise rubric-based feedback. Then assign the final score.

Return only valid JSON with exactly these keys. The value of "score" must be an integer 0, 1, 2, 3, or 4:
{{
  "feedback": "...",
  "score": 0,
  "deduction_flags": []
}}
""",
    },
    "v5_score_only": {
        "name": "Strict score-only",
        "template": """\
Rate the following therapist transcript for one target dimension of transcultural competence.

Use only the rating manual criteria below. Do not infer missing patient background or the hidden video prompt.

Target dimension:
{dimension_name}

Instruction:
{dimension_instruction}

Rating criteria:
{dimension_criteria}

Important dimension-specific note:
{dimension_note}

General scoring rules:
{general_scoring_rules}

Transcript:
{transcript}

Return only valid JSON with exactly this key. The value of "score" must be an integer 0, 1, 2, 3, or 4:
{{
  "score": 0
}}
""",
    },
}


# ---------------------------------------------------------------------------
# Prompt construction
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


def build_prompt(dimension_code: str, transcript: str, variant_id: str) -> str:
    """Construct one rendered prompt from plain variables.

    Args:
        dimension_code: One of the keys in DIMENSIONS, e.g. "d1_illness_beliefs".
        transcript: The participant's transcript text to be rated.
        variant_id: One of the keys in PROMPT_TEMPLATES, e.g. "v1_manual_rubric".
    """
    if dimension_code not in DIMENSIONS:
        valid = ", ".join(DIMENSIONS)
        raise ValueError(f"Unknown dimension code {dimension_code!r}. Valid dimension codes: {valid}")
    if variant_id not in PROMPT_TEMPLATES:
        valid = ", ".join(PROMPT_TEMPLATES)
        raise ValueError(f"Unknown prompt variant {variant_id!r}. Valid variants: {valid}")

    spec = DIMENSIONS[dimension_code]
    values = {
        "common_task_framing": COMMON_TASK_FRAMING.strip(),
        "dimension_code": spec.code,
        "dimension_name": spec.name,
        "dimension_instruction": spec.instruction,
        "dimension_criteria": spec.criteria,
        "dimension_note": spec.note,
        "general_scoring_rules": GENERAL_SCORING_RULES.strip(),
        "anchor_examples": format_anchor_examples(spec.anchor_examples),
        "checklist": format_checklist(spec.checklist),
        "transcript": "" if transcript is None else str(transcript).strip(),
    }

    template = PROMPT_TEMPLATES[variant_id]["template"]
    return template.format(**values).strip()


# Literal placeholder tokens (not real dimension content) used to preview
# template structure without tying the preview to any one dimension.
PLACEHOLDER_VALUES: dict[str, str] = {
    "common_task_framing": COMMON_TASK_FRAMING.strip(),
    "dimension_code": "{dimension_code}",
    "dimension_name": "{dimension_name}",
    "dimension_instruction": "{dimension_instruction}",
    "dimension_criteria": "{dimension_criteria}",
    "dimension_note": "{dimension_note}",
    "general_scoring_rules": GENERAL_SCORING_RULES.strip(),
    "anchor_examples": "{anchor_examples}",
    "checklist": "{checklist}",
    "transcript": "{transcript}",
}


def render_template_with_placeholders(variant_id: str) -> str:
    """Render one variant with its variable fields left as literal placeholders."""
    if variant_id not in PROMPT_TEMPLATES:
        valid = ", ".join(PROMPT_TEMPLATES)
        raise ValueError(f"Unknown prompt variant {variant_id!r}. Valid variants: {valid}")

    template = PROMPT_TEMPLATES[variant_id]["template"]
    return template.format(**PLACEHOLDER_VALUES).strip()


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data" / "assets" / "prompts"


def display_all_prompt_variants(output_dir: Path = DEFAULT_PROMPTS_DIR) -> list[Path]:
    """Render all 5 prompt variants with placeholders and write each to its own .txt file.

    The dimension-specific fields and the transcript are left as literal
    placeholders (e.g. "{dimension_name}", "{transcript}") rather than filled
    in with one dimension's content, since those are supplied per-call by
    whatever script calls build_prompt().

    Args:
        output_dir: Directory where the 5 variant text files are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for variant_id in PROMPT_TEMPLATES:
        prompt = render_template_with_placeholders(variant_id)
        file_path = output_dir / f"{variant_id}.txt"
        file_path.write_text(prompt, encoding="utf-8")
        written.append(file_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge prompt construction utilities.")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Render all 5 prompt variants and save them as .txt files in data/assets/prompts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.display:
        for path in display_all_prompt_variants():
            print(f"Saved: {path}")
    else:
        print("Nothing to do. Pass --display to render the 5 prompt variants to data/assets/prompts.")
