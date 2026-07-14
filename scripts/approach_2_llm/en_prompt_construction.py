### To preview the 6 prompt variants for every dimension, run this script with
### the --display flag. The prompts will be saved as .txt files in
### data/assets/en_assets/en_prompts/<dimension_code>/.
# python en_prompt_construction.py --display


"""
English-language LLM-as-rater prompt templates for transcultural competence ratings.

This is the English counterpart of de_prompt_construction.py. The dimension names,
instructions, criteria, notes, checklists, anchor examples, shared task framing,
scoring rules, and section labels are faithful, human-reviewed English translations
of the German prompt content.

The section-building architecture mirrors de_prompt_construction.py so that the two
modules remain easy to compare. One deliberate implementation difference is that
_section_criteria() splits the criteria text at its first newline rather than using a
German marker phrase. The first line is always the dimension-level criterion summary,
so this produces the same checklist structure without depending on language-specific
wording.

The dimension_code passed to build_prompt must be one of:
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
German and English prompt variants so that downstream parsing code works unchanged
regardless of prompt language.

Usage:
    from en_prompt_construction import build_prompt

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


GENERAL_SCORING_RULES = """- Use exactly one integer from 0 to 4 as the rating.
- Only the values 0, 1, 2, 3, and 4 are valid. Do not use decimal values.
- 0 = inadequately implemented.
- 2 = task fulfilled.
- 4 = excellently implemented.
- A rating of 2 means that the task has been fulfilled.
- The response does not need to be perfect to receive a high rating.
- Not all performance criteria need to be fulfilled for a high rating.
- Even individual criteria that are implemented well may be sufficient for a high rating.
- Deduct points if the task is only partially fulfilled or not fulfilled at all.
- In addition to the performance criteria, consider basic conversational skills: the response should be appropriate and easy for patients to understand.
- Deduct points for unclear, potentially confusing, leading, lecturing, derogatory, or incomprehensible statements; very long or convoluted statements; or many questions asked in succession.
- Treat transcription errors as leniently as possible. As long as a statement remains understandable, transcription errors should not affect the rating.
- If the transcript content is incomprehensible, assign a low rating.
"""


STANDARD_TASK_FRAMING = """You are a trained rater for a behavioral test of transcultural competence in psychotherapy.

You will receive a transcript of a therapeutic response and information about the dimension to be assessed.

Base your rating solely on the provided transcript and the information in this prompt. Do not assume information that is not included in the transcript."""


MINIMAL_TASK_FRAMING = """Rate the following therapeutic response with respect to the specified target dimension.

Base your rating solely on the transcript. Do not assume information that is not included in the transcript."""


MINIMAL_SCORING_RULES = """- Use exactly one integer from 0 to 4 as the rating.
- 0 = inadequately implemented.
- 2 = task fulfilled.
- 4 = excellently implemented.
"""


# ---------------------------------------------------------------------------
# Section labels used by the section-builder functions below
# ---------------------------------------------------------------------------


LABEL_TARGET = "Target dimension:"
LABEL_INSTRUCTION = "Instruction:"
LABEL_CRITERIA_BULLETS = "Evaluation criteria:"
LABEL_MANUAL_PERFORMANCE_INDICATORS = "The manual's performance indicators include:"
LABEL_CRITERIA_CHECKLIST_HEADER = (
    "Evaluation criteria (checklist format; the same manual criteria, restructured):"
)
LABEL_CRITERIA_CHECKLIST_INTRO_PREFIX = (
    "Use the following checklist to assess whether the response meets the same "
    "criteria described in the manual:"
)
LABEL_CRITERIA_CHECKLIST_GUIDANCE = (
    "Use the checklist to guide your attention, but do not count items mechanically. "
    "Assign the final rating from 0 to 4 holistically according to the rating manual."
)
LABEL_NOTE = "Important dimension-specific note:"
LABEL_ANCHORS = "Anchor examples from the rating manual:"
LABEL_ANCHOR_SCORE_PREFIX = "Examples for rating"
LABEL_SCORING_GENERAL = "General evaluation rules:"
LABEL_SCORING_MINIMAL = "Rating scale:"
LABEL_TRANSCRIPT = "Transcript:"
LABEL_OUTPUT_NO_RATIONALE = (
    'Return only valid JSON with exactly this key. The value of "score" must be an '
    "integer: 0, 1, 2, 3, or 4:"
)
LABEL_OUTPUT_RATIONALE_BEFORE_LEAD = (
    "First provide a brief rationale based on specific evidence from the transcript. "
    "Then assign the final rating, ensuring that it is consistent with the rationale."
)
LABEL_OUTPUT_RATIONALE_BEFORE_INSTR = (
    'Return only valid JSON with exactly these keys, in this order. The value of "score" '
    "must be an integer: 0, 1, 2, 3, or 4:"
)
LABEL_OUTPUT_RATIONALE_AFTER_INSTR = (
    'Return only valid JSON with exactly these keys. The value of "score" must be an '
    "integer: 0, 1, 2, 3, or 4:"
)


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
        name="Elicit subjective beliefs about illness and healing",
        instruction="Explore how patients themselves explain their problems.",
        criteria="""Explores the patient's subjective explanatory model and/or beliefs about healing, validates the patient's perspective, and flexibly adapts the therapeutic style.
- Summarizes explanatory models in their own words ("So you seem convinced that stress was a trigger.").
- Asks about explanatory models ("How do you explain your symptoms?", "Where do you think this problem comes from?").
- Asks about beliefs regarding healing ("What do you think could help you?").
- Clarifies expectations regarding therapy ("You probably have an idea of how I can support you.").
- Defines the therapist's role.
- Addresses obvious reservations or concerns expressed by the patient.
- Responds with understanding and sensitivity to doubts or objections ("I notice that you are still unsure whether therapy can help you at all.").
- Validates the patient's openness, such as their willingness to express doubts.
- Validates the patient's explanatory model.""",
        note=(
            "Neither exploring the symptoms nor validation alone is sufficient. The "
            "response must explicitly ask about the patient's own understanding of "
            "illness and their beliefs about healing."
        ),
        checklist=[
            "Does the response summarize the patient's explanatory model in its own words?",
            "Does the response ask about the patient's explanatory model, for example, how the patient explains the symptoms or where the patient believes the problem comes from?",
            "Does the response ask about the patient's beliefs regarding healing, for example, what the patient believes might help?",
            "Does the response clarify the patient's expectations regarding therapy or how the therapist can provide support?",
            "Does the response define the therapist's role?",
            "Does the response address any obvious reservations or concerns expressed by the patient?",
            "Does the response demonstrate understanding and sensitivity toward the patient's doubts or objections?",
            "Does the response validate the patient's openness in expressing doubts or concerns?",
            "Does the response validate the patient's explanatory model?",
        ],
        anchor_examples={
            4: [
                "Yes, I understand. You notice that there may be a connection between losing contact with your family and the symptoms you have listed. How do you think the two might be connected?",
                "You doubt that therapy can help you because only God can help you. I can imagine that it is frustrating to sit in therapy and question what any of this will achieve. I am wondering what your own ideas are and how I might nevertheless be able to help you.",
            ],
            3: [
                "You say that your flight, or the symptoms that developed, could be a punishment from God. What exactly do you mean by that? What would the punishment from God be, and what would he be punishing you for?",
                "That means you feel guilty, correct? What exactly do you think you were punished for?",
            ],
            2: [
                "If I put myself in your position, I can imagine that it is quite exhausting to go into more and more detail and keep telling more. However, I am only the specialist with professional knowledge. You are the expert on yourself, and only you can answer certain questions about yourself.",
                "I understand your frustration, but unfortunately I cannot see inside your mind. We therefore need to work together to find out how we can help you.",
            ],
            1: [
                "I can understand that this is very distressing for you and that leaving your parents behind in this situation feels unpleasant.",
                "You have the impression that you cannot get out of this on your own, correct? You think you need outside support.",
            ],
            0: [
                "I think you see a connection where there may not be one. Physical symptoms do not simply arise because you no longer have contact with your family.",
                "It sounds as though you are getting in your own way. You need to accept that it is over. If you focus only on your loneliness, nothing will change.",
            ],
        },
    ),
    "d2_lack_of_knowledge": DimensionSpec(
        code="d2_lack_of_knowledge",
        name="Proactively address gaps in knowledge",
        instruction="Encourage patients to explain their statement in greater detail.",
        criteria="""Engages openly with patients, acknowledges gaps in the therapist's own knowledge, and asks questions to better understand patients' statements.
- Clarifies vague patient statements ("What exactly do you mean by ...?", "When you say '...', what exactly do you mean by that term?").
- Openly acknowledges a lack of knowledge ("That is new to me; I would like to understand it better.").
- Openly acknowledges uncertainty ("Unfortunately, I also do not know what would be best right now; let us consider together what might be helpful for you.").
- Emphasizes the patient's expertise ("You are the expert ...").
- Checks the therapist's understanding by asking follow-up questions ("I want to make sure I understand you correctly.").
- Appropriately encourages the patient to take an active role ("It would help me greatly if you could give me another example.").""",
        note=(
            "This dimension concerns specific follow-up questions about patients' "
            "statements, not further exploration of their symptoms."
        ),
        checklist=[
            "Does the response clarify vague statements made by the patient, for example, by asking what exactly the patient means?",
            "Does the response openly acknowledge the therapist's own lack of knowledge where relevant?",
            "Does the response openly acknowledge the therapist's own uncertainty where relevant?",
            "Does the response emphasize the patient's expertise regarding their own experiences or intended meaning?",
            "Does the response check the therapist's understanding by asking follow-up questions?",
            "Does the response appropriately encourage the patient to take an active role, for example, by providing another example or a more detailed explanation?",
        ],
        anchor_examples={
            4: [
                "You said that you cannot talk to your family about being in therapy. What exactly do you mean by that? What is your family's attitude toward psychotherapy?",
                "You feel that your community may speak badly about you and see you as weak if they learn about your daughter's husband. When you say 'a weak man,' what exactly do you mean?",
            ],
            3: [
                "I would be interested to know how this is generally viewed in your culture. Are men who do not follow the traditions there considered weak? Or how should I understand that?",
                "It sounds as though you place yourself under considerable pressure regarding the language. Unfortunately, I also do not know what would be best for you right now. Perhaps we can discuss together what would be best for you at the moment. To do so, I would like to understand your situation somewhat better.",
            ],
            2: [
                "What makes you think that I have so little understanding or am unable to understand your situation and give you appropriate advice? Think about whether I may have said something wrong, or whether there are any indications that led you to this conclusion.",
                "You see, I would like to understand exactly what this means for you and your family so that I can better place the situation and, of course, help you.",
            ],
            1: [
                "I can imagine that it was very difficult for you to leave your country and start again in a foreign country. I would like to know what led you to leave the country at that time. Many different factors may have played a role.",
                "I can understand these concerns very well. I can tell you that therapy can be very helpful for the problems you have described, such as feeling stressed or tense.",
            ],
            0: [
                "Okay, so it bothers you that your daughter is going her own way and asserting her own will. And there is nothing you can do about that.",
                "I have worked as a therapist for many years and can help you even though I do not come from your culture. That is not relevant.",
            ],
        },
    ),
    "d3_cultural_factors": DimensionSpec(
        code="d3_cultural_factors",
        name="Consider cultural and contextual factors",
        instruction="Together with the patient, consider whether their culture or external circumstances, such as migration or discrimination, could influence their symptoms or the therapy.",
        criteria="""Suggests possible links between cultural or contextual factors, such as migration, language, discrimination, or religion, and the patient's symptoms or the therapeutic collaboration.
- Asks whether cultural or contextual conditions, such as migration, language, or experiences of discrimination, could predispose, trigger, or maintain the symptoms ("Could the language barrier be related to your feeling of loneliness?").
- Openly addresses differences between the patient and therapist as a possible influence on therapy ("Here we are, you as a Muslim woman and I as a white man. Do you have an idea of how this might affect our work together?").
- Addresses culture or faith as a resource or protective factor ("You seem very proud when you talk about your tradition.").
- Asks about distress or internal conflicts resulting from cultural differences.
- Develops hypotheses jointly with the patient ("Patients often report that their faith or culture plays a role. Could that also apply to you?").""",
        note=(
            "The aim is to understand the personal meaning of cultural and contextual "
            "influences as potential stressors or protective factors and to integrate "
            "them constructively into the therapeutic process. This dimension requires "
            "proactively proposing possible influences rather than merely exploring them."
        ),
        checklist=[
            "Does the response ask whether cultural or contextual conditions, such as migration, language, or discrimination, could predispose, trigger, or maintain the symptoms?",
            "Does the response openly address differences between the patient and therapist as a possible influence on therapy?",
            "Does the response address culture or faith as a resource or protective factor where relevant?",
            "Does the response ask about distress or internal conflicts resulting from cultural differences?",
            "Does the response develop hypotheses jointly with the patient about whether culture, faith, migration, language, discrimination, religion, or other external circumstances influence the symptoms, the therapeutic collaboration, or the therapy?",
        ],
        anchor_examples={
            4: [
                "First of all, I am sorry to hear that you experienced this. The idea that you should not study or pursue a leadership position might therefore be related to your experiences of racism. I wonder whether these experiences have weakened your self-confidence.",
                "It sounds as though you are in a difficult situation between two worlds. I wonder whether 'two worlds' also means that you are between two cultures. Could the different expectations and demands of the two cultures be wearing you down?",
            ],
            3: [
                "Do you experience a connection between your background and the feeling that you never fully belong?",
                "It is possible that your beliefs about who belongs at university and who does not also have a cultural background. We could examine this further and consider how these patterns of thought developed in your life, at which stages they occurred, and what role your migration background played.",
            ],
            2: [
                "I am sorry to hear that you feel like a failure. It is understandable that going through migration and also learning the language can be very difficult. I also think that migration plays a role here.",
                "That sounds like a great responsibility. I would be interested to know how you developed the belief that you are obligated to provide for your family. Is that something you know from your family?",
            ],
            1: [
                "Okay, do you have an idea of what might be causing you to feel excluded?",
                "Why do you think that people like you who have a migration background do not belong at university? If you look at the university, a fairly large proportion of people have a migration background. Why do you think that way?",
            ],
            0: [
                "But if you have friends, then you are not really excluded. Perhaps you are imagining it or simply expecting too much. People in Germany are somewhat more distant. This has nothing to do with your background.",
                "Perhaps you simply need to learn not to take everything so personally; otherwise, it will be difficult to cope with everyday life.",
            ],
        },
    ),
    "d4_family_system": DimensionSpec(
        code="d4_family_system",
        name="Involve the family system",
        instruction="Explore the situation described and encourage the patient to consider another perspective.",
        criteria="""Explores the symptoms in the family context and works jointly with the patient to find solutions to difficulties without giving advice.
- Validates the patient's difficult situation.
- Validates feelings of obligation or guilt toward the family ("I understand that you feel a great responsibility because of the sacrifices your parents made.").
- Validates the importance of the family ("It sounds as though your family's support is very important to you.").
- Asks how the family deals with problems in this context ("How does your family usually deal with problems like this?", "Does your family know about your problems?").
- Encourages the patient to independently generate possible solutions within their environment ("Who in your family could support you?").
- Attempts to adopt the perspective of relevant people ("What might help from the perspective of your family, friends, or other people in your social environment?").
- Identifies relevant people ("You mention your sister particularly often; she seems to play an important role.").
- Uses circular questions ("How do you think your father would feel if you told him that you ...?").""",
        note=(
            "The patient's social environment should be included. This dimension is not "
            "about the patient's own perspective."
        ),
        checklist=[
            "Does the response validate the patient's difficult situation?",
            "Does the response validate feelings of obligation or guilt toward the family?",
            "Does the response validate the importance of the family to the patient?",
            "Does the response ask how the family deals with problems in this context?",
            "Does the response encourage the patient to independently generate possible solutions within their own environment?",
            "Does the response invite the perspectives of relevant people from the patient's family or social environment?",
            "Does the response identify relevant people in the patient's family or social environment?",
            "Does the response use circular questions about how relevant family members or other people might feel, think, or react?",
        ],
        anchor_examples={
            4: [
                "Given the sacrifices your parents have made, it is understandable that you feel committed to your family's values. This makes it very difficult for you to find your own path. How do you think your family would react if they knew about your concerns?",
                "It sounds as though you are in a very difficult situation in which you are setting aside your own needs for the sake of your family. How do other women in your environment deal with being a good wife?",
            ],
            3: [
                "I understand that you are very upset right now. Did you speak to your mother about it in that situation?",
                "Do you think that you need to protect your family from your decisions?",
            ],
            2: [
                "I hear that you want freedom and would like to decide for yourself how to live your life. At the same time, you are worried that you will lose contact with your family if you do not meet their expectations. Have I understood that correctly?",
                "Okay, does that mean that you do not feel seen by your family?",
            ],
            1: [
                "I can understand that you are considering the consequences that moving out would have for your family. However, we should also consider the possibility of viewing things from another perspective and then making a free decision about which direction to take.",
                "At this moment, it is important to look inward and consider what feelings your husband's expectations evoke in you. Of course, we also need to consider whether you can and want to meet these expectations at all times and how willing you are to make changes.",
            ],
            0: [
                "You should resolve this now. Either stand by your partner or stop keeping everyone waiting. Secrecy does not help.",
                "You are an adult, and what your family thinks should not trouble you so much. At some point, you need to live your own life.",
            ],
        },
    ),
    "d5_ambiguity_tolerance": DimensionSpec(
        code="d5_ambiguity_tolerance",
        name="Tolerance of ambiguity",
        instruction="Invite the patient to discuss the perspective they expressed.",
        criteria="""Validates that the topic is highly meaningful to the patient, remains open to the views expressed, and explores them without judgment.
- Validates the importance or meaning of the topic ("I notice that this topic is very important to you.").
- Validates the patient's openness.
- Points out strengths and/or resources expressed in the patient's statement.
- Summarizes the perspective and reflects it back ("If I understand you correctly ...").
- Explores the perspective expressed ("People like that? What do you mean by that?").
- Summarizes the patient's statement in their own words.
- Does not judge the perspective as right or wrong ("That is your perspective, and it is important to me.").
- Addresses contradictory statements ("On the one hand, that sounds relieving to you; on the other hand, it also sounds distressing.").
- Respectfully sets personal boundaries, including around taboo topics ("This topic is very sensitive for me. I suggest that we leave it there for now.").""",
        note=(
            "The focus is on the patient's perspective, distinguishing this dimension "
            "from the family-system dimension."
        ),
        checklist=[
            "Does the response validate the importance or meaning of the topic to the patient?",
            "Does the response validate the patient's openness?",
            "Does the response identify strengths or resources expressed in the patient's statement?",
            "Does the response summarize and reflect the patient's perspective?",
            "Does the response explore the perspective expressed, for example, by asking what the patient means?",
            "Does the response summarize the patient's statement in its own words?",
            "Does the response avoid judging the patient's perspective as right or wrong?",
            "Does the response address contradictory statements where relevant?",
            "Does the response respectfully set personal boundaries where necessary?",
        ],
        anchor_examples={
            4: [
                "I can understand that tradition and family are very important to you. Why would you describe fellow students who are atheist as disrespectful? Could you explain that to me in greater detail?",
                "Yes, I can hear that religion is very important to you. It is difficult for you to encounter other people for whom it is not as important. What exactly is it about your religion that you value so much and that is especially important to you?",
            ],
            3: [
                "Could you explain that to me in greater detail? What exactly do you find disrespectful about it?",
                "It sounds to me as though you are quite torn and experiencing an internal conflict. I would like to look more closely with you at both sides that are speaking.",
            ],
            2: [
                "When you say that you need to win the angels' favor so that your problems disappear, is there also something that could have the opposite effect? I would like to discuss in greater detail how important it can be to do something for yourself and to help yourself.",
                "Now that we are talking here, we can consider together whether other factors might be contributing to your psychological problems and whether there are other factors that might help you feel better.",
            ],
            1: [
                "Okay, you have now stated very clearly where you think this comes from. Have you considered whether there might be another trigger or cause for these symptoms?",
                "So your explanation is that you are under a curse. Could there also be another explanation for the situation you are in and for how you are feeling right now?",
            ],
            0: [
                "If you believe that you are surrounded by a curse, then you are not considering all the facts. We should also consider other facts.",
                "A curse? That is superstition. Your symptoms have medical causes. You should accept that.",
            ],
        },
    ),
}


CriteriaFormat = Literal["bullets", "checklist"]
TaskFraming = Literal["standard", "minimal"]
ScoringRules = Literal["full", "minimal"]


@dataclass(frozen=True)
class VariantSpec:
    id: str
    name: str
    why_chosen: str
    include_anchors: bool
    include_rationale: bool
    rationale_before_score: bool = False
    criteria_format: CriteriaFormat = "bullets"
    include_criteria: bool = True
    include_note: bool = True
    task_framing: TaskFraming = "standard"
    scoring_rules: ScoringRules = "full"


VARIANTS: dict[str, VariantSpec] = {
    "V1_full_manual_baseline": VariantSpec(
        id="V1_full_manual_baseline",
        name="Full manual baseline",
        why_chosen="Full manual-guided reference condition",
        include_anchors=True,
        include_rationale=True,
    ),
    "V2_no_anchors": VariantSpec(
        id="V2_no_anchors",
        name="Manual without anchor examples",
        why_chosen="Do anchor examples improve scoring?",
        include_anchors=False,
        include_rationale=True,
    ),
    "V3_no_rationale": VariantSpec(
        id="V3_no_rationale",
        name="Manual without a rationale requirement",
        why_chosen="Does requiring a rationale influence scoring?",
        include_anchors=True,
        include_rationale=False,
    ),
    "V4_evidence_before_score": VariantSpec(
        id="V4_evidence_before_score",
        name="Evidence before score",
        why_chosen="Does evidence-before-score prompting influence performance?",
        include_anchors=True,
        include_rationale=True,
        rationale_before_score=True,
    ),
    "V5_structured_checklist": VariantSpec(
        id="V5_structured_checklist",
        name="Performance criteria restructured as a checklist",
        why_chosen="Does checklist-based decomposition influence scoring?",
        include_anchors=True,
        include_rationale=True,
        criteria_format="checklist",
    ),
    "V6_minimal_natural": VariantSpec(
        id="V6_minimal_natural",
        name="Minimal natural rating",
        why_chosen="How does the model perform with minimal instructions?",
        include_anchors=False,
        include_rationale=True,
        include_criteria=False,
        include_note=False,
        task_framing="minimal",
        scoring_rules="minimal",
    ),
}


def format_anchor_examples(anchor_examples: dict[int, list[str]]) -> str:
    """Format score-level anchor examples for insertion into a prompt."""
    lines: list[str] = []
    for score in sorted(anchor_examples.keys(), reverse=True):
        lines.append(f"{LABEL_ANCHOR_SCORE_PREFIX} {score}:")
        for example in anchor_examples[score]:
            lines.append(f"- {example}")
    return "\n".join(lines)


def format_checklist(checklist: list[str]) -> str:
    """Format the checklist as a numbered list."""
    return "\n".join(f"{index}. {item}" for index, item in enumerate(checklist, start=1))


def _section_task_framing(variant: VariantSpec) -> str:
    if variant.task_framing == "minimal":
        return MINIMAL_TASK_FRAMING.strip()
    return STANDARD_TASK_FRAMING.strip()


def _section_target(spec: DimensionSpec) -> str:
    return f"{LABEL_TARGET}\n{spec.name}"


def _section_instruction(spec: DimensionSpec) -> str:
    return f"{LABEL_INSTRUCTION}\n{spec.instruction}"


def _section_criteria(spec: DimensionSpec, variant: VariantSpec) -> str:
    if not variant.include_criteria:
        return ""
    if variant.criteria_format == "checklist":
        criteria_intro = spec.criteria.split("\n", 1)[0].strip()
        return (
            f"{LABEL_CRITERIA_CHECKLIST_HEADER}\n"
            f"{LABEL_CRITERIA_CHECKLIST_INTRO_PREFIX} {criteria_intro}\n"
            f"{LABEL_CRITERIA_CHECKLIST_GUIDANCE}\n\n"
            f"{format_checklist(spec.checklist)}"
        )
    criteria_summary, criteria_items = spec.criteria.split("\n", 1)
    return (
        f"{LABEL_CRITERIA_BULLETS}\n"
        f"{criteria_summary.strip()} {LABEL_MANUAL_PERFORMANCE_INDICATORS}\n"
        f"{criteria_items.strip()}"
    )


def _section_note(spec: DimensionSpec, variant: VariantSpec) -> str:
    if not variant.include_note:
        return ""
    return f"{LABEL_NOTE}\n{spec.note}"


def _section_anchors(spec: DimensionSpec, variant: VariantSpec) -> str:
    if not variant.include_anchors:
        return ""
    return f"{LABEL_ANCHORS}\n{format_anchor_examples(spec.anchor_examples)}"


def _section_scoring_rules(variant: VariantSpec) -> str:
    if variant.scoring_rules == "minimal":
        return f"{LABEL_SCORING_MINIMAL}\n{MINIMAL_SCORING_RULES.strip()}"
    return f"{LABEL_SCORING_GENERAL}\n{GENERAL_SCORING_RULES.strip()}"


def _section_transcript(transcript: str) -> str:
    return f"{LABEL_TRANSCRIPT}\n{transcript}"


def _section_output(variant: VariantSpec) -> str:
    if not variant.include_rationale:
        return (
            f"{LABEL_OUTPUT_NO_RATIONALE}\n"
            "{\n"
            '  "score": 0\n'
            "}"
        )
    if variant.rationale_before_score:
        return (
            f"{LABEL_OUTPUT_RATIONALE_BEFORE_LEAD}\n\n"
            f"{LABEL_OUTPUT_RATIONALE_BEFORE_INSTR}\n"
            "{\n"
            '  "brief_rationale": "...",\n'
            '  "score": 0\n'
            "}"
        )
    return (
        f"{LABEL_OUTPUT_RATIONALE_AFTER_INSTR}\n"
        "{\n"
        '  "score": 0,\n'
        '  "brief_rationale": "..."\n'
        "}"
    )


def build_prompt(dimension_code: str, transcript: str, variant_id: str) -> str:
    """Construct one rendered English prompt from plain variables.

    Args:
        dimension_code: One of the keys in DIMENSIONS, e.g. "d1_illness_beliefs".
        transcript: The participant's transcript text to be rated.
        variant_id: One of the keys in VARIANTS, e.g. "V1_full_manual_baseline".
    """
    if dimension_code not in DIMENSIONS:
        valid = ", ".join(DIMENSIONS)
        raise ValueError(
            f"Unknown dimension code {dimension_code!r}. Valid dimension codes: {valid}"
        )
    if variant_id not in VARIANTS:
        valid = ", ".join(VARIANTS)
        raise ValueError(f"Unknown prompt variant {variant_id!r}. Valid variants: {valid}")

    spec = DIMENSIONS[dimension_code]
    variant = VARIANTS[variant_id]
    transcript_text = "" if transcript is None else str(transcript).strip()

    sections = [
        _section_task_framing(variant),
        _section_target(spec),
        _section_instruction(spec),
        _section_criteria(spec, variant),
        _section_note(spec, variant),
        _section_anchors(spec, variant),
        _section_scoring_rules(variant),
        _section_transcript(transcript_text),
        _section_output(variant),
    ]
    return "\n\n".join(section for section in sections if section).strip()


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data" / "assets" / "en_assets" / "en_prompts"


def display_variants_for_dimension(
    dimension_code: str,
    output_dir: Path = DEFAULT_PROMPTS_DIR,
) -> list[Path]:
    """Render all prompt variants for one dimension and write each to a .txt file.

    The transcript field is left as a literal placeholder because it is supplied
    per call by the assessment script.
    """
    if dimension_code not in DIMENSIONS:
        valid = ", ".join(DIMENSIONS)
        raise ValueError(
            f"Unknown dimension code {dimension_code!r}. Valid dimension codes: {valid}"
        )

    dimension_dir = output_dir / dimension_code
    dimension_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for variant_id in VARIANTS:
        prompt = build_prompt(
            dimension_code,
            transcript="{transcript}",
            variant_id=variant_id,
        )
        file_path = dimension_dir / f"{variant_id}.txt"
        file_path.write_text(prompt, encoding="utf-8")
        written.append(file_path)

    return written


def display_variants_for_all_dimensions(
    output_dir: Path = DEFAULT_PROMPTS_DIR,
) -> list[Path]:
    """Render all prompt variants for every dimension and write them to .txt files."""
    written: list[Path] = []
    for dimension_code in DIMENSIONS:
        written.extend(display_variants_for_dimension(dimension_code, output_dir))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="English LLM-as-rater prompt construction utilities."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Render prompt variants for every dimension and save them as .txt files "
            "under data/assets/en_assets/en_prompts/<dimension_code>/."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.display:
        for path in display_variants_for_all_dimensions():
            print(f"Saved: {path}")
    else:
        print("Nothing to do. Pass --display to render all prompt variants.")
