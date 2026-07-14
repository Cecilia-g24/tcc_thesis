### to preview the 6 prompt variants for every dimension, run this script with the
### --display flag. The prompts will be saved as .txt files in
### data/assets/de_assets/de_prompts/<dimension_code>/.
# python de_prompt_construction.py --display

"""
German-language LLM-as-judge prompt templates for transcultural competence ratings.

This is the German counterpart of prompt_construction.py. It is intentionally
independent (no imports from prompt_construction.py) so the German prompt wording
can be maintained on its own without risk of silently changing the English prompts.
The section-building architecture mirrors prompt_construction.py exactly so the two
modules stay easy to compare; only the language and the source manual differ.

All dimension content (name, instruction, criteria, note, checklist, anchor examples)
is transcribed from the German source rating manual:
    data/assets/de_assets/de_doc/3_(de)2026_05_Ratingmanual_transkulturelleKompetenz.docx
not translated from the English prompt_construction.py content. Wording, including
quoted example utterances, is taken verbatim from that manual. One duplicated bullet
in the manual's table for d1_illness_beliefs ("Fasst Erklärungsmodelle in eigenen
Worten zusammen ...", which appears twice in the source table) has been deduplicated,
matching the 9 distinct performance criteria used for that dimension.

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
    V6_minimal_natural

V1-V5 are each defined as a single-factor change relative to V1_full_manual_baseline
(see VariantSpec / VARIANTS below), not as an independently written template. This keeps
the ablation clean: rendering V1 and any other variant for the same dimension differs
in exactly the one toggle that variant is meant to test, and nowhere else.

V6_minimal_natural is the exception: it uses a deliberately simple task framing and
keeps only the target dimension, the plain-language instruction given to participants,
a minimal 0-4 scale, the transcript, and the standard JSON output. It omits the manual's
detailed performance criteria, dimension-specific note, score-labelled anchors, and
full scoring guidance. This provides a natural-rating baseline for estimating the effect
of the detailed manual-based instructions.

The output JSON keys ("score", "brief_rationale") are kept in English across both the
German and English prompt variants, so downstream parsing code (e.g. llm_assessment.py's
valid_integer_score(parsed.get("score"))) works unchanged regardless of prompt language.

Usage (from an assessment script):
    from de_prompt_construction import build_prompt

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
- Verwenden Sie genau eine Ganzzahl von 0 bis 4 als Bewertung.
- Gültig sind ausschließlich die Werte 0, 1, 2, 3 und 4. Verwenden Sie keine Dezimalzahlen.
- 0 = mangelhaft umgesetzt.
- 2 = Aufgabenstellung erfüllt.
- 4 = exzellent umgesetzt.
- Eine Bewertung von 2 bedeutet, dass die Aufgabenstellung erfüllt ist.
- Für eine hohe Bewertung muss die Reaktion nicht perfekt sein.
- Für eine hohe Bewertung müssen nicht alle Leistungsmerkmale erfüllt sein.
- Auch einzelne gut umgesetzte Leistungsmerkmale können für eine hohe Punktzahl ausreichen.
- Ziehen Sie Punkte ab, wenn die Aufgabenstellung nur teilweise oder gar nicht erfüllt wurde.
- Berücksichtigen Sie neben den Leistungsmerkmalen auch die Basisfertigkeiten der Gesprächsführung: Die Reaktion sollte angemessen und für Patient:innen leicht verständlich sein.
- Ziehen Sie Punkte ab bei unklaren, missverständlichen, suggestiven, belehrenden, abwertenden, unverständlichen, sehr langen/verschachtelten Aussagen oder vielen Fragen hintereinander.
- Gehen Sie mit Transkriptionsfehlern so wohlwollend wie möglich um: Solange eine Aussage noch verstanden werden kann, sollten Transkriptionsfehler nicht in die Bewertung einfließen.
- Wenn der Inhalt des Transkripts unverständlich ist, vergeben Sie eine niedrige Bewertung.
"""

STANDARD_TASK_FRAMING = """\
Sie fungieren als geschulte:r Rater:in für einen Verhaltenstest zur transkulturellen Kompetenz in der Psychotherapie.

Sie erhalten ein Transkript einer therapeutischen Reaktion sowie Informationen zur zu bewertenden Dimension.

Stützen Sie Ihre Bewertung ausschließlich auf das bereitgestellte Transkript und die Informationen in diesem Prompt. Nehmen Sie keine Informationen an, die nicht im Transkript enthalten sind.
"""

MINIMAL_TASK_FRAMING = """\
Bewerten Sie die folgende therapeutische Reaktion im Hinblick auf die angegebene Zieldimension.

Stützen Sie Ihre Bewertung ausschließlich auf das Transkript. Nehmen Sie keine Informationen an, die nicht im Transkript enthalten sind.
"""

MINIMAL_SCORING_RULES = """\
- Verwenden Sie genau eine Ganzzahl von 0 bis 4 als Bewertung.
- 0 = mangelhaft umgesetzt.
- 2 = Aufgabenstellung erfüllt.
- 4 = exzellent umgesetzt.
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
        name="Subjektive Krankheits- und Heilungsvorstellungen erfragen",
        instruction="Explorieren Sie, wie sich die Patient:innen ihre Probleme selbst erklären.",
        criteria=(
            "Exploriert das subjektive Erklärungsmodell und/oder die Heilungsvorstellungen der Patient:innen, validiert deren Sichtweise "
            "und passt den therapeutischen Stil dabei flexibel an. Leistungsmerkmale des Manuals umfassen:\n"
            "- Fasst Erklärungsmodelle in eigenen Worten zusammen („Sie scheinen also überzeugt zu sein, dass Stress ein Auslöser war.“).\n"
            "- Fragt nach Erklärungsmodellen („Wie erklären Sie sich selbst Ihre Beschwerden?“, „Was glauben Sie selbst, wo dieses Problem herkommt?“).\n"
            "- Fragt nach Heilvorstellungen („Was glauben Sie, könnte Ihnen helfen?“).\n"
            "- Klärt Erwartungen an die Therapie („Sie haben sicher eine Vorstellung davon, wie ich Sie unterstützen kann“).\n"
            "- Definiert die Rolle als Therapeut:in.\n"
            "- Geht auf offensichtliche Vorbehalte/Bedenken von Patient:innen ein.\n"
            "- Zeigt sich verständnisvoll und sensibel für Zweifel/Einwände („Ich merke, dass Sie sich noch unsicher sind, ob Ihnen eine Therapie überhaupt helfen kann“).\n"
            "- Validiert Offenheit der Patient:innen (z. B. die Offenheit, eigene Zweifel zu äußern).\n"
            "- Validiert das Erklärungsmodell der Patient:innen."
        ),
        note=(
            "Weder die Exploration der Symptomatik noch die Validierung allein reichen aus. "
            "Es soll das Krankheits- und Heilverständnis explizit erfragt werden."
        ),
        checklist=[
            "Fasst die Reaktion das Erklärungsmodell der Patientin/des Patienten in eigenen Worten zusammen?",
            "Fragt die Reaktion nach dem Erklärungsmodell, z. B. wie die Patientin/der Patient sich die Beschwerden selbst erklärt oder was sie/er glaubt, wo das Problem herkommt?",
            "Fragt die Reaktion nach den Heilungsvorstellungen der Patientin/des Patienten, z. B. was ihr/ihm helfen könnte?",
            "Klärt die Reaktion die Erwartungen an die Therapie oder wie die Therapeutin/der Therapeut unterstützen kann?",
            "Definiert die Reaktion die Rolle der Therapeutin/des Therapeuten?",
            "Geht die Reaktion auf offensichtliche Vorbehalte oder Bedenken der Patientin/des Patienten ein?",
            "Zeigt sich die Reaktion verständnisvoll und sensibel gegenüber Zweifeln oder Einwänden der Patientin/des Patienten?",
            "Validiert die Reaktion die Offenheit der Patientin/des Patienten, eigene Zweifel oder Bedenken zu äußern?",
            "Validiert die Reaktion das Erklärungsmodell der Patientin/des Patienten?",
        ],
        anchor_examples={
            4: [
                "Ja, ich verstehe, Sie merken, dass es da einen Zusammenhang geben könnte zwischen dem Kontaktabbruch mit Ihrer Familie und den Beschwerden, die Sie aufzählen. Was glauben Sie denn, wie das miteinander zusammenhängen könnte?",
                "Sie zweifeln daran, dass die Therapie ihnen helfen kann, weil nur Gott ihnen helfen kann. Ich kann mir vorstellen, dass es frustrierend ist, wenn sie in der Therapie sitzen und hinterfragen, was das alles bringen soll. Ich stelle mir die Frage, welche Vorstellungen sie haben, und wie ich ihnen trotzdem helfen kann?",
            ],
            3: [
                "Sie sagen, Ihre Flucht könnte eine Strafe Gottes sein oder die Symptome, die da entstanden sind. Wie meinen Sie das dann? Also was wäre dann die Strafe Gottes? Also wofür würde er Sie bestrafen?",
                "Das heißt, Sie fühlen sich schuldig, oder? Was glauben Sie, wofür genau wurden Sie bestraft?",
            ],
            2: [
                "Wenn ich mich so in ihre Lage versetze, kann ich mir vorstellen, dass es auch relativ anstrengend ist, immer und immer wieder mehr ins Detail zu gehen und noch mehr zu erzählen. Allerdings bin ich nur die Spezialistin, die das Fachwissen hat. Sie sind der Spezialist von Ihnen und nur Sie können bestimmte Fragen über Sie beantworten.",
                "Ich verstehe Ihre Frustration, aber leider kann ich nicht in Ihren Kopf reinschauen. Also müssen wir zusammenarbeiten, um herauszufinden, wie wir Ihnen helfen können.",
            ],
            1: [
                "Ich kann gut nachvollziehen, dass Sie sich dadurch sehr belastet fühlen und es für Sie ein unangenehmes Gefühl ist, die Eltern in dieser Situation zurückzulassen.",
                "Sie haben den Eindruck, dass Sie sich da gar nicht allein rausholen können, oder? Sie denken, Sie brauchen Unterstützung von außen.",
            ],
            0: [
                "Ich denke, Sie sehen da einen Zusammenhang, wo vielleicht gar keiner ist. Körperliche Beschwerden entstehen nicht einfach so, nur weil man keinen Kontakt zu seiner Familie hat.",
                "Das klingt so, als würden Sie sich selbst im Weg stehen. Sie müssen akzeptieren, dass es vorbei ist. Wenn Sie sich nur auf Ihre Einsamkeit konzentrieren, wird sich auch nichts ändern.",
            ],
        },
    ),
    "d2_lack_of_knowledge": DimensionSpec(
        code="d2_lack_of_knowledge",
        name="Proaktiver Umgang mit Nichtwissen",
        instruction="Ermutigen Sie die Patient:innen, ihre Aussage genauer zu erklären.",
        criteria=(
            "Kann sich Patient:innen offen zuwenden, zeigt eigene Wissenslücken und fragt nach, um Aussagen besser zu verstehen. "
            "Leistungsmerkmale des Manuals umfassen:\n"
            "- Klärt ungenaue Patient:innenaussagen („Was genau meinen Sie mit …?“, „Wenn Sie sagen »…«, was genau verstehen Sie unter diesem Begriff?“).\n"
            "- Benennt offen eigenes Nichtwissen („Das ist mir neu – ich würde das gern besser verstehen“).\n"
            "- Benennt offen eigene Unsicherheiten („Ich weiß leider auch nicht, was jetzt das Beste ist – lassen Sie uns gemeinsam überlegen, was für Sie hilfreich sein könnte.“).\n"
            "- Betont die Expertise der Patient:innen („Sie sind Experte…“).\n"
            "- Sichert das Verständnis der Patient:in, indem Rückfragen gestellt werden („Ich möchte sicherstellen, dass ich Sie richtig verstehe.“).\n"
            "- Ermutigt Patient:innen in angemessener Weise, eine aktive Rolle einzunehmen („Es hilft mir sehr, wenn Sie mir dazu nochmal ein Beispiel geben.“)."
        ),
        note=(
            "Hier geht es um konkrete Nachfragen zu Patient:innenaussagen, nicht um die weitere Exploration der Beschwerden."
        ),
        checklist=[
            "Klärt die Reaktion ungenaue Aussagen der Patientin/des Patienten, z. B. indem gefragt wird, was genau gemeint ist?",
            "Benennt die Reaktion offen das eigene Nichtwissen der Therapeutin/des Therapeuten, wo relevant?",
            "Benennt die Reaktion offen eigene Unsicherheiten der Therapeutin/des Therapeuten, wo relevant?",
            "Betont die Reaktion die Expertise der Patientin/des Patienten hinsichtlich der eigenen Erfahrung oder Bedeutung?",
            "Sichert die Reaktion das Verständnis der Therapeutin/des Therapeuten durch Rückfragen ab?",
            "Ermutigt die Reaktion die Patientin/den Patienten in angemessener Weise, eine aktive Rolle einzunehmen, z. B. durch ein weiteres Beispiel oder eine genauere Erklärung?",
        ],
        anchor_examples={
            4: [
                "Sie sagten, dass Sie mit Ihrer Familie nicht über die Therapie hier sprechen können. Was genau meinen Sie damit? Welche Haltung besteht in Ihrer Familie gegenüber Psychotherapie?",
                "Sie haben das Gefühl, dass Ihre Gemeinschaft schlecht über Sie redet und Sie als schwach ansehen könnte, wenn sie von dem Mann Ihrer Tochter erfährt. Wenn Sie von einem „schwachen Mann“ sprechen, was genau meinen Sie damit?",
            ],
            3: [
                "Mich würde nur interessieren, wie es in der Regel in Ihrer Kultur ist. Werden Männer, die sich nicht an die dort geltenden Traditionen halten, für schwach gehalten? Oder wie soll ich das verstehen?",
                "Es klingt, als würden Sie sich in Bezug auf die Sprache sehr unter Druck setzen. Leider weiß ich auch nicht, was jetzt das Beste für Sie ist. Vielleicht können wir gemeinsam besprechen, was im Moment am besten für Sie ist. Dazu würde ich Ihre Situation gern noch etwas besser verstehen.",
            ],
            2: [
                "Wie kommen Sie darauf, dass ich so wenig Verständnis habe oder nicht in der Lage bin, Ihre Situation zu begreifen und Ihnen entsprechenden Rat geben zu können? Denken Sie einmal darüber nach, was ich möglicherweise Falsches gesagt habe, oder gibt es irgendwelche Anhaltspunkte, die Sie zu diesem Schluss kommen lassen?",
                "Sehen Sie, ich würde gerne verstehen, was es genau für Sie und Ihre Familie bedeutet, um das Ganze besser einzuordnen und Ihnen dann natürlich gerne auch weiterzuhelfen.",
            ],
            1: [
                "Ich kann mir gut vorstellen, dass es für Sie sehr schwierig war, Ihr Land zu verlassen und in einem fremden Land ganz neu anzufangen. Ich würde gerne wissen, was Sie damals dazu bewegt hat, das Land zu verlassen. Es gibt ja ganz viele Faktoren, die dabei eine Rolle spielen.",
                "Diese Sorgen kann ich sehr gut nachvollziehen. Ich kann Ihnen sagen, dass eine Therapie bei den von Ihnen geschilderten Problemen, also wenn Sie sich gestresst fühlen oder angespannt sind, wirklich sehr hilfreich sein kann.",
            ],
            0: [
                "Okay, das heißt es belastet Sie, dass ihre Tochter ihren eigenen Weg geht und ihren eigenen Willen hat. Und da können Sie nichts gegen machen.",
                "Ich bin seit vielen Jahren als Therapeutin tätig und kann Ihnen helfen, auch wenn ich nicht aus Ihrer Kultur komme. Das ist nicht relevant.",
            ],
        },
    ),
    "d3_cultural_factors": DimensionSpec(
        code="d3_cultural_factors",
        name="Berücksichtigung kultureller und kontextueller Faktoren",
        instruction=(
            "Überlegen Sie gemeinsam mit den Patient:innen, ob ihre Kultur oder äußere Umstände (z. B. Migration, Diskriminierung) "
            "ihre Beschwerden oder die Therapie beeinflussen könnten."
        ),
        criteria=(
            "Schlägt vor, ob kulturelle oder kontextuelle Faktoren (z. B. Migration, Sprache, Diskriminierung, Religion) die Beschwerden "
            "oder die Zusammenarbeit beeinflussen. Leistungsmerkmale des Manuals umfassen:\n"
            "- Fragt, ob kulturelle oder kontextuelle Bedingungen (z. B. Migration, Sprache, Diskriminierungserfahrungen) prädisponierend, "
            "auslösend oder aufrechterhaltend auf die Beschwerden wirken könnten („Könnte die Sprachbarriere etwas mit dem Gefühl der Einsamkeit zu tun haben?“).\n"
            "- Thematisiert Unterschiede zwischen Patient:in und Therapeut:in offen als Einflussfaktor auf die Therapie („Jetzt sitzen wir hier, Sie als Muslima und ich als weißer Mann. Haben Sie eine Idee, wie sich das auf unsere Zusammenarbeit auswirken könnte?“).\n"
            "- Greift Kultur/Glaube als Ressource/Schutzfaktor auf („Sie wirken sehr stolz, wenn Sie von Ihrer Tradition erzählen.“).\n"
            "- Fragt nach Belastung/inneren Konflikten, die aus kulturellen Unterschieden resultieren.\n"
            "- Entwickelt gemeinsam mit Patient:innen Hypothesen („Häufig berichten Patient:innen, dass ihr Glaube oder ihre Kultur eine Rolle spielt – könnte das auch bei Ihnen zutreffen?“)."
        ),
        note=(
            "Ziel ist es, die persönliche Bedeutung kultureller und kontextueller Einflüsse als potenzielle Belastungs- oder Schutzfaktoren "
            "zu verstehen und konstruktiv in den Therapieprozess einzubeziehen. Diese Dimension erfordert proaktive Vorschläge, nicht nur allgemeine Exploration."
        ),
        checklist=[
            "Fragt die Reaktion, ob kulturelle oder kontextuelle Bedingungen, z. B. Migration, Sprache oder Diskriminierung, die Beschwerden prädisponieren, auslösen oder aufrechterhalten könnten?",
            "Thematisiert die Reaktion offen Unterschiede zwischen Patientin/Patient und Therapeutin/Therapeut als möglichen Einflussfaktor auf die Therapie?",
            "Greift die Reaktion Kultur oder Glauben als Ressource oder Schutzfaktor auf, wo relevant?",
            "Fragt die Reaktion nach Belastungen oder inneren Konflikten, die aus kulturellen Unterschieden resultieren?",
            "Entwickelt die Reaktion gemeinsam mit der Patientin/dem Patienten Hypothesen darüber, ob Kultur, Glaube, Migration, Sprache, Diskriminierung, Religion oder andere äußere Umstände die Beschwerden, die Zusammenarbeit oder die Therapie beeinflussen?",
        ],
        anchor_examples={
            4: [
                "Zunächst einmal tut es mir leid zu hören, dass Sie das erlebt haben. Der Gedanke, dass sie nicht studieren und eine Führungsposition anstreben sollten, könnte also mit ihren Rassismuserfahrungen zusammenhängen. Ich frage mich, ob diese Erfahrungen ihr Selbstvertrauen geschwächt haben.",
                "Das klingt, als befänden Sie sich in einer schwierigen Situation zwischen zwei Welten. Ich frage mich, ob zwei Welten auch bedeutet, dass Sie sich zwischen zwei Kulturen befinden. Könnte es sein, dass die unterschiedlichen Erwartungen und Ansprüche der beiden Kulturen an Ihnen zehren?",
            ],
            3: [
                "Erleben Sie einen Zusammenhang zwischen Ihrer Herkunft und dem Gefühl, nie vollständig dazu zu gehören?",
                "Möglicherweise könnten Ihre Vorstellungen, wer an die Universität gehört und wer nicht, auch einen kulturellen Hintergrund haben. Dies könnte man noch einmal hinterfragen und dabei untersuchen, wie es zu diesen Gedankengängen in Ihrem Leben gekommen ist, an welchen Stationen Ihres Lebens dies der Fall war und welche Rolle der Migrationshintergrund dabei spielt.",
            ],
            2: [
                "Das tut mir leid zu hören, dass sie sich so wie ein Versager fühlen. Verständlich, wenn man diesen Migrationsprozess macht, und wenn man dann auch die Sprache zu lernen, das kann ich mir sehr schwierig vorstellen. Ich denke aber auch, dass die Migration eine Rolle hier spielt.",
                "Das hört sich nach einer großen Verantwortung an. Mich würde interessieren, wie Sie die Überzeugung entwickelt haben, dass Sie verpflichtet sind, Ihre Familie zu versorgen. Kennen Sie das aus Ihrer Familie?",
            ],
            1: [
                "Okay, da haben sie eine Idee, woran das liegen könnte, dass sie sich ausgeschlossen fühlen?",
                "Warum denken Sie denn, dass Menschen wie Sie mit Migrationshintergrund nicht an die Uni gehören? Also wenn Sie sich die Uni mal anschauen, dann ist da doch ein ziemlich großer Anteil mit Migrationshintergrund. Wie kommt das, dass Sie so denken?",
            ],
            0: [
                "Aber wenn Sie Freunde haben, dann sind Sie doch gar nicht wirklich ausgeschlossen. Vielleicht bilden Sie sich das ein oder erwarten einfach zu viel. In Deutschland sind die Leute eben etwas distanzierter. Das hat nichts mit Herkunft zu tun.",
                "Vielleicht müssen Sie einfach lernen, nicht alles so persönlich zu nehmen – sonst wird es schwierig, im Alltag zurechtzukommen.",
            ],
        },
    ),
    "d4_family_system": DimensionSpec(
        code="d4_family_system",
        name="Einbezug des Familiensystems",
        instruction="Explorieren Sie die geschilderte Situation und regen Sie die Patientin dazu an, die Perspektive zu wechseln.",
        criteria=(
            "Exploriert Beschwerden im familiären Kontext und sucht gemeinsam nach Lösungen für Schwierigkeiten, ohne Ratschläge zu geben. "
            "Leistungsmerkmale des Manuals umfassen:\n"
            "- Validiert die schwierige Situation der Patientin.\n"
            "- Validiert das Gefühl der Verpflichtung oder der Schuld gegenüber der Familie („Ich verstehe, dass Sie aufgrund der Opfer Ihrer Eltern eine große Verantwortung empfinden.“).\n"
            "- Validiert die Bedeutung der Familie („Es klingt, als sei die Unterstützung Ihrer Familie für Sie sehr wichtig.“).\n"
            "- Fragt nach familiärem Umgang im Problemkontext („Wie geht Ihre Familie normalerweise mit solchen Problemen um?“, „Weiß Ihre Familie über Ihre Probleme?“).\n"
            "- Regt Patient:innen an, selbstständig mögliche Lösungen in ihrem Umfeld zu generieren („Wer in Ihrer Familie könnte Sie unterstützen?“).\n"
            "- Versucht, sich in die Sichtweise von relevanten Bezugspersonen hineinzuversetzen („Was würde aus Sicht Ihrer Familie, Ihrer Freunde oder anderer Menschen in Ihrem sozialen Umfeld helfen?“).\n"
            "- Erfasst relevante Bezugspersonen („Sie erwähnen Ihre Schwester besonders oft - sie scheint eine wichtige Rolle zu spielen.“).\n"
            "- Verwendet zirkuläre Fragen („Was denken Sie, wie es Ihrem Vater geht, wenn Sie ihm sagen, dass Sie ...“)."
        ),
        note=(
            "Es soll das Umfeld der Patientin/des Patienten einbezogen werden. Bei dieser Dimension geht es nicht in erster Linie "
            "um die eigene Sichtweise der Patientin/des Patienten."
        ),
        checklist=[
            "Validiert die Reaktion die schwierige Situation der Patientin/des Patienten?",
            "Validiert die Reaktion das Gefühl der Verpflichtung oder Schuld gegenüber der Familie?",
            "Validiert die Reaktion die Bedeutung der Familie für die Patientin/den Patienten?",
            "Fragt die Reaktion nach dem familiären Umgang mit dem Problem, z. B. wie die Familie normalerweise mit solchen Problemen umgeht?",
            "Regt die Reaktion die Patientin/den Patienten an, selbstständig mögliche Lösungen im eigenen Umfeld zu generieren?",
            "Lädt die Reaktion zur Perspektivübernahme relevanter Personen aus dem familiären oder sozialen Umfeld der Patientin/des Patienten ein?",
            "Erfasst die Reaktion relevante Bezugspersonen im familiären oder sozialen Umfeld?",
            "Verwendet die Reaktion zirkuläre Fragen dazu, wie relevante Familienmitglieder oder andere Personen fühlen, denken oder reagieren könnten?",
        ],
        anchor_examples={
            4: [
                "Angesichts der Opfer, die Ihre Eltern gebracht haben, ist es verständlich, dass Sie sich den Werten Ihrer Familie verpflichtet fühlen. Das macht es für Sie sehr schwer, Ihren eigenen Weg zu finden. Was glauben Sie, wie Ihre Familie reagieren würde, wenn sie von Ihren Sorgen wüsste?",
                "Das klingt, als befänden Sie sich gerade in einer ziemlich schwierigen Situation, in der Sie Ihre eigenen Bedürfnisse zum Wohle Ihrer Familie zurückstellen. Wie gehen andere Frauen in Ihrem Umfeld damit um, eine gute Ehefrau zu sein?",
            ],
            3: [
                "Ich verstehe, dass Sie da jetzt sehr aufgewühlt sind. Haben Sie da Ihre Mutter dann darauf angesprochen in der Situation?",
                "Denken Sie, dass Sie Ihre Familie schützen müssen vor Ihren Entscheidungen?",
            ],
            2: [
                "Ich höre, dass Sie einen Wunsch nach Freiheit haben und gerne selbst entscheiden möchten, wie Sie ihr Leben führen und gleichzeitig haben Sie die Sorge, dass Sie den Kontakt zu Ihrer Familie verlieren, wenn Sie nicht den Erwartungen Ihrer Familie entsprechen. Habe ich das richtig verstanden?",
                "Okay, das heißt, sie fühlen sich nicht gesehen von ihrer Familie?",
            ],
            1: [
                "Ich kann gut nachvollziehen, dass Sie die Konsequenzen Ihres Auszugs für Ihre Familie überdenken. Wir sollten aber auch die Möglichkeiten sehen, Dinge aus einer anderen Perspektive zu betrachten und dann eine freie Entscheidung zu treffen, in welche Richtung es gehen könnte.",
                "In diesem Moment ist es wichtig, in sich hineinzuschauen und zu überlegen, welche Gefühle die Erwartungen Ihres Mannes in Ihnen auslösen. Natürlich müssen wir auch schauen, ob Sie diesen Erwartungen immer gerecht werden können und wollen und wie groß Ihre Veränderungsbereitschaft ist.",
            ],
            0: [
                "Das sollten Sie jetzt aber mal klären. Entweder stehen Sie zu Ihrem Partner oder Sie hören auf, alle hinzuhalten. Heimlichkeiten bringen doch nichts.",
                "Sie sind erwachsen, und was Ihre Familie denkt, sollte Sie nicht so sehr belasten. Irgendwann muss man sein eigenes Leben leben.",
            ],
        },
    ),
    "d5_ambiguity_tolerance": DimensionSpec(
        code="d5_ambiguity_tolerance",
        name="Ambiguitätstoleranz",
        instruction="Laden Sie die Patient:innen zu einem Dialog über die genannte Sichtweise ein.",
        criteria=(
            "Validiert, dass das Thema für die Patient:innen eine große Bedeutung hat, zeigt sich offen gegenüber den genannten Sichtweisen "
            "und exploriert diese ohne Bewertung. Leistungsmerkmale des Manuals umfassen:\n"
            "- Validiert die Wichtigkeit/Bedeutung („Ich merke, dass dieses Thema für Sie eine große Bedeutung hat.“).\n"
            "- Validiert die Offenheit der Patient:innen.\n"
            "- Weist auf Stärken und/oder Ressourcen hin, die in der Aussage der Patient:innen zum Ausdruck kommen.\n"
            "- Fasst die Sichtweise zusammen und spiegelt sie zurück („Wenn ich Sie richtig verstehe...“).\n"
            "- Exploriert die genannte Sichtweise („Leute wie diese? Was meinen Sie damit?“).\n"
            "- Fasst die Aussage der Patient:innen in eigenen Worten zusammen.\n"
            "- Beurteilt die Sichtweise weder als richtig noch als falsch („Das ist Ihre Sichtweise, und die ist für mich wichtig.“).\n"
            "- Greift widersprüchliche Aussagen auf („Einerseits klingt das für Sie erleichternd, andererseits auch belastend.“).\n"
            "- Setzt respektvoll eigene (Tabu-)Grenzen („Dieses Thema ist für mich sehr sensibel. Ich würde vorschlagen, dass wir das erstmal so stehen lassen.“)."
        ),
        note=(
            "Im Fokus steht die Sichtweise der Patientin/des Patienten. Dies grenzt sich von der Dimension Familiensystem ab."
        ),
        checklist=[
            "Validiert die Reaktion die Wichtigkeit oder Bedeutung des Themas für die Patientin/den Patienten?",
            "Validiert die Reaktion die Offenheit der Patientin/des Patienten?",
            "Weist die Reaktion auf Stärken oder Ressourcen hin, die in der Aussage der Patientin/des Patienten zum Ausdruck kommen?",
            "Fasst die Reaktion die Sichtweise der Patientin/des Patienten zusammen und spiegelt sie zurück?",
            "Exploriert die Reaktion die genannte Sichtweise, z. B. indem gefragt wird, was die Patientin/der Patient damit meint?",
            "Fasst die Reaktion die Aussage der Patientin/des Patienten in eigenen Worten zusammen?",
            "Vermeidet die Reaktion, die Sichtweise der Patientin/des Patienten als richtig oder falsch zu beurteilen?",
            "Greift die Reaktion widersprüchliche Aussagen auf, wo relevant?",
            "Setzt die Reaktion respektvoll eigene Grenzen, wo nötig?",
        ],
        anchor_examples={
            4: [
                "Ich kann gut verstehen, dass Ihnen das Thema Tradition sehr wichtig ist und Familie. Wieso würden Sie denn, ja, Kommilitoninnen, die jetzt atheistisch orientiert sind, als respektlos bezeichnen? Können Sie mir das genauer erklären?",
                "Ja, ich höre raus, dass Ihnen das Thema Religion sehr wichtig ist. Es ist schwierig für Sie, dann mit anderen Menschen konfrontiert zu sein, denen das nicht so wichtig ist. Was genau an ihrer Religion ist es denn, was sie so sehr schätzen und was ihnen auch besonders wichtig ist?",
            ],
            3: [
                "Können Sie mir das noch genauer erklären? Was genau finden Sie darin respektlos?",
                "Für mich klingt es so, als wenn sie da ziemlich hin und her gerissen und wie in einem inneren Kampf. Ich würde mir gerne beide Seiten, die da sprechen, mit ihnen mal genauer anschauen.",
            ],
            2: [
                "Wenn Sie jetzt sagen, dass Sie die Engel positiv stimmen müssen, damit Ihre Probleme verschwinden, gibt es dann auch ein Gegenteil? Ich würde gerne noch einmal ausführlicher mit Ihnen darüber sprechen, wie wichtig es sein kann, selbst etwas für sich zu tun und sich selbst zu helfen.",
                "Nun, da wir ja gerade hier zusammensprechen, können wir gemeinsam versuchen zu überlegen, ob es nicht irgendwelche anderen Faktoren gibt, die zu ihren psychischen Problemen beitragen könnten und ob es weitere Faktoren gibt, die helfen können, dass sie sich besser fühlen würden.",
            ],
            1: [
                "Okay, jetzt haben Sie ja schon sehr klar gesagt, woher das Ihrer Meinung nach kommt. Haben Sie sich schon mal Gedanken darüber gemacht, ob es vielleicht noch einen anderen Auslöser irgendwie für diese Symptome gibt oder eine andere Entstehung dafür?",
                "Ihre Erklärung ist also, dass ein Fluch auf Ihnen liegt, könnte es auch noch eine andere Erklärung geben für die Situation, in der Sie sind, für das, wie Sie sich gerade fühlen.",
            ],
            0: [
                "Wenn Sie daran glauben, von einem Fluch umgeben zu sein, dann betrachten Sie nicht alle Fakten. Wir sollten auch andere Fakten berücksichtigen.",
                "Ein Fluch? Das ist doch Aberglaube. Ihre Symptome haben medizinische Ursachen. Das sollten Sie akzeptieren.",
            ],
        },
    ),
}


# ---------------------------------------------------------------------------
# Six prompt variants. V1-V5 are single-factor toggles relative to
# V1_full_manual_baseline; V6 uses minimal task framing and a minimal scale
# while dropping detailed manual-specific guidance (see the module docstring).
# A shared rendering function (build_prompt)
# assembles sections in a fixed order and includes, excludes, or reorders a
# section only according to these toggles, so that comparing V1's rendering
# to any other variant's rendering isolates exactly the intended change(s).
# ---------------------------------------------------------------------------

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
        name="Vollständige Manual-Baseline",
        why_chosen="Vollständige manualbasierte Referenzbedingung",
        include_anchors=True,
        include_rationale=True,
        rationale_before_score=False,
        criteria_format="bullets",
    ),
    "V2_no_anchors": VariantSpec(
        id="V2_no_anchors",
        name="Manual ohne Ankerbeispiele",
        why_chosen="Helfen Ankerbeispiele?",
        include_anchors=False,
        include_rationale=True,
        rationale_before_score=False,
        criteria_format="bullets",
    ),
    "V3_no_rationale": VariantSpec(
        id="V3_no_rationale",
        name="Manual ohne Begründungsausgabe",
        why_chosen="Beeinflusst eine geforderte explizite Begründung die Bewertung?",
        include_anchors=True,
        include_rationale=False,
        rationale_before_score=False,
        criteria_format="bullets",
    ),
    "V4_evidence_before_score": VariantSpec(
        id="V4_evidence_before_score",
        name="Begründung vor Bewertung",
        why_chosen="Beeinflusst eine explizite evidenzbasierte Begründung vor der abschließenden Bewertung die Leistung?",
        include_anchors=True,
        include_rationale=True,
        rationale_before_score=True,
        criteria_format="bullets",
    ),
    "V5_structured_checklist": VariantSpec(
        id="V5_structured_checklist",
        name="Leistungsmerkmale als Checkliste umstrukturiert",
        why_chosen="Beeinflusst die Umstrukturierung derselben Leistungsmerkmale als Checkliste die Bewertung?",
        include_anchors=True,
        include_rationale=True,
        rationale_before_score=False,
        criteria_format="checklist",
    ),
    "V6_minimal_natural": VariantSpec(
        id="V6_minimal_natural",
        name="Minimale natürliche Bewertung",
        why_chosen="Wie bewertet das Modell mit minimaler Anleitung, und wie groß ist der Effekt der detaillierten Manualinstruktionen?",
        include_anchors=False,
        include_rationale=True,
        rationale_before_score=False,
        criteria_format="bullets",
        include_criteria=False,
        include_note=False,
        task_framing="minimal",
        scoring_rules="minimal",
    ),
}


# ---------------------------------------------------------------------------
# Formatting helpers for dimension-specific content
# ---------------------------------------------------------------------------

def format_anchor_examples(anchor_examples: dict[int, list[str]]) -> str:
    """Format score-level anchor examples for insertion into a prompt."""
    lines: list[str] = []
    for score in sorted(anchor_examples.keys(), reverse=True):
        lines.append(f"Beispiele für Bewertung {score}:")
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

def _section_task_framing(variant: VariantSpec) -> str:
    if variant.task_framing == "minimal":
        return MINIMAL_TASK_FRAMING.strip()
    return STANDARD_TASK_FRAMING.strip()


def _section_target(spec: DimensionSpec) -> str:
    return f"Zieldimension:\n{spec.name}"


def _section_instruction(spec: DimensionSpec) -> str:
    return f"Instruktion:\n{spec.instruction}"


def _section_criteria(spec: DimensionSpec, variant: VariantSpec) -> str:
    if not variant.include_criteria:
        return ""
    if variant.criteria_format == "checklist":
        criteria_intro = spec.criteria.split("Leistungsmerkmale des Manuals umfassen:", 1)[0].strip()
        return (
            "Bewertungskriterien (Checklistenform; dieselben Leistungsmerkmale des Manuals, umstrukturiert):\n"
            f"Nutzen Sie die folgende Checkliste, um zu beurteilen, ob die Reaktion dieselben im Manual beschriebenen Kriterien erfüllt: {criteria_intro}\n"
            "Nutzen Sie die Checkliste zur Steuerung der Aufmerksamkeit, zählen Sie die Punkte aber nicht mechanisch ab. Vergeben Sie die abschließende Bewertung von 0-4 ganzheitlich gemäß dem Ratingmanual.\n\n"
            f"{format_checklist(spec.checklist)}"
        )
    return f"Bewertungskriterien:\n{spec.criteria}"


def _section_note(spec: DimensionSpec, variant: VariantSpec) -> str:
    if not variant.include_note:
        return ""
    return f"Wichtiger dimensionsspezifischer Hinweis:\n{spec.note}"


def _section_anchors(spec: DimensionSpec, variant: VariantSpec) -> str:
    if not variant.include_anchors:
        return ""
    return f"Ankerbeispiele aus dem Ratingmanual:\n{format_anchor_examples(spec.anchor_examples)}"


def _section_scoring_rules(variant: VariantSpec) -> str:
    if variant.scoring_rules == "minimal":
        return f"Bewertungsskala:\n{MINIMAL_SCORING_RULES.strip()}"
    return f"Allgemeine Bewertungsregeln:\n{GENERAL_SCORING_RULES.strip()}"


def _section_transcript(transcript: str) -> str:
    return f"Transkript:\n{transcript}"


def _section_output(variant: VariantSpec) -> str:
    if not variant.include_rationale:
        return (
            'Geben Sie ausschließlich gültiges JSON mit genau diesem Schlüssel zurück. '
            'Der Wert von "score" muss eine Ganzzahl 0, 1, 2, 3 oder 4 sein:\n'
            "{\n"
            '  "score": 0\n'
            "}"
        )
    if variant.rationale_before_score:
        return (
            "Geben Sie zunächst eine kurze Begründung an, die sich auf konkrete Belege aus dem Transkript stützt. "
            "Vergeben Sie anschließend die abschließende Bewertung, die mit dieser Begründung übereinstimmt.\n\n"
            'Geben Sie ausschließlich gültiges JSON mit genau diesen Schlüsseln in dieser Reihenfolge zurück. '
            'Der Wert von "score" muss eine Ganzzahl 0, 1, 2, 3 oder 4 sein:\n'
            "{\n"
            '  "brief_rationale": "...",\n'
            '  "score": 0\n'
            "}"
        )
    return (
        'Geben Sie ausschließlich gültiges JSON mit genau diesen Schlüsseln zurück. '
        'Der Wert von "score" muss eine Ganzzahl 0, 1, 2, 3 oder 4 sein:\n'
        "{\n"
        '  "score": 0,\n'
        '  "brief_rationale": "..."\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(dimension_code: str, transcript: str, variant_id: str) -> str:
    """Construct one rendered German prompt from plain variables.

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
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data" / "assets" / "de_assets" / "de_prompts"


def display_variants_for_dimension(dimension_code: str, output_dir: Path = DEFAULT_PROMPTS_DIR) -> list[Path]:
    """Render all prompt variants for one dimension and write each to its own .txt file.

    The transcript field is left as a literal placeholder since it is supplied
    per-call by whatever script calls build_prompt().

    Args:
        dimension_code: One of the keys in DIMENSIONS, e.g. "d1_illness_beliefs".
        output_dir: Directory where the variant text files are written, under
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


def display_variants_for_all_dimensions(output_dir: Path = DEFAULT_PROMPTS_DIR) -> list[Path]:
    """Render all prompt variants for every dimension and write each to its own .txt file."""
    written: list[Path] = []
    for dimension_code in DIMENSIONS:
        written.extend(display_variants_for_dimension(dimension_code, output_dir))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="German LLM-as-judge prompt construction utilities.")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Render prompt variants for every dimension and save them as .txt files "
        "under data/assets/de_assets/de_prompts/<dimension_code>/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.display:
        for path in display_variants_for_all_dimensions():
            print(f"Saved: {path}")
    else:
        print("Nothing to do. Pass --display to render the prompt variants for a chosen dimension.")
