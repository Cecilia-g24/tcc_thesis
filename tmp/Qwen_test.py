"""Qwen-7B inference over the test prompts (prompts/test/p1_prompt.py).

A single `--mode` flag selects which evaluation to run; each mode writes its result
JSON files into its own folder under the model's results directory configured in
configs.jsonl (Datasets.results.Qwen.<mode>):

  --mode single-turn   (over the P1 single-query dataset, Datasets.instructions_final.p1)
      constraints_detect.json          — model extracts structured constraints from each
                                          query.                       -> [{id, constraints}]
      recommend_query_direct.json      — dish_gen WITHOUT constraints: recommend directly
                                          from the raw query.          -> [{id, recommendation}]
      recommend_self_constraints.json  — dish_gen from the constraints the model itself
                                          detected above.              -> [{id, recommendation}]
      recommend_std_constraints.json   — dish_gen from the standard (ground-truth)
                                          constraints.                 -> [{id, recommendation}]

  --mode multi-turn    (over the P2 multi-turn dataset, Datasets.instructions_final.p2 —
                        each item is {id, turns:[...]})
      constraints_detect.json          — turns are fed in order as a conversation; only the
                                          LAST turn's constraint extraction is kept.
                                                                       -> [{id, constraints}]
      recommend_query_direct.json      — dish_gen WITHOUT constraints; again only the LAST
                                          turn's recommendation is kept.
                                                                       -> [{id, recommendation}]

  --mode conflict      (over the P3 conflicting-query dataset, Datasets.instructions_final.p3)
      recommend_query_direct.json      — dish_gen from the raw (conflicting) query.
                                                                       -> [{id, recommendation}]

Each model call is batched so the model runs once per experiment over all items.
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

# Make project packages (Utils, Main, …) importable no matter where this script is
# launched from — find the repo root by its marker file, not by counting dirs.
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "configs.jsonl").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Utils.paths import CONFIG_FILE, repo_path
from Utils.gen import load_json, save_to_json, normalize_std_constraints
from Main.model.Qwen_llm import vllm_LLM
from Main.prompts.test.p1_prompt import (
    CONSTRAINS_DETECT,
    QUERY_BASED_DISH_RECOMMAND,
    CONSTRAINS_BASED_DISH_RECOMMAND,
    CONVERSATION_SYSTEM_PROMPT,
    MULTI_TURN_CONSTRAINS_DETECT,
    MULTI_TURN_DISH_RECOMMAND,
)

def _results_dir(cfg, model_key, mode):
    """Resolve the output folder for `model_key` + `mode` from the config.

    Prefers an explicit Datasets.results.<model_key>.<mode> entry; falls back to
    <results_root>/<model_key>/<mode> for any model without its own results block.
    """
    entry = cfg['Datasets']['results'].get(model_key)
    if isinstance(entry, dict) and mode in entry:
        return entry[mode]
    return os.path.join(cfg['Datasets']['results_root'], model_key, mode)


# --------------------------------------------------------------------------- #
# JSON parsing of model output
# --------------------------------------------------------------------------- #
def parse_json_object(raw):
    """Best-effort extraction of a single JSON object from a model response.

    Strips Markdown code fences and grabs the outermost {...}. Returns the parsed
    object on success, or None if nothing parseable is found.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Message builders. The recommend prompts carry literal JSON braces, so we fill
# their {Query}/{Constrains} slots with str.replace rather than str.format.
# --------------------------------------------------------------------------- #
def build_detect_messages(instruction):
    return [
        {"role": "system", "content": CONSTRAINS_DETECT},
        {"role": "user", "content": instruction},
    ]


def build_query_messages(instruction):
    content = QUERY_BASED_DISH_RECOMMAND.replace("{Query}", instruction)
    return [{"role": "user", "content": content}]


def build_constraints_messages(constraints):
    constraints_str = json.dumps(constraints, ensure_ascii=False)
    content = CONSTRAINS_BASED_DISH_RECOMMAND.replace("{Constrains}", constraints_str)
    return [{"role": "user", "content": content}]


def build_shared_histories(llm, items):
    """Replay every turn EXCEPT each conversation's last, letting the model actually
    reply at each intermediate turn.

    The system prompt (CONVERSATION_SYSTEM_PROMPT) keeps these intermediate replies as
    brief natural chat rather than JSON. The model's real reply is appended back into
    the history before the next user turn, so context accumulates genuinely turn by turn.
    Returns (histories, turns_list); the histories stop just before each conversation's
    last turn, which the caller asks separately with the task prompt.

    Conversations have varying length, so each turn-index is batched over only the
    conversations that still have an intermediate turn there.
    """
    histories = [[{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}] for _ in items]
    turns_list = [[t for t in it.get("turns", []) if str(t).strip()] for it in items]
    max_turns = max((len(t) for t in turns_list), default=0)

    for t in range(max_turns):
        # A turn is "intermediate" (gets a natural reply) when it is not the last turn
        # of its own conversation, i.e. t < len(turns) - 1.
        active = [i for i, turns in enumerate(turns_list) if t < len(turns) - 1]
        if not active:
            continue
        for i in active:
            histories[i].append({"role": "user", "content": turns_list[i][t]})
        replies = llm.batch_chat([histories[i] for i in active])
        for i, reply in zip(active, replies):
            histories[i].append({"role": "assistant", "content": reply})
        print(f"  intermediate turn {t}: {len(active)} conversations advanced ...")

    return histories, turns_list


def ask_final_turn(llm, histories, turns_list, final_request):
    """Append each conversation's last turn merged with the task prompt, then run one
    final model call. The shared `histories` are reused (not mutated) so both the
    constraint and dish experiments share the same intermediate conversation buildup.
    """
    conversations = []
    for hist, turns in zip(histories, turns_list):
        messages = list(hist)  # copy: leave the shared history untouched for reuse
        if turns:
            messages.append({"role": "user", "content": f"{turns[-1]}\n\n{final_request}"})
        else:
            messages.append({"role": "user", "content": final_request})
        conversations.append(messages)
    return llm.batch_chat(conversations)


# --------------------------------------------------------------------------- #
# Result collectors
# --------------------------------------------------------------------------- #
def _collect_constraints(ids, raw_outputs):
    """Map raw extraction responses onto [{id, constraints}] records.

    On parse failure the record keeps the raw text and flags the error so nothing
    is silently dropped.
    """
    results = []
    for iid, raw in zip(ids, raw_outputs):
        parsed = parse_json_object(raw)
        record = {"id": iid, "constraints": parsed if parsed is not None else {}}
        if parsed is None:
            record["parse_error"] = True
            record["raw"] = raw
        results.append(record)
    return results


def _collect_recommendations(ids, raw_outputs):
    """Map raw recommendation responses onto [{id, recommendation}] records.

    `recommendation` is the parsed JSON object ({"dish": [...]}); on parse failure the
    record keeps the raw text and flags the error so nothing is silently dropped.
    """
    results = []
    for iid, raw in zip(ids, raw_outputs):
        parsed = parse_json_object(raw)
        record = {"id": iid, "recommendation": parsed if parsed is not None else {}}
        if parsed is None:
            record["parse_error"] = True
            record["raw"] = raw
        results.append(record)
    return results


def _make_saver(output_dir):
    """Return a save helper that writes into `output_dir` (created if missing)."""
    output_dir = repo_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def _save(filename, data):
        path = os.path.join(output_dir, filename)
        save_to_json(data, path)
        print(f"  saved {len(data)} records -> {path}")

    return _save


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_single_turn(cfg, llm, args, model_key):
    """P1 single-query: constraint extraction + three dish_gen variants."""
    instructions = load_json(repo_path(cfg['Datasets']['instructions_final']['p1']))
    if args.limit is not None:
        instructions = instructions[:args.limit]
    combins = load_json(repo_path(cfg['Datasets']['combins']['p1']))

    # Normalise the standard constraints onto the flat detection schema so the gt-constraints
    # dish_gen feeds the recommend prompt the SAME shape as the self-detected constraints;
    # the only variable between the two then is constraint correctness, not format.
    std_constraints_by_id = {
        item["id"]: normalize_std_constraints(item["constraints"]) for item in combins
    }

    ids = [item["id"] for item in instructions]
    queries = [item.get("instruction", "") for item in instructions]
    save = _make_saver(_results_dir(cfg, model_key, 'single_turn'))

    # constraint extraction
    print(f"[single-turn] detecting constraints for {len(ids)} queries ...")
    detect_raw = llm.batch_chat([build_detect_messages(q) for q in queries])
    detect_results = _collect_constraints(ids, detect_raw)
    detected_by_id = {r["id"]: r["constraints"] for r in detect_results}
    save("constraints_detect.json", detect_results)

    # dish_gen without constraints (directly from the raw query)
    print(f"[single-turn] dish_gen without constraints (query direct) ...")
    query_raw = llm.batch_chat([build_query_messages(q) for q in queries])
    save("recommend_query_direct.json", _collect_recommendations(ids, query_raw))

    # dish_gen from the model's own self-extracted constraints
    print(f"[single-turn] dish_gen from self-extracted constraints ...")
    self_raw = llm.batch_chat(
        [build_constraints_messages(detected_by_id.get(iid, {})) for iid in ids]
    )
    save("recommend_self_constraints.json", _collect_recommendations(ids, self_raw))

    # dish_gen from the ground-truth (standard) constraints
    print(f"[single-turn] dish_gen from gt constraints ...")
    std_raw = llm.batch_chat(
        [build_constraints_messages(std_constraints_by_id.get(iid, {})) for iid in ids]
    )
    save("recommend_std_constraints.json", _collect_recommendations(ids, std_raw))


def run_multi_turn(cfg, llm, args, model_key):
    """P2 multi-turn: replay each conversation turn by turn with the model's own real
    replies, then ask the task only at the LAST turn and keep that result."""
    items = load_json(repo_path(cfg['Datasets']['instructions_final']['p2']))
    if args.limit is not None:
        items = items[:args.limit]
    ids = [item["id"] for item in items]
    save = _make_saver(_results_dir(cfg, model_key, 'multi_turn'))

    # Replay every turn but the last, letting the model genuinely respond each time;
    # the resulting history is shared by both experiments below.
    print(f"[multi-turn] replaying intermediate turns over {len(items)} conversations ...")
    histories, turns_list = build_shared_histories(llm, items)

    # constraint extraction — asked at the final turn, only its result is kept
    print(f"[multi-turn] extracting constraints at the final turn ...")
    constr_raw = ask_final_turn(llm, histories, turns_list, MULTI_TURN_CONSTRAINS_DETECT)
    save("constraints_detect.json", _collect_constraints(ids, constr_raw))

    # dish_gen without constraints — asked at the final turn, only its result is kept
    print(f"[multi-turn] dish_gen without constraints at the final turn ...")
    dish_raw = ask_final_turn(llm, histories, turns_list, MULTI_TURN_DISH_RECOMMAND)
    save("recommend_query_direct.json", _collect_recommendations(ids, dish_raw))


def run_conflict(cfg, llm, args, model_key):
    """P3 conflicting-query: dish_gen directly from the raw (conflicting) query."""
    items = load_json(repo_path(cfg['Datasets']['instructions_final']['p3']))
    if args.limit is not None:
        items = items[:args.limit]
    ids = [item["id"] for item in items]
    queries = [item.get("instruction", "") for item in items]
    save = _make_saver(_results_dir(cfg, model_key, 'conflict'))

    print(f"[conflict] dish_gen over {len(items)} conflicting queries ...")
    query_raw = llm.batch_chat([build_query_messages(q) for q in queries])
    save("recommend_query_direct.json", _collect_recommendations(ids, query_raw))


MODES = {
    "single-turn": run_single_turn,
    "multi-turn": run_multi_turn,
    "conflict": run_conflict,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = load_json(CONFIG_FILE)
    model_names = list(cfg['Models'].keys())
    model_choices = "\n".join(f"    {i}: {name}" for i, name in enumerate(model_names))

    parser = argparse.ArgumentParser(
        description="Qwen inference over the test-prompt experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=list(MODES),
                        help="Which evaluation to run: 'single-turn' (constraint extraction "
                             "+ 3 dish_gen variants over P1), 'multi-turn' (extraction + "
                             "dish_gen over P2, keeping only the last turn), or 'conflict' "
                             "(dish_gen over the P3 conflicting queries).")
    parser.add_argument("--model", type=int, default=0, metavar="INDEX",
                        help="Index of the model to use from the config Models block; its "
                             "model_path / llm / sampling params are taken automatically.\n"
                             f"{model_choices}")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on the number of items (for quick tests).")
    args = parser.parse_args()

    if not 0 <= args.model < len(model_names):
        parser.error(f"--model must be in [0, {len(model_names) - 1}]; got {args.model}. "
                     f"Available models:\n{model_choices}")
    model_key = model_names[args.model]
    print(f"Using model [{args.model}] {model_key}")

    # Model + decoding hyperparameters come from the selected Models.<name> block.
    # Each model's sampling profile should be tuned for clean structured output:
    # deterministic, penalty-free decoding with room for multi-dish JSON (penalties /
    # a short max_tokens corrupt the JSON these prompts must emit — see
    # Main/ig/eval/instruction_filter.py).
    model_cfg = cfg['Models'][model_key]
    llm = vllm_LLM(
        model_path=model_cfg['model_path'],
        sampling_parameters=model_cfg['sampling'],
        llm_parametres=model_cfg['llm'],
    )

    MODES[args.mode](cfg, llm, args, model_key)


if __name__ == "__main__":
    main()
