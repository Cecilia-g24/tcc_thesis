"""
1. Fetch available models from NHR and GWDG providers via their /models endpoints.
2. Test connectivity for every fetched model.
3. Save only the models that passed the connectivity test to configs/api_models.json.

Run from the repo root:  python scripts/approach_2_llm/fetch_and_test_llm_api.py
"""

from openai import OpenAI
from dotenv import load_dotenv
import os, json, time, datetime
from pathlib import Path

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_PATH = REPO_ROOT / "configs" / "api_models.json"

PROVIDERS = {
    "nhr": {
        "api_key": os.getenv("NHR_API_KEY"),
        "base_url": os.getenv("NHR_BASE_URL"),
    },
    "gwdg": {
        "api_key": os.getenv("GWDG_API_KEY"),
        "base_url": os.getenv("GWDG_BASE_URL"),
    },
}

# Model ID substrings that identify non-chat models to skip
EXCLUDE_KEYWORDS = {"embed", "bge-", "e5-", "ocr", "rerank", "whisper", "clip"}

TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Reply with exactly one sentence confirming you are working."},
]


def _client(provider_name: str) -> OpenAI:
    cfg = PROVIDERS[provider_name]
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=30.0)


def is_chat_model(model_id: str) -> bool:
    mid = model_id.lower()
    return not any(kw in mid for kw in EXCLUDE_KEYWORDS)


def fetch_models(provider_name: str) -> dict[str, str]:
    """Return {model_id: model_id} for all chat-capable models at this provider."""
    client = _client(provider_name)
    models: dict[str, str] = {}
    try:
        for model in client.models.list().data:
            if is_chat_model(model.id):
                models[model.id] = model.id
        print(f"  {provider_name.upper()}: {len(models)} chat models found")
    except Exception as e:
        print(f"  {provider_name.upper()}: ERROR fetching models — {e}")
    return models


def save_models(api_models: dict[str, dict[str, str]]) -> None:
    today = str(datetime.date.today())
    output: dict = {}
    for provider_name, models in api_models.items():
        base = PROVIDERS[provider_name]["base_url"] or ""
        output[provider_name] = {
            "_comment": (
                f"{provider_name.upper()} models — auto-fetched {today}. "
                "Embedding/non-chat and failed-connectivity models excluded."
            ),
            "_source": base.rstrip("/") + "/models",
            **models,
        }
    MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {MODELS_PATH}\n")


def test_model(provider_name: str, model_id: str) -> dict:
    client = _client(provider_name)
    result = {
        "provider": provider_name,
        "model_id": model_id,
        "status": None,
        "response": None,
        "error": None,
        "latency_s": None,
    }
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=TEST_MESSAGES,
            temperature=0.1,
            max_tokens=64,
        )
        result["latency_s"] = round(time.time() - t0, 2)
        content = resp.choices[0].message.content
        result["response"] = content.strip() if content else ""
        result["status"] = "OK"
    except Exception as e:
        result["latency_s"] = round(time.time() - t0, 2)
        result["status"] = "FAIL"
        result["error"] = str(e)
    return result


def main() -> None:
    # ── Step 1: fetch candidate models ────────────────────────────────────────
    print("=" * 60)
    print("Fetching available models from providers...")
    print("=" * 60)
    candidates: dict[str, dict[str, str]] = {}
    for provider_name in PROVIDERS:
        candidates[provider_name] = fetch_models(provider_name)

    total = sum(len(v) for v in candidates.values())
    print(f"\nTesting {total} candidate models across {len(candidates)} providers\n")

    # ── Step 2: test connectivity for every candidate ─────────────────────────
    results: list[dict] = []
    for provider_name, models in candidates.items():
        print(f"{'=' * 60}")
        print(f"Provider: {provider_name.upper()}  |  {PROVIDERS[provider_name]['base_url']}")
        print(f"{'=' * 60}")
        for model_id in models:
            print(f"  {model_id} ...", end=" ", flush=True)
            r = test_model(provider_name, model_id)
            results.append(r)
            if r["status"] == "OK":
                print(f"OK ({r['latency_s']}s)")
                print(f"    {r['response'][:120]}")
            else:
                print(f"FAIL ({r['latency_s']}s)")
                print(f"    {r['error'][:200]}")

    # ── Step 3: keep only working models and save ─────────────────────────────
    working: dict[str, dict[str, str]] = {provider_name: {} for provider_name in PROVIDERS}
    for r in results:
        if r["status"] == "OK":
            working[r["provider"]][r["model_id"]] = r["model_id"]
    save_models(working)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"Summary: {ok}/{len(results)} models OK (only these were saved to {MODELS_PATH})\n")
    for r in results:
        mark = "OK  " if r["status"] == "OK" else "FAIL"
        print(f"  {mark}  [{r['provider']}] {r['model_id']:<55} {r['latency_s']}s")


if __name__ == "__main__":
    main()
