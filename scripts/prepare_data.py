import json
from datasets import load_dataset
from pathlib import Path

KMMLU_PATH = Path("data/kmmlu")
KMMLU_PATH.mkdir(parents=True, exist_ok=True)
KMMLU_OUT_FILE = KMMLU_PATH / "kmmlu_test.json"


def save_to_json(data, out_file: Path):
    print("💾 Saving to JSON ...")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_kmmlu_test(save: bool = False):
    print("🔄 Loading test set ...")
    test_set = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    print(f"✅ Successfully loaded {len(test_set)} questions")
    if save:
        save_to_json(test_set.to_list(), KMMLU_OUT_FILE)
    return test_set


if __name__ == "__main__":
    load_kmmlu_test(save=True)
