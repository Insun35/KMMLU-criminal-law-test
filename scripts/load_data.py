import json
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

KMMLU_PATH = Path("data/kmmlu")
KMMLU_PATH.mkdir(parents=True, exist_ok=True)
KMMLU_OUT_FILE = KMMLU_PATH / "kmmlu_test.json"

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
LBOX_OUT_FILE = RAW_DATA_PATH / "ljp_criminal.jsonl"


def load_kmmlu_test(save: bool = True):
    print("🔄 Loading test set ...")
    test_set = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    if save:
        save_to_json(test_set.to_list(), KMMLU_OUT_FILE)
        print(f"✅ Successfully saved {len(test_set)} questions")
    return test_set


def load_ljp_criminal_data(save: bool = True):
    print("🔄 Loading ljp criminal data ...")
    ljp_criminal_data = load_dataset("lbox/lbox_open", "ljp_criminal", split="train")
    if save:
        with LBOX_OUT_FILE.open("w", encoding="utf-8") as f:
            for case in tqdm(ljp_criminal_data, desc="Saving to JSONL"):
                # Put facts and reason in the same line to make it informative
                text = f"{case['facts'].strip()}\n{case['reason'].strip()}"
                json_line = json.dumps(
                    {"id": case["id"], "text": text}, ensure_ascii=False
                )
                f.write(json_line + "\n")
        print(f"✅ Successfully saved {len(ljp_criminal_data)} data samples")
    return ljp_criminal_data


if __name__ == "__main__":
    load_kmmlu_test()
    load_ljp_criminal_data()
