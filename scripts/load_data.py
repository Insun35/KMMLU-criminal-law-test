import re, json, os
from datasets import load_dataset, Dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm

KMMLU_DIR = Path("data/kmmlu")
KMMLU_DIR.mkdir(parents=True, exist_ok=True)
KMMLU_OUT_FILE = KMMLU_DIR / "kmmlu_test.json"

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
LBOX_OUT_FILE = RAW_DATA_DIR / "lbox_criminal.jsonl"


def load_kmmlu_test(save: bool = True):
    print("🔄 Loading test set ...")
    test_set = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    if save:
        print("💾 Saving to JSON ...")
        with KMMLU_OUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(test_set.to_list(), f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully saved {len(test_set)} questions")
    return test_set


def looks_criminal(text: str) -> bool:
    return ("피고인" in text or "검사" in text) and ("징역" in text or "벌금" in text)


def load_lbox_data():
    # Load LJP criminal data
    print("🔄 Loading ljp criminal data ...")
    ljp_criminal_data = load_dataset("lbox/lbox_open", "ljp_criminal", split="train")

    # Load precedent corpus data
    print("🔄 Loading precedent corpus data ...")
    precedent_corpus_data = load_dataset(
        "lbox/lbox_open", "precedent_corpus", split="train"
    )

    with LBOX_OUT_FILE.open("w", encoding="utf-8") as f:
        for ex in tqdm(ljp_criminal_data, desc="Writing ljp_criminal"):
            combined_text = ex["facts"].strip() + "\n" + ex["reason"].strip()
            record = {"source": "ljp_criminal", "id": ex["id"], "text": combined_text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for ex in tqdm(precedent_corpus_data, desc="Filtering precedent_corpus"):
            txt = ex["precedent"]
            if looks_criminal(txt):
                record = {
                    "source": "precedent_corpus",
                    "id": ex["id"],
                    "text": txt.strip(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Successfully saved LBOX criminal data")


if __name__ == "__main__":
    # load_kmmlu_test()
    load_lbox_data()
