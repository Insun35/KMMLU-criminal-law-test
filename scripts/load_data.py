import json, os
from pathlib import Path
import requests
import xmltodict

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

KMMLU_DIR = Path("data/kmmlu")
KMMLU_DIR.mkdir(parents=True, exist_ok=True)
KMMLU_OUT_FILE = KMMLU_DIR / "kmmlu_test.json"

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

LAW_SERVICE_OC = os.getenv("LAW_SERVICE_OC")
LAW_SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
LAW_SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"

LAW_ARTICLES_OUT_FILE = RAW_DATA_DIR / "law_articles.jsonl"


def load_kmmlu_test(save: bool = True):
    print("🔄 Loading test set ...")
    test_set = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    if save:
        print("💾 Saving to JSON ...")
        with KMMLU_OUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(test_set.to_list(), f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully saved {len(test_set)} questions")
    return test_set


def find_law_list(law_name: str, display: int = 100):
    params = {
        "OC": LAW_SERVICE_OC,
        "target": "law",
        "Type": "json",
        "query": law_name,
        "display": display,
        "sort": "name",
    }
    resp = requests.get(LAW_SEARCH_URL, params=params)
    resp.raise_for_status()
    data = resp.json().get("LawSearch", [])
    if not data:
        print(f"Couldn't find any content corresponds to [{law_name}].")
        return

    return [item["법령일련번호"] for item in data["law"]]


def save_law_articles(law_list: list[str]):
    for law_id in tqdm(law_list, desc="💾 Saving law articles"):
        params = {
            "OC": LAW_SERVICE_OC,
            "target": "law",
            "Type": "xml",
            "MST": law_id,
        }
        resp = requests.get(LAW_SERVICE_URL, params=params)
        law_articles_dict = xmltodict.parse(resp.text)

        law = law_articles_dict.get("법령", {})
        with LAW_ARTICLES_OUT_FILE.open("a", encoding="utf-8") as f:
            basic = law.get("기본정보", {})
            full_title = basic.get("법령명_한글", "")
            short_title = basic.get("법령명약칭", "")
            article_title = short_title if short_title else full_title

            if short_title:
                custom_idx = law_id + "-" + "0"
                title_record = full_title.strip()
                title_record += "\n약칭: " + short_title.strip()
                record = {"id": custom_idx, "text": title_record}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            cm = law.get("조문", {}).get("조문단위")
            units = cm if isinstance(cm, list) else [cm]
            for unit in units:
                custom_idx = law_id + "-" + unit.get("조문번호", "").strip()
                no = unit.get("조문번호", "").strip()
                title = unit.get("조문제목", "").strip()
                # If there is no title for the same article number,
                # it will be a meaningless duplicate paragraph.
                if not title:
                    continue
                text = unit.get("조문내용", "").strip()
                flat_text = "\n".join(
                    filter(
                        None,
                        [
                            f"{article_title}",
                            f"제{no}조 {title}",
                            text,
                        ],
                    )
                )
                record = {"id": custom_idx, "text": flat_text}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Successfully saved {len(law_list)} law articles")


def main():
    # Save KMMLU criminal law test set
    load_kmmlu_test()

    # Save law contents from Law Service API
    print("🔄 Loading law articles ...")
    law_list = []
    for name in ["형법", "형사소송법"]:
        law_list.extend(find_law_list(name))
    save_law_articles(law_list)


if __name__ == "__main__":
    main()
