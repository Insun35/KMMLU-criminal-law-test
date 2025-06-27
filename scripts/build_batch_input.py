import json
import os
from pathlib import Path
from tqdm import tqdm
from agent import prompt_tpl, llm

BATCH_DIR = Path("data/batch")
BATCH_DIR.mkdir(parents=True, exist_ok=True)
BATCH_IN = BATCH_DIR / "eval_input.jsonl"


def main():
    test_set = json.load(open("data/kmmlu/kmmlu_test.json", "r", encoding="utf-8"))
    agent = llm.Agent(os.getenv("OPENAI_API_KEY"))
    with BATCH_IN.open("w", encoding="utf-8") as f:
        for idx, q in enumerate(tqdm(test_set, desc="Building batch input"), start=1):
            context_chunks = agent.retriever.retrieve(q["question"])
            context = "\n\n".join(context_chunks)
            prompt = prompt_tpl.PROMPT_TPL.format(
                question=q["question"],
                A=q["A"],
                B=q["B"],
                C=q["C"],
                D=q["D"],
                context=context,
            )
            record = {
                "custom_id": f"q{idx:04d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 1,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("✅ Batch input file created.")


if __name__ == "__main__":
    main()
