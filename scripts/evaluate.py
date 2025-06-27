import os
from dotenv import load_dotenv
import json
import time
from pathlib import Path

from openai import OpenAI
from datasets import load_dataset

# Configuration
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BATCH_DIR = Path("data/batch")
EVAL_INPUT = BATCH_DIR / "eval_input.jsonl"
EVAL_OUTPUT = BATCH_DIR / "eval_output.jsonl"
SCORE_FILE = Path("data/score.txt")

POLL_INTERVAL = 60


def submit_eval_batch(client: OpenAI):
    # Upload eval input file for batch processing
    print("Uploading eval input file...")
    upload_resp = client.files.create(file=open(EVAL_INPUT, "rb"), purpose="batch")
    input_file_id = upload_resp.id
    print("Uploaded. input_file_id =", input_file_id)
    # Create batch job for chat completions
    print("Creating eval batch job...")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print("Eval batch job created. batch_id =", batch.id)
    return batch.id


def poll_until_done(client: OpenAI, batch_id: str):
    print("Polling eval batch status...")
    while True:
        status = client.batches.retrieve(batch_id)
        print("Status:", status.status)
        if status.status in ("completed", "failed", "cancelled"):
            return status
        time.sleep(POLL_INTERVAL)


def download_eval_results(client: OpenAI, status):
    if status.status != "completed":
        raise RuntimeError(f"Eval batch did not complete: {status.status}")
    output_file_id = status.output_file_id
    print("Downloading eval results file:", output_file_id)
    content = client.files.content(output_file_id).text
    with EVAL_OUTPUT.open("w", encoding="utf-8") as f:
        f.write(content)
    print("Eval output JSONL saved.")


def compute_accuracy():
    # Load test set labels
    ds = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    id_to_label = {}
    for idx, ex in enumerate(ds, start=1):
        custom_id = f"q{idx:04d}"
        id_to_label[custom_id] = ex["answer"]

    # Read predictions
    total = 0
    correct = 0
    with EVAL_OUTPUT.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            custom_id = rec["custom_id"]
            # Extract predicted answer (first character, uppercased)
            raw = rec["response"]["body"]["choices"][0]["message"]["content"]
            pred = raw.strip().upper()[0]
            label = id_to_label.get(custom_id)
            total += 1
            if int(pred) == label:
                correct += 1
    accuracy = correct / total * 100 if total else 0.0
    result = f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%"
    print("\n✅", result)
    # Save to score file
    with SCORE_FILE.open("w", encoding="utf-8") as f:
        f.write(result + "\n")
    print(f"Score written to {SCORE_FILE}")


def main():
    # 1) Submit eval batch
    batch_id = submit_eval_batch(client)
    # 2) Poll status
    status = poll_until_done(client, batch_id)
    # 3) Download results
    download_eval_results(client, status)
    # 4) Compute accuracy
    compute_accuracy()


if __name__ == "__main__":
    main()
