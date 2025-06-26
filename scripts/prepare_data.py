import os
import json
import time
import re
from pathlib import Path
from itertools import islice
from tqdm import tqdm

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAW_FILE = Path("data/raw/lbox_criminal.jsonl")
BATCH_DIR = Path("data/openai_batch")
BATCH_DIR.mkdir(parents=True, exist_ok=True)
LBOX_BATCH_INPUT = BATCH_DIR / "lbox_batch_input.jsonl"
LBOX_BATCH_OUTPUT = BATCH_DIR / "lbox_batch_output.jsonl"

EMBED_DIR = Path("data/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
LBOX_CHUNKS_FILE = EMBED_DIR / "lbox_text_chunks.json"

BATCH_POLL_INTERVAL = 60  # seconds between status checks
EMBED_MODEL = "text-embedding-3-small"

MAX_BATCH_SIZE = 10000


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks


def load_and_chunk():
    all_chunks = []
    with RAW_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading & chunking"):
            text = json.loads(line)["text"]
            all_chunks.extend(chunk_text(text))
    return all_chunks


def split_chunks(chunks: list[str], max_size: int = 10000) -> list[list[str]]:
    return [chunks[i : i + max_size] for i in range(0, len(chunks), max_size)]


def build_batch_input(chunks: list[str], part: int):
    path = BATCH_DIR / f"lbox_batch_input_{part}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1 + part * MAX_BATCH_SIZE):
            custom_id = f"chunk-{idx:05d}"
            record = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": EMBED_MODEL, "input": chunk},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Save the chunks list to disk for later alignment
    with LBOX_CHUNKS_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print("Batch input JSONL and chunk list saved.")
    return path


def submit_batch(client: OpenAI, input_path: Path):
    print("Uploading batch input file to OpenAI…")
    upload_resp = client.files.create(file=open(input_path, "rb"), purpose="batch")
    input_file_id = upload_resp.id
    print("Uploaded. input_file_id =", input_file_id)

    print("Creating batch embedding job…")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
    )
    print("Batch job created. batch_id =", batch.id)
    return batch.id


def wait_for_batch(client: OpenAI, batch_id: str):
    print("Polling batch status...")
    while True:
        status = client.batches.retrieve(batch_id)
        print("Status:", status.status)
        if status.status in ("completed", "failed", "cancelled"):
            return status
        time.sleep(BATCH_POLL_INTERVAL)


def download_results(client: OpenAI, status, part: int):
    if status.status != "completed":
        raise RuntimeError(f"Batch job did not complete: {status.status}")
    output_path = BATCH_DIR / f"lbox_batch_output_{part}.jsonl"
    file_id = status.output_file_id[0]
    print("Downloading results file:", file_id)
    content = client.files.content(file_id).text
    with output_path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"Batch output part {part} JSONL saved.")
    return output_path


def build_faiss_index():
    all_records = []
    # Read all records from batch output files
    for part_file in sorted(BATCH_DIR.glob("lbox_batch_output_*.jsonl")):
        for line in open(part_file, encoding="utf-8"):
            all_records.append(json.loads(line))
    # Sort by custom_id to align with chunks
    all_records.sort(key=lambda r: int(r["custom_id"].split("-")[1]))
    # Parse embeddings from batch output
    embeddings = [r["response"]["body"]["data"][0]["embedding"] for r in all_records]
    # Build index
    vecs_np = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(vecs_np.shape[1])
    index.add(vecs_np)
    faiss.write_index(index, str(EMBED_DIR / "lbox_index.faiss"))
    print("FAISS index built and saved.")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 1) Generate entire chunks
    chunks = load_and_chunk()
    total_chunks = len(chunks)
    print(f"Total chunks created: {total_chunks}")

    # 2) Split into parts
    parts = split_chunks(chunks)
    for i, part_chunks in enumerate(parts):
        # 3) Build batch input
        input_path = build_batch_input(part_chunks, i)
        print(f"Part {i}: input -> {input_path.name} ({len(part_chunks)} requests)")
        # 4) Submit batch
        batch_id = submit_batch(client, input_path)
        print(f"Part {i}: batch -> {batch_id}")
        # 5) Wait for batch
        status = wait_for_batch(client, batch_id)
        print(f"Part {i}: status -> {status.status}")
        # 6) Download results
        output_path = download_results(client, status, i)
        print(f"Part {i}: output -> {output_path.name} ({status.status})")
    # 7) Build FAISS index
    build_faiss_index()
    print("✅ All done!")


if __name__ == "__main__":
    main()
