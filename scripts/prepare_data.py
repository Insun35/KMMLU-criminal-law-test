import os
import json
import time
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


def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        block = list(islice(it, n))
        if not block:
            return
        yield block


def chunk_text(text: str, max_chars: int = 2500) -> list[str]:
    import re

    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks


def load_and_chunk():
    all_chunks = []
    with RAW_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading & chunking"):
            text = json.loads(line)["text"]
            all_chunks.extend(chunk_text(text))
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def build_batch_input(chunks: list[str]):
    with LBOX_BATCH_INPUT.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1):
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


def submit_batch(client: OpenAI):
    print("Uploading batch input file to OpenAI…")
    upload_resp = client.files.create(
        file=open(LBOX_BATCH_INPUT, "rb"), purpose="batch"
    )
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
        if status.status in ("succeeded", "failed", "cancelled"):
            return status
        time.sleep(BATCH_POLL_INTERVAL)


def download_results(client: OpenAI, status):
    if status.status != "succeeded":
        raise RuntimeError(f"Batch job did not succeed: {status.status}")
    file_id = status.output_files[0]
    print("Downloading results file:", file_id)
    content = client.files.content(file_id).text
    with LBOX_BATCH_OUTPUT.open("w", encoding="utf-8") as f:
        f.write(content)
    print("Batch output JSONL saved.")


def build_faiss_index():
    # Load chunks
    chunks = json.loads(LBOX_CHUNKS_FILE.read_text(encoding="utf-8"))
    # Parse embeddings from batch output
    records = [
        json.loads(line) for line in LBOX_BATCH_OUTPUT.open("r", encoding="utf-8")
    ]
    # Sort by custom_id to align with chunks
    records.sort(key=lambda r: int(r["custom_id"].split("-")[1]))
    embeddings = [r["response"]["body"]["data"][0]["embedding"] for r in records]
    # Build index
    vecs_np = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(vecs_np.shape[1])
    index.add(vecs_np)
    faiss.write_index(index, str(EMBED_DIR / "lbox_index.faiss"))
    print("FAISS index built and saved.")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 1) Chunk
    chunks = load_and_chunk()
    # 2) Batch input
    build_batch_input(chunks)
    # 3) Submit
    batch_id = submit_batch(client)
    # 4) Wait
    status = wait_for_batch(client, batch_id)
    # 5) Download
    download_results(client, status)
    # 6) FAISS
    build_faiss_index()
    print("All done!")


if __name__ == "__main__":
    main()
