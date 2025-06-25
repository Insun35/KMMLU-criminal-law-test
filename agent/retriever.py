import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI

EMBED_DIR = Path("data/embeddings")
INDEX_FILE = EMBED_DIR / "ljp_index.faiss"
CHUNKS_FILE = EMBED_DIR / "ljp_text_chunks.json"


class Retriever:
    def __init__(self, api_key: str, k: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.index = faiss.read_index(str(INDEX_FILE))
        self.chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        self.k = k

    def get_query_embedding(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model="text-embedding-3-small", input=[query], encoding_format="float"
        )
        return np.array(resp.data[0].embedding, dtype="float32")

    def retrieve(self, query: str) -> list[str]:
        q_vec = self.get_query_embedding(query)
        D, I = self.index.search(q_vec.reshape(1, -1), self.k)
        return [self.chunks[i] for i in I[0]]
