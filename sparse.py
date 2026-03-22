from pinecone_text.sparse import BM25Encoder
import os

BM25_FILE = "bm25_values.json"

def create_and_save_bm25(corpus):
    bm25 = BM25Encoder().default()
    bm25.fit(corpus)
    bm25.dump(BM25_FILE)
    print("BM25 saved!")

def load_bm25():
    if not os.path.exists(BM25_FILE):
        raise FileNotFoundError(
            "bm25_values.json not found.Please run create_and_save_bm25()."
        )

    return BM25Encoder().load(BM25_FILE)
