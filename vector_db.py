from sentence_transformers import SentenceTransformer
import chromadb

# Your docs look like:
# docs = [{"text": "...", "meta": {"title": "...", "chunk_index": 0}}, ...]

def upsert_docs_to_chroma(
    docs,
    persist_dir="chroma_db",
    collection_name="matrix_bzd",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=256,
):
    # 1) Embedding model
    model = SentenceTransformer(model_name)

    # 2) Chroma persistent client
    client = chromadb.PersistentClient(path=persist_dir)

    # 3) Create/get collection
    # Note: we're passing precomputed embeddings, so no embedding_function needed.
    collection = client.get_or_create_collection(name=collection_name)

    # 4) Prepare data
    texts = [d["text"] for d in docs]
    metas = [d.get("meta", {}) for d in docs]

    # Stable unique ids (title + chunk_index). Adjust if you can have duplicates.
    ids = [
        f'{m.get("title","untitled")}::chunk::{m.get("chunk_index",-1)}'
        for m in metas
    ]

    # 5) Batch embed + upsert
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_metas = metas[start:end]
        batch_ids = ids[start:end]

        # embeddings: list[list[float]] (dim=384 for MiniLM-L6)
        batch_emb = model.encode(
            batch_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine-friendly
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=batch_emb,
        )

    return collection

# Usage:
# collection = upsert_docs_to_chroma(docs)
