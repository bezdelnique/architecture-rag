from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("kb")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q = "Who is Yan04ka?"
# q = "Who is 0lezeq?"
# q = "What relations between 0lezeq and Yan04ka?"
# q = "Who love 0lezeq?"
# q = "Who love Yan04ka"
# q = "Where 0lezeq come from?"
# q = "Who is 0lezeq enemy?"
q_emb = model.encode([q], normalize_embeddings=True).tolist()

res = collection.query(
    query_embeddings=q_emb,
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(dist, meta, doc[:120], "...")
