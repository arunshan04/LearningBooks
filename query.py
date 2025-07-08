# query.py
from sentence_transformers import SentenceTransformer
import sys
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("eol_products")


model = SentenceTransformer("all-MiniLM-L6-v2")

query = " ".join(sys.argv[1:]) or input("🔍 Enter product & version: ")

embedding = model.encode([query]).tolist()
results = collection.query(query_embeddings=embedding, n_results=1)

# Strict match threshold
score = results["distances"][0][0]
doc = results["documents"][0][0]
meta = results["metadatas"][0][0]

if score < 0.35:  # Lower is better in L2 distance (Chroma)
    print("❌ Product not found. Please double-check the name/version.")
else:
    print(f"✅ {meta['product']} {meta['cycle']}")
    print(f"📆 Released: {meta['release']}")
    print(f"📅 End of Life: {meta['eol']}")
    print(f"📝 Details: {doc}")
