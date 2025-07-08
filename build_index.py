# build_index.py
import os
import frontmatter
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


# ✅ New way (no Settings needed)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("eol_products")


model = SentenceTransformer("all-MiniLM-L6-v2")

def convert_to_str(metadata):
    return {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for k, v in metadata.items()}

def batch_insert(collection, documents, embeddings, metadatas, ids, batch_size=5000):
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_embeds = embeddings[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"✅ Inserted {i+len(batch_docs)} / {len(documents)} entries")



def load_eol():
    entries = []
    for file in Path("endoflife.date/products").glob("*.md"):
        post = frontmatter.load(file)
        product = post.get("title", file.stem)
        for release in post.get("releases", []):
            cycle = release.get("releaseCycle", "unknown")
            release_date = release.get("releaseDate", "unknown")
            eol = release.get("eol", "TBD" if release.get("eol") is False else release.get("eol", "unknown"))
            desc = f"{product} {cycle}, released on {release_date}, reaches EOL on {eol}."
            entries.append({
                "id": f"{product}_{cycle}".lower().replace(" ", "_"),
                "text": desc,
                "meta": {
                    "product": product,
                    "cycle": cycle,
                    "release": release_date,
                    "eol": eol,
                }
            })
    return entries

entries = load_eol()
documents = [e["text"] for e in entries]
ids = [e["id"] for e in entries]
metas = [convert_to_str(e["meta"]) for e in entries]
embeds = model.encode(documents).tolist()

# collection.add(documents=documents, metadatas=metas, ids=ids, embeddings=embeds)
# Then call:
batch_insert(collection, documents, embeds, metas, ids)

print(f"✅ Indexed {len(entries)} EOL entries.")
