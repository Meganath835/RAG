"""
Check what is stored in ChromaDB (run store_in_chroma.py first).

Run:  python check_chroma.py
"""
import chromadb

client = chromadb.PersistentClient(path="chroma_db")

# 1. Which collections exist?
print("Collections:", [c.name for c in client.list_collections()])

collection = client.get_collection("rag_chunks")

# 2. How many chunks are stored?
print("Total chunks stored:", collection.count())

# 3. How many came from each file?
for file_type in ["pdf", "xlsx"]:
    found = collection.get(where={"type": file_type})
    print(f"  {file_type}: {len(found['ids'])} chunks")

# 4. Look at a few stored entries: id, metadata, text, and the stored vector
items = collection.get(limit=3, include=["documents", "metadatas", "embeddings"])
for id_, doc, meta, vec in zip(items["ids"], items["documents"], items["metadatas"], items["embeddings"]):
    print(f"\n--- {id_}  {meta} ---")
    print(f"text   : {doc[:120]}...")
    print(f"vector : {len(vec)} dims, first 5 = {[round(float(v), 4) for v in vec[:5]]}")

# 5. Fetch one specific chunk by id
one = collection.get(ids=["xlsx-1"])
print("\nChunk xlsx-1:\n" + one["documents"][0])
