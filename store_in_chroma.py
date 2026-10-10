"""
Embed the PDF + XLSX chunks with MiniLM and store them in ChromaDB.

Install:  pip install chromadb langchain-community langchain-text-splitters pypdf openpyxl
Run:      python store_in_chroma.py
"""
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from recursive_chunking import get_pdf_chunks
from recursive_chunking_xlsx import get_xlsx_chunks

# 1. Get the chunks from both files
pdf_chunks = get_pdf_chunks()
xlsx_chunks = get_xlsx_chunks()
print(f"PDF chunks: {len(pdf_chunks)}   XLSX chunks: {len(xlsx_chunks)}")

ids, texts, metadatas = [], [], []
for i, chunk in enumerate(pdf_chunks, start=1):
    ids.append(f"pdf-{i}")
    texts.append(chunk.page_content)
    metadatas.append({"source": chunk.metadata["source"], "type": "pdf", "page": chunk.metadata["page"] + 1})
for i, chunk in enumerate(xlsx_chunks, start=1):
    ids.append(f"xlsx-{i}")
    texts.append(chunk.page_content)
    metadatas.append({"source": chunk.metadata["source"], "type": "xlsx", "sheet": chunk.metadata["sheet"]})

# 2. Convert chunks to vectors. Chroma's default embedding function is all-MiniLM-L6-v2
#    (run through ONNX, so no PyTorch needed) -> one 384-number vector per chunk.
embed = DefaultEmbeddingFunction()
vectors = embed(texts)
print(f"Created {len(vectors)} vectors, each with {len(vectors[0])} dimensions\n")
for id_, vec in list(zip(ids, vectors))[:3]:
    print(f"{id_:<8} first 5 values: {[round(float(v), 4) for v in vec[:5]]}")
print("...\n")

# 3. Store chunks + vectors in a ChromaDB collection saved on disk in ./chroma_db
client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("rag_chunks")  # start fresh on every run
except Exception:
    pass
collection = client.create_collection(
    name="rag_chunks",
    embedding_function=embed,               # used to embed query text later
    configuration={"hnsw": {"space": "cosine"}},
)
collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
print(f"Stored {collection.count()} chunks in ChromaDB collection 'rag_chunks' (./chroma_db)\n")

# 4. Quick check: search the collection with a question
query = "How many days of sick leave do employees get?"
results = collection.query(query_texts=[query], n_results=3)
print(f"Query: {query}")
for id_, doc, meta, dist in zip(results["ids"][0], results["documents"][0],
                                results["metadatas"][0], results["distances"][0]):
    print(f"\n--- {id_}  (similarity {1 - dist:.3f}, {meta}) ---")
    print(doc)
