"""
Recursive chunking of a PDF with LangChain.

Install:  pip install langchain-community langchain-text-splitters pypdf
Run:      python recursive_chunking.py
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "leave_policy_plan.pdf"


def get_pdf_chunks(path=PDF_PATH):
    # 1. Load the PDF (one Document per page, with page number in metadata)
    pages = PyPDFLoader(path).load()

    # 2. Recursive splitter: tries to split on paragraphs first, then lines, then words,
    #    then characters, until every chunk fits within chunk_size.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # max characters per chunk
        chunk_overlap=50,     # characters shared between neighbouring chunks
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(pages)


if __name__ == "__main__":
    # 3. Display the chunks
    chunks = get_pdf_chunks()
    print(f"Created {len(chunks)} chunks from {PDF_PATH}\n")
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", 0) + 1
        print(f"===== Chunk {i}  (page {page}, {len(chunk.page_content)} chars) =====")
        print(chunk.page_content)
        print()
