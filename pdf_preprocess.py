from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
def pdf_to_document(pdf_url):
    loader = PyPDFLoader(pdf_url)
    pages = loader.load()
    pages = pages[:3]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(pages)
    documents = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
    return documents
def save_documents(documents, file_path="documents.json"):
    data = []
    
    for doc in documents:
        data.append({
            "page_content": doc.page_content,
            "page_no": doc.metadata.get("page", i)# i is fallback value
        })
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Documents saved!")
def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [
        Document(
            page_content=d["page_content"],
            metadata=d.get("metadata", {})
        )
        for d in data
    ]

    return documents

documents=pdf_to_document("GA4GH-DEMO/Framework-Version-3September20191.pdf")
save_documents(documents)
