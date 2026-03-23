from sparse import create_and_save_bm25
from embeddings import get_embeddings, get_pinecone_index
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from pdf_preprocess import load_documents

# Prepare corpus
documents = load_documents("GA4GH-DEMO/documents.json")
corpus = [doc.page_content for doc in documents]

#1.Save BM25
create_and_save_bm25(corpus)

#2.Load BM25
bm25 = BM25Encoder().load("GA4GH-DEMO/bm25_values.json")

#3.Setup retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=get_embeddings(),
    sparse_encoder=bm25,
    index=get_pinecone_index(),
    text_key="text"
)
retriever.add_texts(corpus)