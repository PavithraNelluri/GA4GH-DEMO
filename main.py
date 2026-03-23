import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from embeddings import get_embeddings, get_pinecone_index
from sparse import load_bm25

load_dotenv()

st.set_page_config(page_title="GA4GH Q&A Bot", layout="wide")
st.title("GA4GH Q&A Bot")

@st.cache_resource
def initialize():
    embeddings = get_embeddings()
    index = get_pinecone_index()
    bm25 = load_bm25()
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        text_key="text",
        alpha=0.7
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )
    return rag_chain
rag_chain = initialize()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your question:")

if st.button("Ask") and query:
    with st.spinner("Thinking... "):
        response = rag_chain.invoke(query)
        answer = response["result"]
        sources = response.get("source_documents", [])
        st.session_state.history.append(("You", query))
        st.session_state.history.append(("Bot", answer))
        st.session_state.history.append(("Sources", sources))
for item in st.session_state.history:
    role, msg = item

    if role == "You":
        st.markdown(f"** {role}:** {msg}")

    elif role == "Bot":
        st.markdown(f"**{role}:** {msg}")

    elif role == "Sources":
        with st.expander("Sources"):
            for doc in msg:
                st.write(doc.page_content[:200] + "...")
