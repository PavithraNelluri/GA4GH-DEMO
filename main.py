import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
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
    prompt_template = """
                    You are a strict question-answering assistant.
                    Rules:
                    1. If the question is irrelevant to the context OR the answer is not found, respond EXACTLY:
                       "I have knowledge only related to the first 3 pages of the GA4GH Framework document. Please ask questions based on it."
                    2. Do NOT guess or make up answers.
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Answer:
                    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return rag_chain
rag_chain = initialize()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask your question...")

if query:
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
        st.markdown(f"**{role}:** {msg}")

    elif role == "Bot":
        st.markdown(f"**{role}:** {msg}")

    elif role == "Sources":
        with st.expander("Sources"):
            for doc in msg:
                st.write(doc.page_content[:200] + "...")
