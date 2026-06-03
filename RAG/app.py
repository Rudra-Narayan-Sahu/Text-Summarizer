import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Web RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot using Groq + Chroma")
st.write("Ask questions about the loaded webpage.")

# Create Vector DB once
if "vector" not in st.session_state:

    with st.spinner("Loading and indexing documents..."):

        # Load webpage
        loader = WebBaseLoader(
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )

        docs = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        texts = splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Store
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings
        )

        st.session_state.vector = vector_store

# LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY from the provided context.

<context>
{context}
</context>

Question: {input}
"""
)

# Create Chains
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

retriever = st.session_state.vector.as_retriever(
    search_kwargs={"k": 3}
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# User Input
query = st.text_input("Enter your question:")

if query:

    with st.spinner("Generating answer..."):

        response = retrieval_chain.invoke(
            {"input": query}
        )

    st.subheader("Answer")
    st.write(response["answer"])