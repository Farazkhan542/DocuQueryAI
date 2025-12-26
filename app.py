"""
Streamlit app for the ACME RAG chatbot (Gemini-based).
Run locally with:
    streamlit run app.py
"""

import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


DEFAULT_DOC_PATH = "Company_sample.txt"


def resolve_api_key(manual_key: str) -> str:
    """Get API key from user input, env var, or fallback value."""
    if manual_key:
        return manual_key.strip()

    if "GOOGLE_API_KEY" in os.environ:
        return os.environ["GOOGLE_API_KEY"]

    # Fallback value
    return "AIzaSyAaRCWlyIQyR5YdahHyKhjFpRNNqX2wMmY"


@st.cache_resource(show_spinner=False)
def build_vectorstore(doc_path: str, api_key: str):
    """Load documents, chunk them, embed them, and return a Chroma vectorstore."""
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # Using in-memory Chroma (no FAISS needed)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="acme_rag"
    )

    return vectorstore


@st.cache_resource(show_spinner=False)
def build_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.5,
        convert_system_message_to_human=True,
    )


@st.cache_resource(show_spinner=False)
def prompt_template() -> PromptTemplate:
    """Prompt template for the RAG chain."""
    return PromptTemplate(
        input_variables={"context", "question"},
        template=(
            "You are a helpful assistant that answers questions based on "
            "the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer clearly and only using the context. "
            "If the context is insufficient, say 'Not enough information'."
        ),
    )


def run_rag(query: str, retriever, llm, template: PromptTemplate):
    """Execute retrieval + generation for a user query."""
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, docs


def main():
    st.set_page_config(page_title="ACME RAG Chatbot", page_icon="🤖")
    st.title("ACME RAG Chatbot (Gemini)")
    st.write(
        "Ask questions about the ACME company manual. "
        "The app retrieves relevant context and responds with Gemini."
    )

    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key.",
        )
        doc_path = st.text_input(
            "Document path",
            value=DEFAULT_DOC_PATH,
            help="Path to the source text file.",
        )
        show_context = st.checkbox("Show retrieved context", value=False)

    api_key = resolve_api_key(api_key_input)
    if not api_key:
        st.warning("Please enter a Google API key.")
        st.stop()

    if not os.path.exists(doc_path):
        st.error(f"Document not found: {doc_path}")
        st.stop()

    try:
        vectorstore = build_vectorstore(doc_path, api_key)
        retriever = vectorstore.as_retriever(search_type="similarity", k=3)
        llm = build_llm(api_key)
        template = prompt_template()
    except Exception as exc:
        st.error(f"Failed to initialize RAG components: {exc}")
        st.stop()

    user_question = st.text_input("Ask a question about the document")
    if user_question:
        with st.spinner("Thinking..."):
            answer, docs = run_rag(user_question, retriever, llm, template)
        st.subheader("Answer")
        st.write(answer)

        if show_context:
            st.subheader("Retrieved Context")
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
