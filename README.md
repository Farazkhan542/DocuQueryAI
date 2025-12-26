🤖 ACME RAG Chatbot – Document Question Answering System

This project is an AI-powered Retrieval-Augmented Generation (RAG) chatbot that enables users to ask natural-language questions from a document and receive accurate, context-aware answers.

The system uses LangChain, Google Gemini, and ChromaDB to retrieve relevant document chunks and generate responses through a clean Streamlit interface.

🚀 Features

📄 Load and process text documents

✂️ Automatic text chunking for efficient retrieval

🧠 Semantic embeddings using Gemini Embedding Model

🔍 Similarity-based retrieval using ChromaDB

💬 Context-aware answers powered by Gemini LLM

🎨 Interactive web UI built with Streamlit

🛠️ Tech Stack

Python

Streamlit

LangChain

Google Gemini (LLM + Embeddings)

ChromaDB

FAISS (CPU)

TextLoader & RecursiveCharacterTextSplitter

📂 Project Structure
├── app.py                     # Main Streamlit application
├── Company_sample.txt         # Sample document for querying
├── RAG_Pipeline_Components.ipynb  # RAG pipeline experimentation notebook
├── requirements.txt           # Project dependencies
└── README.md

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py


Make sure to provide your Google API key when prompted or set it as an environment variable.

🎯 Use Case

Company manuals & policies

Knowledge-base chatbots

Document-based Q&A systems

RAG learning & experimentation

📌 Learning Outcomes

Hands-on experience with RAG architecture

Understanding of vector databases

Practical use of LLMs in real-world applications

Building end-to-end AI applications with Streamlit
