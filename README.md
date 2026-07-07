# 🎯 AI Resume Screening & Ranking Engine

A production-ready Retrieval-Augmented Generation (RAG) system that extracts text from PDF resumes, builds a high-performance local vector index using NumPy matrix layouts, and generates structured candidate skill audits using Llama-3 over the Groq Cloud API.

---

## 🛠️ System Architecture & Technology Stack

This project is built as a lightweight, cost-effective RAG pipeline that operates without conversational overhead or expensive cloud vector database subscriptions.

*   **User Interface:** `Streamlit (v1.36.0)` for a clean, dual-column HR analytics dashboard.
*   **Document Parser:** `PyMuPDF (v1.24.1)` for rapid, high-fidelity local text extraction from PDF structures.
*   **Embeddings Engine:** `Sentence-Transformers (v3.0.1)` using the `all-MiniLM-L6-v2` model to map unstructured text into 384-dimensional dense vector spaces locally on CPU.
*   **Vector Database Layout:** A native pure-Python implementation powered by `NumPy` computing mathematical Cosine Similarity metrics.
*   **Inference Pipeline:** `Groq (v0.9.0)` Python SDK routing extracted context to the ultra-fast `llama3-8b-8192` model.

---

## 🚀 Installation & Local Deployment Guide

### 1. Clone the Project Repository
```bash
git clone <your-github-repository-url>
cd <repository-folder-name>
```

### 2. Configure Environment Keys
Create a `.env` file in the root project directory to securely map your API endpoints:
```text
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Install Pinned Dependencies
Ensure clean environment version matching by running:
```bash
pip install -r requirements.txt
```

### 4. Boot the Dashboard
Launch the Streamlit interface locally:
```bash
streamlit run app.py
```

---

## ⚙️ Operational Workflow

1.  **Pool Uploading:** Drop multiple candidate resumes (PDF format) into the interface. The storage directory is automatically flushed and refreshed per batch to keep evaluation runs distinct.
2.  **Vector Assembly:** Clicking **"Process & Parse Documents"** extracts text characters, calculates sentence structural matrices, and locks them into memory.
3.  **Analytics Matching:** Paste the target hiring guidelines into the specification panel and execute the pipeline. The matrix parses data instantly, flashes a completion toast, updates session states, and outputs a downloadable CSV leaderboard.
