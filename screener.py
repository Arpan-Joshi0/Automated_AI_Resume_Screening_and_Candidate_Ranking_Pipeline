import os
import re
import json
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load local environment configuration keys
load_dotenv()

# 1. Initialize Local Free Embeddings Engine (Runs 100% Free on CPU)
# SentenceTransformer natively calculates local vector weights without DB wrappers
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to the Free Groq Inference Engine
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 3. Native Pure-Python Vector Store Engine using NumPy
class PurePythonVectorStore:
    def __init__(self):
        self.clear()

    def clear(self):
        """Flushes storage arrays out of local execution memory."""
        self.documents = []
        self.filenames = []
        self.embeddings = []

    def add(self, documents, filenames):
        """Generates vectors locally and caches them in a NumPy matrix layout."""
        self.documents = documents
        self.filenames = filenames
        # Calculate mathematical weights for text strings
        self.embeddings = embedding_model.encode(documents, convert_to_numpy=True)

    def query(self, query_text, n_results=3):
        """Calculates semantic match scores using standard Cosine Similarity."""
        if not self.documents:
            return {"ids": [], "documents": []}
            
        # Encode target query string into vector weights
        query_vector = embedding_model.encode([query_text], convert_to_numpy=True)[0]
        
        # Calculate dot products across standardized vector bounds
        scores = []
        for emb in self.embeddings:
            dot_product = np.dot(query_vector, emb)
            norm_q = np.linalg.norm(query_vector)
            norm_e = np.linalg.norm(emb)
            cosine_sim = dot_product / (norm_q * norm_e) if (norm_q * norm_e) > 0 else 0
            scores.append(cosine_sim)
            
        # Sort files from highest matching score to lowest
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:min(n_results, len(self.documents))]
        
        retrieved_ids = [self.filenames[idx] for idx in top_indices]
        retrieved_docs = [self.documents[idx] for idx in top_indices]
        
        return {"ids": retrieved_ids, "documents": retrieved_docs}

# Spin up global collection instance for the Streamlit UI to hook into
collection = PurePythonVectorStore()

def extract_text_from_pdf(pdf_path):
    """Opens a local PDF resume file and reads text data from pages."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def index_resumes(resume_folder):
    """Flushes previous sessions, reads all PDFs, and indexes them into the vector space."""
    collection.clear()
    documents = []
    filenames = []
    
    if not os.path.exists(resume_folder):
        os.makedirs(resume_folder)
        
    for filename in os.listdir(resume_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(resume_folder, filename)
            text = extract_text_from_pdf(path)
            
            if text.strip():  # Filter out broken or blank file assets
                documents.append(text)
                filenames.append(filename)
            
    if documents:
        collection.add(documents=documents, filenames=filenames)
        
    return len(documents)

def screen_and_rank(job_description, top_n=3):
    """RAG Core Pipeline: Queries local mathematical matrix indices and feeds results to Llama-3."""
    # Step A: Query Native Vector Array Store
    results = collection.query(job_description, n_results=top_n)
    
    ranked_candidates = []
    
    if not results or not results['ids'] or len(results['ids']) == 0:
        return pd.DataFrame() 

    # Step B: Iterate over retrieved vector document context
    for i in range(len(results['ids'])):
        filename = results['ids'][i]
        resume_text = results['documents'][i]
        
        # Step C: Prompt Engineering - Dictate rigorous constraints for JSON mapping
        prompt = f"""
        Analyze this Resume against the Job Description requirements.
        
        Job Description:
        {job_description}
        
        Candidate Resume Context:
        {resume_text[:2500]}
        
        Respond ONLY with a valid JSON block containing these exact keys. Do not include conversational text or wrap the JSON in code markdown syntax blocks.
        {{
            "matched_skills": ["list", "of", "skills"],
            "missing_skills": ["list", "of", "missing", "skills"],
            "verdict": "one concise sentence summarizing overall candidate fit"
        }}
        """
        
        # Route sanitized payload packets safely over to Groq free endpoints
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        raw_content = response.choices.message.content.strip()
        
        # Clean out common markdown layout code formatting blocks
        if raw_content.startswith("```"):
            raw_content = re.sub(r'^```[a-zA-Z]*\n', '', raw_content)
            raw_content = re.sub(r'\n```$', '', raw_content)
        
        try:
            analysis = json.loads(raw_content)
        except Exception:
            analysis = {
                "matched_skills": ["Review Manually"],
                "missing_skills": ["Review Manually"],
                "verdict": "Profile processed by Vector DB, check text contents manually."
            }
            
        ranked_candidates.append({
            "Rank": i + 1,
            "Candidate File": filename,
            "Matched Skills": ", ".join(analysis.get("matched_skills", [])),
            "Missing Critical Skills": ", ".join(analysis.get("missing_skills", [])),
            "Executive Fit Verdict": analysis.get("verdict", "N/A")
        })
        
    return pd.DataFrame(ranked_candidates)
