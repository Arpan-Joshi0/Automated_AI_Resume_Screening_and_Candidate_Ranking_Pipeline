import streamlit as st
import os
import pandas as pd
from screener import index_resumes, screen_and_rank

st.set_page_config(page_title="AI Resume Ranking Engine", layout="wide")


st.markdown("""
    <style>
    .main-title { font-size:40px !important; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 20px;}
    .section-header { font-size:22px !important; font-weight: 600; color: #2563EB; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎯 AI Resume Screening & Rank Engine</div>', unsafe_allow_html=True)
st.caption("A production-ready RAG application for automated human resources matching without conversational interfaces.")

# Create the folder for storing PDFs dynamically if it's missing
UPLOAD_DIR = "uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="section-header">1. Upload Candidate Resumes Pool</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop candidate PDF files here:", accept_multiple_files=True, type=["pdf"])
    
    if st.button("🚀 Process & Parse Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Extracting text and running embedding vectors..."):
                # Save out raw file payloads
                for file in uploaded_files:
                    with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
                        f.write(file.getbuffer())
                
                # Execute indexing engine
                count = index_resumes(UPLOAD_DIR)
                st.success(f"System Ready: Indexed {count} candidate records into local vector space.")
        else:
            st.warning("Please upload at least one PDF resume first.")

with col2:
    st.markdown('<div class="section-header">2. Define Target Job Specification</div>', unsafe_allow_html=True)
    jd_input = st.text_area("Paste structural Job Description requirements here:", height=180, placeholder="Required: Python, Machine Learning, SQL, FastAPI, 2+ Years Experience...")
    top_k = st.slider("Select maximum candidate pull count (Top K):", min_value=1, max_value=10, value=3)
    
    if st.button("🔥 Run Analytics Match Engine", use_container_width=True):
        if not jd_input.strip():
            st.error("Operation Denied: Please provide a job description for semantic indexing.")
        else:
            with st.spinner("Comparing vector datasets and running Llama-3 evaluation..."):
                df_results = screen_and_rank(jd_input, top_n=top_k)
                
                if not df_results.empty:
                    st.toast("Analysis Completed Successfully!", icon="✅")
                    st.session_state['evaluation_results'] = df_results
                else:
                    st.error("No matches found. Ensure resumes are parsed and indexed correctly before matching.")

# Bottom analytics panel layout
if 'evaluation_results' in st.session_state:
    st.markdown('<div class="section-header">📊 Candidate Rank Leaderboard Matrix</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state['evaluation_results'], use_container_width=True, hide_index=True)
    
    # Enable immediate spreadsheet export download options for HR
    csv_payload = st.session_state['evaluation_results'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Structured Match Matrix (CSV File)",
        data=csv_payload,
        file_name="candidate_screening_matrix.csv",
        mime="text/csv",
        use_container_width=True
    )
