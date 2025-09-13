import sys
from pathlib import Path
import os
import json
import streamlit as st
from fpdf import FPDF
import time
import numpy as np

from components.llm_review import get_backend_info

ROOT = Path(__file__).parent.resolve()

# ✅ Force components dir into path
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "components"))

# ✅ Force artifacts inside app/artifacts
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

def load_json_safe(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default

# Modify the load_roles function to be quieter
def load_roles():
    """Load roles from multiple possible sources"""
    roles = []
    try:
        metadata = load_json_safe(ART_DIR / "model_metadata.json")
        if metadata and "roles" in metadata:
            roles = sorted(metadata["roles"])
            return roles
    except Exception:
        pass
    
    try:
        y_path = ART_DIR / "y_positions.npy"
        if y_path.exists():
            y_positions = np.load(y_path, allow_pickle=True)
            roles = sorted(set(y_positions.tolist()))
            return roles
    except Exception:
        pass
    
    try:
        from components.jd_index import JDIndex
        jd = JDIndex()
        jd.load()
        if hasattr(jd, "classes_") and jd.classes_ is not None:
            roles = sorted(set(jd.classes_.tolist()))
            return roles
    except Exception:
        pass
    
    return roles

def create_pdf_report(ats_score, ats_details, llm_feedback, job_role):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Resume Analysis Report', 0, 1, 'C')
    pdf.line(10, 30, 200, 30)
    
    # Job Role
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Position: {job_role}', 0, 1)
    
    # ATS Score
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f'ATS Score: {ats_score}/100', 0, 1)
    
    # ATS Details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'ATS Analysis:', 0, 1)
    pdf.set_font('Arial', '', 10)
    for key, value in ats_details.items():
        pdf.multi_cell(0, 10, f'{key}: {value}')
    
    # LLM Feedback
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'LLM Feedback:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, llm_feedback)
    
    return pdf

# Update the main UI
st.set_page_config(page_title="🎯 Smart Resume Reviewer", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with animation
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #2E4053;'>🎯 Smart Resume Reviewer</h1>
        <p style='color: #566573;'>Powered by AI - Get instant feedback on your resume</p>
    </div>
    """, unsafe_allow_html=True)

# Add this line after ART_DIR initialization and before the UI code
roles = load_roles()  # Initialize roles list

# Main content in tabs
tab1, tab2 = st.tabs(["📝 Resume Analysis", "ℹ️ How it Works"])

with tab1:
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("### 📋 Job Details")
        available_roles = roles if roles else ["(Run training first)"]
        job_role = st.selectbox(
            "Select Target Position", 
            options=available_roles
        )
        jd_text = st.text_area("Job Description (optional)", height=160, 
                              placeholder="Enhance your analysis by pasting the job description...")

    with colB:
        st.markdown("### 📎 Resume Upload")
        up = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        resume_text = st.text_area("or Paste Resume Text", height=220,
                                 placeholder="Copy and paste your resume content here...")

if up is not None:
    tmp_path = os.path.join("/tmp", up.name)
    with open(tmp_path, "wb") as f:
        f.write(up.read())
    from components.resume_parser import extract_text_from_pdf
    text, pages = extract_text_from_pdf(tmp_path)
    if text:
        resume_text = text
        st.success(f"Extracted text from PDF ({pages} pages).")
    else:
        st.error("Could not extract text from PDF. Paste manually.")

st.divider()

# Add this function after the load_json_safe function
def load_faiss_metadata():
    """Load role metadata from faiss_meta.json"""
    meta_path = ART_DIR / "faiss_meta.json"
    return load_json_safe(meta_path, default=[])

# Add this line after the backend selection and before the colA, colB = st.columns(2)
meta = load_faiss_metadata()

# Replace the review button section with this:
if st.button("🚀 Analyze Resume", type="primary"):
    if not roles:
        st.error("⚠️ System not ready. Please contact support.")
    elif not (resume_text and resume_text.strip()):
        st.warning("📄 Please provide your resume content.")
    else:
        with st.spinner("🔄 Analyzing your resume..."):
            guidance_blobs, required_skills = [], []
            
            # Try to get guidance from faiss metadata
            meta_found = False
            if meta and isinstance(meta, (list, dict)):
                if isinstance(meta, dict):
                    meta = [meta]  # Convert single dict to list
                for m in meta:
                    if m.get("job_position") == job_role:
                        meta_found = True
                        guidance_blobs.append(m.get("text", ""))
                        if isinstance(m.get("skills"), list):
                            required_skills.extend(m.get("skills"))
            
            # Fallback to JDIndex if no metadata found
            if not meta_found:
                from components.jd_index import JDIndex
                jd = JDIndex()
                jd.load()
                guidance_blobs.append(f"Guidance for {job_role} based on classifier knowledge.")

            from components.llm_review import review_resume
            resp = review_resume(
                resume_text=resume_text,
                job_role=job_role,
                jd_text=jd_text,
                guidance_blobs=guidance_blobs,
                required_skills=required_skills,
            )

            ats = resp.get("ats", {})
            st.subheader("ATS Score")
            if ats:
                st.metric("Overall", f"{ats.get('score', 0.0):.1f}/100")
                st.json(ats.get("detail", {}), expanded=False)

            st.subheader("LLM Feedback (raw)")
            st.code(resp.get("llm_feedback_raw", "[No LLM output]"), language="json")
            
            # Display results in expanders
            with st.expander("📊 ATS Score Analysis", expanded=True):
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.metric("Match Score", f"{ats.get('score', 0.0):.1f}/100")
                st.json(ats.get("detail", {}))
            
            with st.expander("💡 AI Feedback", expanded=True):
                st.markdown(resp.get("llm_feedback_raw", "No feedback available"))
            
            # Generate and offer PDF download
            pdf = create_pdf_report(
                ats_score=ats.get('score', 0.0),
                ats_details=ats.get('detail', {}),
                llm_feedback=resp.get('llm_feedback_raw', ''),
                job_role=job_role
            )
            
            pdf_file = "resume_analysis.pdf"
            pdf.output(pdf_file)
            
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=f,
                    file_name="resume_analysis.pdf",
                    mime="application/pdf"
                )

with tab2:
    st.markdown("""
    ### How to Get the Best Results
    1. **Upload your resume** in PDF format or paste the text
    2. **Select the target position** from our comprehensive database
    3. **Paste the job description** (optional but recommended)
    4. Click **Analyze Resume** to get instant feedback
    
    ### What You Get
    - **ATS Compatibility Score**
    - **Detailed Skills Analysis**
    - **AI-Powered Recommendations**
    - **Downloadable PDF Report**
    """)

# Access secrets securely
api_key = st.secrets["OPENAI_API_KEY"]
model_backend = st.secrets.get("MODEL_BACKEND", "dummy")

# Get backend info
backend, model_name = get_backend_info()

# Remove debug button and clean up sidebar
st.sidebar.markdown("### System Status")
if backend and backend != "dummy":
    st.sidebar.success(f"🟢 Using {model_name}")
else:
    st.sidebar.warning("⚠️ Using Dummy Backend")
