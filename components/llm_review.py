# components/llm_review.py
import os, json
from typing import Dict, Any, List, Optional
from .utils import load_env
from .ats_scoring import ats_score

# -----------------------------
# Backend selection
# -----------------------------
def choose_backend():
    load_env()
    backend = os.getenv("MODEL_BACKEND", "groq").lower()   # ✅ default groq
    # ✅ Updated to supported Groq model
    name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")  
    return backend, name

def get_backend_info():
    """Get the current backend configuration"""
    try:
        # You can modify this based on your actual backend configuration
        return "openai", "GPT-3.5"  # or whatever backend you're using
    except Exception:
        return "dummy", "None"

# -----------------------------
# Build prompt
# -----------------------------
def build_prompt(resume_text: str, job_role: str, guidance_blobs: List[str], jd_text: str = "") -> str:
    guidance = "\n\n".join(guidance_blobs[:3])
    template = f"""You are an expert resume reviewer.
Target Role: {job_role}
Optional Job Description (JD):
{jd_text[:2000]}

Domain Guidance (from internal knowledge base):
{guidance}

TASKS:
1) Give section-wise feedback (Summary, Experience, Education, Skills, Projects, Certifications).
2) List *missing* skills/keywords for this role (high priority).
3) Rewrite 3-5 bullets to be quantifiable and tailored to the JD. Use STAR actions when possible.
4) Flag vague or redundant language and suggest concise alternatives.
5) Suggest formatting/clarity improvements.
6) Provide a brief 3-line profile summary for the candidate tailored to the role.

Resume Text:
{resume_text[:6000]}

Return JSON with keys: feedback_by_section, missing_keywords, bullet_rewrites,
language_fixes, formatting_suggestions, tailored_summary.
"""
    return template

# -----------------------------
# LLM call
# -----------------------------
def call_llm(prompt: str, system: Optional[str] = None) -> str:
    backend, model = choose_backend()

    if backend == "groq":
        try:
            from groq import Groq
            
            # Try multiple ways to get API key
            api_key = (
                os.getenv("GROQ_API_KEY") or 
                os.environ.get("GROQ_API_KEY") or 
                load_env().get("GROQ_API_KEY")
            )
            
            if not api_key:
                raise ValueError("""
                GROQ_API_KEY not found. Please set it using one of these methods:
                1. Export in terminal: export GROQ_API_KEY='your-key'
                2. Add to .env file: GROQ_API_KEY=your-key
                3. Set environment variable in your system
                """)

            client = Groq(api_key=api_key)
            
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})

            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=0.2,
                max_tokens=800,
            )
            return resp.choices[0].message.content.strip()
            
        except ImportError:
            raise RuntimeError("Please install groq: pip install groq")
        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}")

    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

# -----------------------------
# Resume review
# -----------------------------
def review_resume(
    resume_text: str, job_role: str, jd_text: str, guidance_blobs: List[str], required_skills: List[str]
) -> Dict[str, Any]:
    prompt = build_prompt(resume_text, job_role, guidance_blobs, jd_text)
    system = f"You are a meticulous ATS-savvy resume coach for {job_role}."
    llm_json = call_llm(prompt, system=system)

    score, detail = ats_score(resume_text + "\n" + jd_text, required_skills)

    return {
        "ats": {"score": score, "detail": detail},
        "llm_feedback_raw": llm_json,
    }