import re, os, json, pathlib
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
import gdown
import streamlit as st

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\u0000", " ")
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def split_csv_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[|,;\/]+", s)
    return [p.strip() for p in parts if p.strip()]

def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    return dict(os.environ)

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

ARTIFACT_URLS = {
    "X_dense.npy": "1lbpOK3iJxkp2M9eGf_DF-Lv_zLdh4JgI",
    "vectorizer.pkl.gz": "1vwfxAAA44bnDIMSKaD-inP9AC4mGtV0O",
    "faiss_metadata.json": "1ylb5npstJMtZbctsEW44XAwatVeARceI",
    "role_match_clf.pkl": "1yELjaRa65rAKRasXSoX47uzztavJ8BWe",
    "y_positions.npy": "11Is_ytPg_8d_vsuQsLdFjj-xphegDeHJ"
}

def download_from_gdrive(file_id: str, output_path: Path) -> bool:
    """Download a file from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()
    except Exception as e:
        st.error(f"Failed to download: {str(e)}")
        return False

def download_artifacts(art_dir: Path) -> bool:
    """Download required artifacts if they don't exist locally"""
    try:
        art_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, file_id in ARTIFACT_URLS.items():
            filepath = art_dir / filename
            if not filepath.exists():
                try:
                    with st.spinner(f"📥 Downloading {filename}..."):
                        if not download_from_gdrive(file_id, filepath):
                            st.error(f"Failed to download {filename}")
                            continue
                except Exception as e:
                    st.error(f"Error downloading {filename}: {str(e)}")
                    continue
                
        # Check if all required files exist
        missing_files = [f for f in ARTIFACT_URLS.keys() 
                        if not (art_dir / f).exists()]
        
        if missing_files:
            st.error(f"Missing required files: {', '.join(missing_files)}")
            return False
            
        return True
    except Exception as e:
        st.error(f"Failed to download artifacts: {str(e)}")
        return False
