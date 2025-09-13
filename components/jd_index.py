import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
from langdetect import detect, DetectorFactory
import json

# Fix randomness in langdetect
DetectorFactory.seed = 0

# ✅ Always point to app/artifacts
ROOT = Path(__file__).resolve().parent.parent
ART_PATH = ROOT / "artifacts"
ART_PATH.mkdir(parents=True, exist_ok=True)

ART_DIR = ART_PATH
print(f"[jd_index] Artifact directory available at: {ART_DIR}")

class JDIndex:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2)
        )
        self.role_match_clf = None
        self.index = None
        self.X_dense = None
        self.y_positions = None
        self.classes_ = None

    def _clean_text(self, x):
        if isinstance(x, str):
            return x.encode("utf-8", errors="ignore").decode("utf-8")
        else:
            return ""

    def _is_english(self, text):
        try:
            return detect(text) == "en"
        except:
            return False

    def build_from_csv(self, csv_path, sample_size=None, chunk_size=2000):
        print("📌 Reading CSV in chunks...")
        all_chunks = []
        for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="📂 Processing chunks"):
            for col in chunk.select_dtypes(include="object").columns:
                chunk[col] = chunk[col].map(self._clean_text)
            if "job_position" in chunk.columns and "relevant_skills" in chunk.columns:
                chunk = chunk[
                    chunk["job_position"].map(self._is_english) &
                    chunk["relevant_skills"].map(self._is_english)
                ]
                all_chunks.append(chunk)

        if not all_chunks:
            raise ValueError("No valid data found in CSV after cleaning & filtering English text.")

        df = pd.concat(all_chunks, ignore_index=True)
        if sample_size:
            df = df.sample(sample_size, random_state=42)

        print(f"✅ Total valid records: {len(df)}")

        X = self.vectorizer.fit_transform(df["relevant_skills"].fillna(""))
        y = df["job_position"].fillna("Unknown").astype(str).values

        clf = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None)
        classes = np.unique(y)

        print("📌 Training classifier with progress bar...")
        for _ in tqdm(range(5), desc="Training epochs"):
            clf.partial_fit(X, y, classes=classes)

        self.role_match_clf = clf
        self.classes_ = classes

        X_dense = X.astype(np.float32).toarray()
        self.index = faiss.IndexFlatL2(X_dense.shape[1])
        self.index.add(X_dense)

        self.X_dense = X_dense
        self.y_positions = y

        joblib.dump(self.vectorizer, ART_PATH / "vectorizer.pkl")
        joblib.dump(self.role_match_clf, ART_PATH / "role_match_clf.pkl")
        np.save(ART_PATH / "X_dense.npy", X_dense)
        np.save(ART_PATH / "y_positions.npy", y)

        # After training, save metadata
        metadata = {
            "total_records": len(df),
            "unique_roles": len(classes),
            "roles": classes.tolist()  # Convert numpy array to list for JSON
        }
        
        with open(ART_PATH / "model_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        print(f"✅ Artifacts saved in {ART_PATH}, total records: {len(df)}")
        print(f"Number of unique roles trained: {len(classes)}")
        print("Sample roles (first 5):", classes[:5])
        print("Total unique roles:", len(classes))

    def load(self):
        self.vectorizer = joblib.load(ART_PATH / "vectorizer.pkl")
        self.role_match_clf = joblib.load(ART_PATH / "role_match_clf.pkl")
        self.X_dense = np.load(ART_PATH / "X_dense.npy")
        self.y_positions = np.load(ART_PATH / "y_positions.npy", allow_pickle=True)

        # Add debug logging
        print("Loading classifier classes...")
        if hasattr(self.role_match_clf, "classes_"):
            self.classes_ = self.role_match_clf.classes_
            print(f"Classes from classifier: {len(self.classes_)}")
        else:
            self.classes_ = np.unique(self.y_positions)
            print(f"Classes from y_positions: {len(self.classes_)}")
        
        print("Sample of available roles:", self.classes_[:5])  # Show first 5 roles

        self.index = faiss.IndexFlatL2(self.X_dense.shape[1])
        self.index.add(self.X_dense)
        print(f"✅ Artifacts loaded successfully — {len(self.classes_)} roles available")

    def query(self, text, k=3):
        vec = self.vectorizer.transform([text]).astype(np.float32).toarray()
        D, I = self.index.search(vec, k)
        return [{"job_position": self.y_positions[idx], "score": score} for idx, score in zip(I[0], D[0])]

    def match_role(self, text):
        vec = self.vectorizer.transform([text])
        proba = self.role_match_clf.predict_proba(vec)[0]
        idx = np.argmax(proba)
        return self.role_match_clf.classes_[idx], proba[idx]
