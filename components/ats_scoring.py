from typing import Dict, List, Tuple
import re
import textstat
from collections import Counter

SECTION_HINTS = {
    "education": ["education", "qualifications", "academics", "b.tech", "m.tech", "bachelor", "master", "university", "college"],
    "experience": ["experience", "work", "employment", "professional", "internship"],
    "skills": ["skills", "technical skills", "tools", "technologies", "stack"],
    "projects": ["projects", "project work", "academic projects", "personal projects"],
    "certifications": ["certifications", "courses", "licenses"],
    "achievements": ["achievements", "awards", "honors"],
    "summary": ["summary", "profile", "objective", "about"],
}

def detect_sections(text: str) -> Dict[str, bool]:
    t = text.lower()
    presence = {}
    for sec, hints in SECTION_HINTS.items():
        presence[sec] = any(h in t for h in hints)
    return presence

def keyword_match_rate(text: str, skills: List[str]) -> float:
    t = text.lower()
    found = 0
    for s in set([x.lower() for x in skills]):
        if re.search(r"\b" + re.escape(s) + r"\b", t):
            found += 1
    return 0.0 if not skills else found / len(set([x.lower() for x in skills]))

def quantify_bullets_ratio(text: str) -> float:
    # crude heuristic: count numbers/percentages
    nums = len(re.findall(r"\b(\d+%?|\d+\.\d+%?)\b", text))
    bullets = len(re.findall(r"(^\s*[-•*])", text, flags=re.MULTILINE))
    sentences = max(1, len(re.findall(r"[\.!?]", text)))
    quantified = nums + bullets
    return min(1.0, quantified / (sentences * 0.6))

def formatting_checks(text: str) -> Dict[str, float]:
    # very rough heuristics
    issues = {}
    issues["all_caps_ratio"] = len(re.findall(r"\b[A-Z]{3,}\b", text)) / max(1, len(text.split()))
    issues["long_sentence_ratio"] = sum(1 for s in re.split(r"[\.!?]", text) if len(s.split())>35) / max(1, len(re.split(r"[\.!?]", text)))
    return issues

def readability_score(text: str) -> float:
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return 50.0

def ats_score(text: str, required_skills: List[str]) -> Tuple[float, Dict]:
    sections = detect_sections(text)
    coverage = sum(1 for v in sections.values() if v)/max(1, len(sections))
    keyword_rate = keyword_match_rate(text, required_skills)
    quantify = quantify_bullets_ratio(text)
    read = readability_score(text)

    fmt = formatting_checks(text)
    penalty = min(0.15, fmt.get("all_caps_ratio",0)*0.2 + fmt.get("long_sentence_ratio",0)*0.2)

    # Weighted sum (0-100)
    score = (
        0.35*keyword_rate +
        0.25*coverage +
        0.20*quantify +
        0.20*max(0.0, min(1.0, (read-30)/70))  # normalize 30..100 -> 0..1
    )
    score = max(0.0, min(1.0, score - penalty)) * 100

    detail = {
        "sections_detected": sections,
        "keyword_match_rate": round(keyword_rate,3),
        "coverage": round(coverage,3),
        "quantification": round(quantify,3),
        "readability": round(read,1),
        "formatting_penalty": round(penalty,3),
    }
    return score, detail
