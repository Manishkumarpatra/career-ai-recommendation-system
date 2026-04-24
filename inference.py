"""
CareerAI Inference Engine
=========================
Loads trained model and exposes:
  - predict(profile_dict)  → top-N ranked career recommendations
  - extract_skills(text)   → NLP-based skill extraction from resume text
  - skill_gap(profile, target_career) → missing skills to reach target
"""

import pickle
import json
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "career_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

_artifacts = None
_meta      = None


def _load():
    global _artifacts, _meta
    if _artifacts is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            _artifacts = pickle.load(f)
        with open(META_PATH, "r") as f:
            _meta = json.load(f)
    return _artifacts, _meta


SKILL_ALIASES = {
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "tf": "TensorFlow", "react.js": "React",
    "reactjs": "React", "nodejs": "Node.js", "node": "Node.js",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "mongo": "MongoDB",
    "k8s": "Kubernetes", "kube": "Kubernetes", "nlp": "NLP",
    "cv": "Computer Vision", "rl": "Reinforcement Learning",
    "llm": "LLMs", "large language": "LLMs", "gpt": "LLMs",
    "bert": "NLP", "transformers": "NLP", "huggingface": "NLP",
    "pandas": "Pandas", "numpy": "NumPy", "matplotlib": "Matplotlib",
    "docker": "Docker", "aws": "AWS", "gcp": "GCP", "azure": "Azure",
    "fastapi": "FastAPI", "django": "Django", "flask": "Flask",
    "spring": "Spring Boot", "java spring": "Spring Boot",
    "tableau": "Tableau", "powerbi": "Power BI", "power bi": "Power BI",
    "airflow": "Airflow", "kafka": "Kafka", "spark": "Spark",
    "linux": "Linux", "bash": "Linux", "shell": "Linux",
    "javascript": "JavaScript", "js": "JavaScript", "typescript": "TypeScript",
    "ts": "TypeScript", "sql": "SQL", "mysql": "MySQL",
    "git": "Git", "github": "Git", "gitlab": "Git",
    "statistics": "Statistics", "stats": "Statistics",
    "machine learning": "scikit-learn", "deep learning": "PyTorch",
    "data structures": "Data Structures", "algorithms": "Algorithms",
    "system design": "System Design", "microservices": "Microservices",
}

SKILL_WEIGHTS_FOR_GAP = {
    "Machine Learning Engineer":   {"Python":9, "PyTorch":8, "scikit-learn":8, "NumPy":7, "MLOps":6},
    "Data Scientist":              {"Python":9, "SQL":9, "Statistics":9, "Pandas":8, "A/B Testing":7},
    "Data Engineer":               {"SQL":9, "Spark":9, "Airflow":8, "AWS":7, "Kafka":7},
    "Backend Engineer":            {"System Design":9, "SQL":8, "Docker":7, "REST API":9, "Go":6},
    "Frontend Engineer":           {"JavaScript":9, "React":9, "TypeScript":8, "CSS":7},
    "Full Stack Developer":        {"JavaScript":9, "React":8, "Node.js":8, "SQL":7},
    "DevOps / MLOps Engineer":     {"Docker":9, "Kubernetes":9, "Terraform":8, "CI/CD":9, "AWS":8},
    "NLP Engineer":                {"Python":9, "NLP":9, "PyTorch":8, "LLMs":8, "Pandas":7},
    "Computer Vision Engineer":    {"Python":9, "Computer Vision":9, "PyTorch":9, "Linear Algebra":8},
    "AI Research Scientist":       {"Python":9, "PyTorch":9, "Statistics":9, "Linear Algebra":9, "Probability":9},
    "Business Intelligence Analyst":{"SQL":9, "Tableau":9, "Excel":8, "Statistics":7},
    "Cloud Architect":             {"AWS":9, "Kubernetes":9, "Terraform":9, "System Design":9, "Docker":8},
}

SALARY_RANGES = {
    "Machine Learning Engineer":    {"entry": "₹8–14 LPA",  "mid": "₹14–28 LPA", "senior": "₹28–50 LPA"},
    "Data Scientist":               {"entry": "₹6–12 LPA",  "mid": "₹12–22 LPA", "senior": "₹22–40 LPA"},
    "Data Engineer":                {"entry": "₹7–13 LPA",  "mid": "₹13–24 LPA", "senior": "₹24–45 LPA"},
    "Backend Engineer":             {"entry": "₹6–12 LPA",  "mid": "₹12–22 LPA", "senior": "₹22–40 LPA"},
    "Frontend Engineer":            {"entry": "₹5–10 LPA",  "mid": "₹10–20 LPA", "senior": "₹20–35 LPA"},
    "Full Stack Developer":         {"entry": "₹6–12 LPA",  "mid": "₹12–22 LPA", "senior": "₹22–38 LPA"},
    "DevOps / MLOps Engineer":      {"entry": "₹8–14 LPA",  "mid": "₹14–26 LPA", "senior": "₹26–48 LPA"},
    "NLP Engineer":                 {"entry": "₹10–16 LPA", "mid": "₹16–30 LPA", "senior": "₹30–55 LPA"},
    "Computer Vision Engineer":     {"entry": "₹10–16 LPA", "mid": "₹16–30 LPA", "senior": "₹30–55 LPA"},
    "AI Research Scientist":        {"entry": "₹12–20 LPA", "mid": "₹20–40 LPA", "senior": "₹40–80 LPA"},
    "Business Intelligence Analyst":{"entry": "₹5–9 LPA",   "mid": "₹9–16 LPA",  "senior": "₹16–28 LPA"},
    "Cloud Architect":              {"entry": "₹10–18 LPA", "mid": "₹18–35 LPA", "senior": "₹35–70 LPA"},
}


def extract_skills(text: str) -> List[str]:
    """Extract known skills from free-form resume/bio text using keyword matching."""
    arts, _ = _load()
    skill_cols = arts["skill_cols"]
    text_lower = text.lower()
    found = set()

    for alias, canonical in SKILL_ALIASES.items():
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text_lower) and canonical in skill_cols:
            found.add(canonical)

    for skill in skill_cols:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(found)


def _build_feature_row(profile: Dict[str, Any]) -> pd.DataFrame:
    arts, _ = _load()
    skill_cols    = arts["skill_cols"]
    numeric_cols  = arts["numeric_cols"]
    cat_cols      = arts["categorical_cols"]

    row = {}
    skills_provided = profile.get("skills", [])
    for s in skill_cols:
        row[s] = 1 if s in skills_provided else 0

    row["years_experience"]        = int(profile.get("years_experience", 0))
    row["gpa"]                     = float(profile.get("gpa", 7.5))
    row["num_projects"]            = int(profile.get("num_projects", 2))
    row["num_certifications"]      = int(profile.get("num_certifications", 0))
    row["has_internship"]          = int(bool(profile.get("has_internship", False)))
    row["open_source_contributions"] = int(bool(profile.get("open_source_contributions", False)))
    row["kaggle_rank_percentile"]  = float(profile.get("kaggle_rank_percentile", 0.0))

    valid_degrees = [
        "BCA", "BSc CS", "B.Tech ECE", "B.Tech IT", "B.Tech CS",
        "MCA", "MBA Analytics", "MSc Data Science", "M.Tech CS", "PhD CS"
    ]
    degree = profile.get("degree", "B.Tech CS")
    row["degree"] = degree if degree in valid_degrees else "B.Tech CS"

    return pd.DataFrame([row])


def predict(profile: Dict[str, Any], top_n: int = 5) -> List[Dict]:
    """
    Predict career path recommendations.

    Args:
        profile: {
            "skills": ["Python", "PyTorch", ...],
            "years_experience": 2,
            "degree": "B.Tech CS",
            "gpa": 8.2,
            "num_projects": 4,
            "num_certifications": 1,
            "has_internship": True,
            "open_source_contributions": False,
            "kaggle_rank_percentile": 0.75
        }
        top_n: number of recommendations to return

    Returns:
        List of recommendation dicts sorted by confidence
    """
    arts, meta = _load()
    pipeline = arts["pipeline"]
    le       = arts["label_encoder"]

    X = _build_feature_row(profile)
    probs = pipeline.predict_proba(X)[0]

    top_indices = np.argsort(probs)[::-1][:top_n]
    exp = int(profile.get("years_experience", 0))
    level = "entry" if exp < 2 else ("mid" if exp < 6 else "senior")

    recommendations = []
    for idx in top_indices:
        career  = le.classes_[idx]
        conf    = round(float(probs[idx]) * 100, 1)
        gap     = skill_gap(profile.get("skills", []), career)
        salaries = SALARY_RANGES.get(career, {})
        recommendations.append({
            "rank":           len(recommendations) + 1,
            "career_path":    career,
            "confidence":     conf,
            "confidence_pct": f"{conf}%",
            "salary_range":   salaries.get(level, "N/A"),
            "skill_gap":      gap[:5],
            "skills_matched": _count_matched(profile.get("skills", []), career),
        })
    return recommendations


def skill_gap(user_skills: List[str], target_career: str) -> List[str]:
    """Return top missing skills for a target career, ranked by importance."""
    weights = SKILL_WEIGHTS_FOR_GAP.get(target_career, {})
    user_set = set(user_skills)
    missing = [(skill, w) for skill, w in weights.items() if skill not in user_set]
    missing.sort(key=lambda x: -x[1])
    return [s for s, _ in missing]


def _count_matched(user_skills: List[str], career: str) -> int:
    weights = SKILL_WEIGHTS_FOR_GAP.get(career, {})
    user_set = set(user_skills)
    return sum(1 for s in weights if s in user_set)


def get_model_meta() -> Dict:
    _, meta = _load()
    return meta


if __name__ == "__main__":
    sample_profile = {
        "skills": ["Python", "PyTorch", "NLP", "Pandas", "NumPy", "SQL", "Git", "LLMs"],
        "years_experience": 2,
        "degree": "B.Tech CS",
        "gpa": 8.5,
        "num_projects": 5,
        "num_certifications": 2,
        "has_internship": True,
        "open_source_contributions": True,
        "kaggle_rank_percentile": 0.80,
    }

    print("\n--- CareerAI Inference Test ---")
    results = predict(sample_profile, top_n=5)
    for r in results:
        print(f"#{r['rank']} {r['career_path']:<30} {r['confidence_pct']:<8} | Salary: {r['salary_range']}")
        print(f"   Skill gap : {', '.join(r['skill_gap']) or 'None'}")
