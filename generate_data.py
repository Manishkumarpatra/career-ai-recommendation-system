"""
Career Path Dataset Generator
Generates synthetic but realistic career profile data for model training.
"""

import numpy as np
import pandas as pd
import random
import json

random.seed(42)
np.random.seed(42)

SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "Go", "Rust", "SQL",
    "R", "MATLAB", "Scala", "Kotlin", "Swift", "PHP", "Ruby",
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "XGBoost", "LightGBM",
    "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Express", "GraphQL", "REST API",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "Elasticsearch",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "CI/CD",
    "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Databricks",
    "NLP", "Computer Vision", "Reinforcement Learning", "MLOps", "LLMs",
    "Statistics", "Linear Algebra", "Probability", "A/B Testing",
    "Tableau", "Power BI", "Looker", "Excel", "Git", "Linux",
    "Agile", "System Design", "Microservices", "Data Structures", "Algorithms",
]

CAREER_PATHS = [
    "Machine Learning Engineer",
    "Data Scientist",
    "Data Engineer",
    "Backend Engineer",
    "Frontend Engineer",
    "Full Stack Developer",
    "DevOps / MLOps Engineer",
    "NLP Engineer",
    "Computer Vision Engineer",
    "AI Research Scientist",
    "Business Intelligence Analyst",
    "Cloud Architect",
]

SKILL_WEIGHTS = {
    "Machine Learning Engineer": {
        "Python": 0.95, "TensorFlow": 0.80, "PyTorch": 0.82, "scikit-learn": 0.85,
        "SQL": 0.70, "NumPy": 0.88, "Pandas": 0.85, "MLOps": 0.65,
        "Docker": 0.60, "AWS": 0.58, "Statistics": 0.80, "Linear Algebra": 0.75,
        "Git": 0.75, "Flask": 0.55, "XGBoost": 0.70, "Spark": 0.45,
    },
    "Data Scientist": {
        "Python": 0.92, "R": 0.50, "SQL": 0.88, "Statistics": 0.92,
        "Pandas": 0.90, "NumPy": 0.85, "scikit-learn": 0.80, "Matplotlib": 0.80,
        "Seaborn": 0.70, "Tableau": 0.55, "A/B Testing": 0.75, "Probability": 0.85,
        "TensorFlow": 0.55, "Excel": 0.60, "Linear Algebra": 0.70,
    },
    "Data Engineer": {
        "Python": 0.88, "SQL": 0.95, "Spark": 0.85, "Kafka": 0.75,
        "Airflow": 0.78, "AWS": 0.78, "GCP": 0.60, "Docker": 0.72,
        "Hadoop": 0.65, "Scala": 0.55, "dbt": 0.65, "PostgreSQL": 0.80,
        "Databricks": 0.60, "Terraform": 0.50, "Git": 0.78,
    },
    "Backend Engineer": {
        "Python": 0.78, "Java": 0.65, "Go": 0.55, "SQL": 0.85,
        "PostgreSQL": 0.80, "Redis": 0.72, "Docker": 0.78, "Kubernetes": 0.65,
        "REST API": 0.92, "Microservices": 0.70, "System Design": 0.80,
        "Django": 0.60, "FastAPI": 0.65, "Spring Boot": 0.50, "Git": 0.88,
        "AWS": 0.70, "MongoDB": 0.60,
    },
    "Frontend Engineer": {
        "JavaScript": 0.95, "TypeScript": 0.82, "React": 0.85, "Angular": 0.55,
        "Vue": 0.50, "CSS": 0.88, "Git": 0.85, "REST API": 0.78,
        "GraphQL": 0.55, "Node.js": 0.65, "Testing": 0.68, "Agile": 0.70,
    },
    "Full Stack Developer": {
        "JavaScript": 0.92, "TypeScript": 0.75, "React": 0.82, "Node.js": 0.80,
        "SQL": 0.75, "MongoDB": 0.65, "Docker": 0.65, "REST API": 0.88,
        "Git": 0.88, "AWS": 0.58, "Python": 0.60, "CSS": 0.82,
    },
    "DevOps / MLOps Engineer": {
        "Docker": 0.95, "Kubernetes": 0.90, "AWS": 0.88, "GCP": 0.70,
        "Terraform": 0.82, "CI/CD": 0.92, "Linux": 0.90, "Git": 0.88,
        "Python": 0.72, "Kafka": 0.65, "Prometheus": 0.68, "MLOps": 0.75,
    },
    "NLP Engineer": {
        "Python": 0.95, "NLP": 0.98, "TensorFlow": 0.72, "PyTorch": 0.85,
        "scikit-learn": 0.78, "NumPy": 0.85, "Statistics": 0.80,
        "LLMs": 0.82, "Linear Algebra": 0.78, "Git": 0.75, "Flask": 0.60,
        "Docker": 0.58, "Pandas": 0.80,
    },
    "Computer Vision Engineer": {
        "Python": 0.95, "Computer Vision": 0.98, "PyTorch": 0.88, "TensorFlow": 0.78,
        "NumPy": 0.90, "Linear Algebra": 0.85, "C++": 0.55, "CUDA": 0.65,
        "scikit-learn": 0.70, "Git": 0.75, "Docker": 0.65,
    },
    "AI Research Scientist": {
        "Python": 0.95, "PyTorch": 0.92, "TensorFlow": 0.75, "Statistics": 0.95,
        "Linear Algebra": 0.95, "Probability": 0.92, "NLP": 0.65,
        "Computer Vision": 0.65, "Reinforcement Learning": 0.70, "LLMs": 0.78,
        "MATLAB": 0.50, "R": 0.55, "NumPy": 0.92,
    },
    "Business Intelligence Analyst": {
        "SQL": 0.95, "Excel": 0.88, "Tableau": 0.88, "Power BI": 0.75,
        "Python": 0.60, "Statistics": 0.78, "A/B Testing": 0.70,
        "Pandas": 0.65, "Looker": 0.62, "R": 0.45,
    },
    "Cloud Architect": {
        "AWS": 0.95, "GCP": 0.82, "Azure": 0.78, "Terraform": 0.88,
        "Docker": 0.85, "Kubernetes": 0.88, "System Design": 0.90,
        "Microservices": 0.82, "CI/CD": 0.78, "Linux": 0.82,
        "Python": 0.65, "Security": 0.72,
    },
}

DEGREES = ["B.Tech CS", "B.Tech IT", "M.Tech CS", "MCA", "BCA", "BSc CS",
           "MSc Data Science", "MBA Analytics", "B.Tech ECE", "PhD CS"]

DEGREE_SCORES = {
    "B.Tech CS": 0.85, "M.Tech CS": 0.95, "PhD CS": 1.0, "MSc Data Science": 0.88,
    "B.Tech IT": 0.80, "MCA": 0.75, "BCA": 0.65, "BSc CS": 0.70,
    "MBA Analytics": 0.72, "B.Tech ECE": 0.72,
}


def generate_profile(career_path):
    weights = SKILL_WEIGHTS.get(career_path, {})
    skill_vector = {}
    for skill in SKILLS:
        base_prob = weights.get(skill, 0.08)
        noise = random.gauss(0, 0.08)
        prob = max(0.0, min(1.0, base_prob + noise))
        skill_vector[skill] = 1 if random.random() < prob else 0

    years_exp = max(0, int(np.random.exponential(3.5)))
    degree = random.choice(DEGREES)
    gpa = round(random.uniform(6.0, 10.0), 1)
    num_projects = random.randint(1, 10)
    num_certifications = random.randint(0, 5)
    has_internship = 1 if random.random() < 0.6 else 0
    open_source_contributions = 1 if random.random() < 0.35 else 0
    kaggle_rank = random.randint(0, 5000) if "Data" in career_path or "ML" in career_path else 0

    profile = {
        **skill_vector,
        "years_experience": years_exp,
        "degree": degree,
        "gpa": gpa,
        "num_projects": num_projects,
        "num_certifications": num_certifications,
        "has_internship": has_internship,
        "open_source_contributions": open_source_contributions,
        "kaggle_rank_percentile": round(1 - kaggle_rank / 5000, 2) if kaggle_rank > 0 else 0.0,
        "career_path": career_path,
    }
    return profile


def generate_dataset(n_samples=3000):
    records = []
    per_class = n_samples // len(CAREER_PATHS)
    for career in CAREER_PATHS:
        for _ in range(per_class):
            records.append(generate_profile(career))
    random.shuffle(records)
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset(3000)
    df.to_csv("/home/claude/career_ai/data/career_profiles.csv", index=False)
    print(f"Dataset generated: {len(df)} profiles, {df.shape[1]} features")
    print(f"Career path distribution:\n{df['career_path'].value_counts()}")
