"""
Career Path Recommendation Neural Network
==========================================
Architecture: Multi-Layer Perceptron (MLP) classifier
- Input  : skill vector + profile metadata (75 features)
- Hidden : [256 → ReLU → Dropout] → [128 → ReLU → Dropout] → [64 → ReLU]
- Output : 12 career path classes (softmax probabilities)

Training: Adam optimizer, early stopping, cross-validation
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

SKILL_COLS = [
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

NUMERIC_COLS = [
    "years_experience", "gpa", "num_projects",
    "num_certifications", "has_internship",
    "open_source_contributions", "kaggle_rank_percentile",
]

CATEGORICAL_COLS = ["degree"]

DEGREE_CATEGORIES = [
    ["BCA", "BSc CS", "B.Tech ECE", "B.Tech IT", "B.Tech CS",
     "MCA", "MBA Analytics", "MSc Data Science", "M.Tech CS", "PhD CS"]
]


def build_preprocessor():
    """Column transformer: ordinal-encode degree, scale numerics, pass skills through."""
    categorical_transformer = OrdinalEncoder(
        categories=DEGREE_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",  categorical_transformer, CATEGORICAL_COLS),
            ("num",  numeric_transformer,     NUMERIC_COLS),
            ("skill","passthrough",            SKILL_COLS),
        ]
    )
    return preprocessor


def build_model():
    """MLPClassifier mimicking a 3-hidden-layer neural network."""
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,                 # L2 regularisation (like weight decay)
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    return clf


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = SKILL_COLS + NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    X = df[feature_cols]
    y = df["career_path"]
    return X, y


def train(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "career_profiles.csv")

    print("=" * 55)
    print("  CareerAI Neural Network — Training Pipeline")
    print("=" * 55)

    print("\n[1/5] Loading dataset …")
    X, y = load_data(csv_path)
    print(f"      Samples : {len(X):,}  |  Features : {X.shape[1]}  |  Classes : {y.nunique()}")

    print("\n[2/5] Encoding labels …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"      Classes : {list(le.classes_)}")

    print("\n[3/5] Building pipeline …")
    preprocessor = build_preprocessor()
    clf = build_model()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])

    print("\n[4/5] Training with 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"      Fold scores : {[f'{s:.3f}' for s in cv_scores]}")

    print("\n[5/5] Final training on 80/20 split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'─'*55}")
    print(f"  Test Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Weighted F1    : {f1:.4f}")
    print(f"{'─'*55}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    artifacts = {
        "pipeline": pipeline,
        "label_encoder": le,
        "feature_cols": SKILL_COLS + NUMERIC_COLS + CATEGORICAL_COLS,
        "skill_cols": SKILL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }
    model_path = os.path.join(MODEL_DIR, "career_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    meta = {
        "model_type": "MLPClassifier (256→128→64)",
        "n_features": X.shape[1],
        "n_classes": y.nunique(),
        "career_paths": list(le.classes_),
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(acc), 4),
        "weighted_f1": round(float(f1), 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved → {model_path}")
    print(f"Metadata    → {meta_path}")
    return artifacts, meta


if __name__ == "__main__":
    train()
