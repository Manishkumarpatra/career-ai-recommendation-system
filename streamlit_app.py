"""
CareerAI — Streamlit Dashboard
Run with: streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.inference import predict, skill_gap, get_model_meta

st.set_page_config(page_title="CareerAI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
body { background: #07090F; color: #E8EEFF; }
.big-number { font-size: 2.5rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 CareerAI — Career Path Recommendation System")
st.caption("Powered by a Neural Network (MLP) trained on 3,000 profiles · 96% accuracy")

# ── SIDEBAR ─────────────────────────────────────────────────
st.sidebar.header("Your Profile")

ALL_SKILLS = [
    "Python","Java","JavaScript","TypeScript","C++","Go","SQL","R",
    "TensorFlow","PyTorch","scikit-learn","Keras","XGBoost","LightGBM",
    "Pandas","NumPy","SciPy","Matplotlib","Seaborn",
    "React","Angular","Vue","Node.js","Django","Flask","FastAPI",
    "Spring Boot","GraphQL","REST API","MySQL","PostgreSQL","MongoDB",
    "Redis","Cassandra","Elasticsearch","AWS","GCP","Azure",
    "Docker","Kubernetes","Terraform","CI/CD","Spark","Hadoop",
    "Kafka","Airflow","dbt","Databricks","NLP","Computer Vision",
    "Reinforcement Learning","MLOps","LLMs","Statistics","Linear Algebra",
    "Probability","A/B Testing","Tableau","Power BI","Excel",
    "Git","Linux","Agile","System Design","Microservices",
    "Data Structures","Algorithms"
]

selected_skills = st.sidebar.multiselect("Select Your Skills", ALL_SKILLS, default=["Python","SQL"])
degree = st.sidebar.selectbox("Degree", ["B.Tech CS","B.Tech IT","M.Tech CS","MCA","BCA","BSc CS","MSc Data Science","MBA Analytics","B.Tech ECE","PhD CS"])
years_exp = st.sidebar.slider("Years of Experience", 0, 15, 2)
gpa = st.sidebar.slider("GPA / CGPA", 0.0, 10.0, 7.5, 0.1)
num_projects = st.sidebar.number_input("Number of Projects", 0, 20, 3)
num_certs = st.sidebar.number_input("Certifications", 0, 10, 1)
has_internship = st.sidebar.checkbox("Internship Experience", value=True)
open_source = st.sidebar.checkbox("Open Source Contributions")
kaggle = st.sidebar.slider("Kaggle Rank Percentile", 0.0, 1.0, 0.0, 0.05)

run = st.sidebar.button("⚡ Get AI Recommendations", use_container_width=True)

# ── MODEL STATS ──────────────────────────────────────────────
meta = get_model_meta()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Accuracy", f"{meta['test_accuracy']*100:.1f}%")
col2.metric("CV Accuracy", f"{meta['cv_accuracy']*100:.1f}%")
col3.metric("Career Paths", meta['n_classes'])
col4.metric("Training Samples", f"{meta['training_samples']:,}")

st.divider()

# ── PREDICTION ───────────────────────────────────────────────
if run:
    if not selected_skills:
        st.warning("Please select at least one skill.")
    else:
        with st.spinner("Neural network running inference…"):
            profile = {
                "skills": selected_skills,
                "years_experience": years_exp,
                "degree": degree,
                "gpa": gpa,
                "num_projects": num_projects,
                "num_certifications": num_certs,
                "has_internship": has_internship,
                "open_source_contributions": open_source,
                "kaggle_rank_percentile": kaggle,
            }
            results = predict(profile, top_n=5)

        st.subheader("🎯 Your Career Recommendations")
        for r in results:
            with st.expander(f"#{r['rank']} {r['career_path']} — {r['confidence_pct']} confidence", expanded=(r['rank']==1)):
                c1, c2 = st.columns(2)
                c1.metric("Confidence", r['confidence_pct'])
                c1.metric("Salary Range", r['salary_range'])
                c2.write("**Skills to learn:**")
                if r['skill_gap']:
                    for s in r['skill_gap']:
                        c2.write(f"• {s}")
                else:
                    c2.success("You already have the core skills!")

        # Bar chart
        df = pd.DataFrame([{"Career Path": r['career_path'], "Confidence (%)": r['confidence']} for r in results])
        st.bar_chart(df.set_index("Career Path"))

        # Skill gap for top result
        top_career = results[0]['career_path']
        st.subheader(f"📋 Full Skill Gap for {top_career}")
        gap = skill_gap(selected_skills, top_career)
        if gap:
            st.write(", ".join([f"`{s}`" for s in gap]))
        else:
            st.success("No major skill gaps found!")
else:
    st.info("👈 Fill in your profile on the left and click **Get AI Recommendations**")
