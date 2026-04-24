# 🧠 AI-Based Career Path Recommendation System

A production-grade ML system that analyzes your skills, degree, and experience to recommend the best-fit tech career paths using a trained neural network.

**96% test accuracy · 97.2% cross-validation accuracy · 12 career paths · 3,000 training profiles**

---

## 🚀 Features

- **Neural Network Model** — MLPClassifier (256→128→64), trained with Adam optimizer & early stopping
- **Skill Extraction** — NLP-based keyword extraction from resume text
- **Skill Gap Analysis** — Identifies missing skills for any target career
- **Flask REST API** — `/predict`, `/extract-skills`, `/skill-gap`, `/model-info` endpoints
- **Streamlit Dashboard** — Interactive analytics & career explorer
- **Job Scraper** — BeautifulSoup + Scrapy pipeline for live job listings

---

## 📁 Project Structure

```
career-ai-recommendation-system/
├── data/
│   ├── generate_data.py        # Synthetic dataset generator (3,000 profiles)
│   └── career_profiles.csv     # Generated training data
├── model/
│   ├── train.py                # Neural network training pipeline
│   ├── career_model.pkl        # Trained sklearn pipeline (serialized)
│   └── model_meta.json         # Model accuracy & metadata
├── utils/
│   └── inference.py            # Inference engine (predict, skill_gap, extract_skills)
├── api/
│   └── app.py                  # Flask REST API
├── dashboard/
│   └── streamlit_app.py        # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/career-ai-recommendation-system.git
cd career-ai-recommendation-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset & train model
python data/generate_data.py
python model/train.py

# 5. Run Flask API
python api/app.py               # → http://localhost:5000

# 6. Run Streamlit Dashboard
streamlit run dashboard/streamlit_app.py
```

---

## 🔌 API Usage

**POST /predict**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["Python", "PyTorch", "NLP", "SQL"],
    "years_experience": 2,
    "degree": "B.Tech CS",
    "gpa": 8.5,
    "num_projects": 4,
    "num_certifications": 1,
    "has_internship": true
  }'
```

**POST /extract-skills**
```bash
curl -X POST http://localhost:5000/extract-skills \
  -H "Content-Type: application/json" \
  -d '{"text": "3 years Python, TensorFlow, NLP experience with Flask REST APIs"}'
```

---

## 🧠 Model Architecture

| Layer | Size | Activation |
|-------|------|------------|
| Input | 85 features | — |
| Hidden 1 | 256 neurons | ReLU |
| Hidden 2 | 128 neurons | ReLU |
| Hidden 3 | 64 neurons | ReLU |
| Output | 12 classes | Softmax |

**Training:** Adam optimizer · L2 regularisation (α=1e-4) · Batch size 64 · Early stopping

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **96.0%** |
| CV Accuracy (5-fold) | **97.23% ± 0.31%** |
| Weighted F1 | **0.9599** |
| Training samples | 2,400 |
| Test samples | 600 |

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `Flask` `Streamlit` `Pandas` `NumPy` `BeautifulSoup` `Scrapy`

---

## 📄 License

MIT License — free to use for academic and personal projects.
