"""
CareerAI — Flask REST API
Endpoints:
  POST /predict          → career path recommendations
  POST /extract-skills   → NLP skill extraction from text
  POST /skill-gap        → missing skills for a target career
  GET  /model-info       → model metadata & accuracy stats
  GET  /careers          → list of all supported career paths
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.inference import predict, extract_skills, skill_gap, get_model_meta

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "CareerAI API is running 🚀"})


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Body: {
      "skills": ["Python", "PyTorch"],
      "years_experience": 2,
      "degree": "B.Tech CS",
      "gpa": 8.0,
      "num_projects": 4,
      "num_certifications": 1,
      "has_internship": true,
      "open_source_contributions": false,
      "kaggle_rank_percentile": 0.0,
      "top_n": 5
    }
    """
    data = request.get_json(force=True)
    if not data or "skills" not in data:
        return jsonify({"error": "Request must include 'skills' list"}), 400
    top_n = int(data.get("top_n", 5))
    results = predict(data, top_n=top_n)
    return jsonify({"status": "success", "recommendations": results})


@app.route("/extract-skills", methods=["POST"])
def extract_skills_route():
    """
    Body: { "text": "I have 3 years of Python, TensorFlow and NLP experience..." }
    """
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Request must include 'text' field"}), 400
    skills = extract_skills(text)
    return jsonify({"status": "success", "skills": skills, "count": len(skills)})


@app.route("/skill-gap", methods=["POST"])
def skill_gap_route():
    """
    Body: { "skills": ["Python", "SQL"], "target_career": "Data Scientist" }
    """
    data = request.get_json(force=True)
    user_skills = data.get("skills", [])
    target = data.get("target_career", "")
    if not target:
        return jsonify({"error": "Request must include 'target_career'"}), 400
    gap = skill_gap(user_skills, target)
    return jsonify({"status": "success", "target_career": target, "missing_skills": gap})


@app.route("/model-info", methods=["GET"])
def model_info_route():
    meta = get_model_meta()
    return jsonify({"status": "success", "model": meta})


@app.route("/careers", methods=["GET"])
def careers_route():
    meta = get_model_meta()
    return jsonify({"status": "success", "career_paths": meta.get("career_paths", [])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
