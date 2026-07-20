import pickle
import numpy as np
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os
import requests

# Connect to Firestore using credentials from Render environment variable
firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
cred_dict = json.loads(firebase_creds_json)
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load sentiment pipeline
  # Hugging Face API configuration
HF_TOKEN = os.environ.get('HF_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def get_bert_sentiment_score(comment):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": comment[:512]}
        )

        result = response.json()[0]
        top = max(result, key=lambda x: x["score"])

        label = top["label"].lower()
        score = top["score"]

        if "positive" in label:
            return score
        elif "negative" in label:
            return -score
        else:
            return 0.0
    except Exception:
        return 0.0
        
# Load trained models
with open('recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

learner_enc = model_data['learner_enc']
teacher_enc = model_data['teacher_enc']
learner_factors = model_data['learner_factors']
teacher_factors = model_data['teacher_factors']
scoring_model = model_data['scoring_model']
df = model_data['df']

app = Flask(__name__)

def get_learner_preferred_style(learner_id):
    try:
        sessions = db.collection('sessions').where('learner_id', '==', learner_id).get()
        real_sessions = [s.to_dict() for s in sessions]
        if real_sessions:
            high_rated = [s for s in real_sessions if s.get('rating', 0) >= 4]
            source = high_rated if high_rated else real_sessions
            styles = [s.get('teaching_style', 'mixed') for s in source]
            return max(set(styles), key=styles.count)
    except:
        pass
    learner_sessions = df[df['learner_id'] == learner_id]
    if len(learner_sessions) == 0:
        return 'mixed'
    high_rated = learner_sessions[learner_sessions['rating'] >= 4]
    if len(high_rated) > 0:
        return high_rated['teaching_style'].mode()[0]
    return learner_sessions['teaching_style'].mode()[0]

def get_teacher_style_from_firestore(teacher_id):
    try:
        sessions = db.collection('sessions').where('teacher_id', '==', teacher_id).get()
        styles = [s.to_dict().get('teaching_style', 'mixed') for s in sessions]
        if styles:
            return max(set(styles), key=styles.count)
    except:
        pass
    return 'mixed'

def get_teacher_comments_from_firestore(teacher_id):
    try:
        sessions = db.collection('sessions').where('teacher_id', '==', teacher_id).get()
        comments = []
        for s in sessions:
            data = s.to_dict()
            comment = data.get('review_text', '')
            if comment:
                comments.append(comment)
        return comments
    except:
        return []

def get_svd_score(learner_id, teacher_id):
    if learner_id in learner_enc.classes_:
        learner_idx = learner_enc.transform([learner_id])[0]
        learner_vector = learner_factors[learner_idx]
    else:
        learner_vector = learner_factors.mean(axis=0)

    if teacher_id in teacher_enc.classes_:
        teacher_idx = teacher_enc.transform([teacher_id])[0]
        teacher_vector = teacher_factors[teacher_idx]
    else:
        teacher_vector = teacher_factors.mean(axis=0)

    return float(np.dot(learner_vector, teacher_vector))

def recommend_teachers(learner_id, skill, level, top_n=5):
    preferred_style = get_learner_preferred_style(learner_id)

    try:
        users = db.collection('users').get()
    except:
        return [], preferred_style

    results = []
    for user_doc in users:
        user_data = user_doc.to_dict()
        teacher_id = user_data.get('uid', user_doc.id)

        if teacher_id == learner_id:
            continue

        mentor_profile = user_data.get('mentorProfile', {})
        teacher_skills = mentor_profile.get('skills', [])

        matched = False
        for skill_obj in teacher_skills:
            skill_name = skill_obj.get('skillName', '')
            skill_level = skill_obj.get('level', '')
            if skill_name.lower() == skill.lower() and skill_level.lower() == level.lower():
                matched = True
                break

        if not matched:
            continue

        svd_score = get_svd_score(learner_id, teacher_id)

        real_comments = get_teacher_comments_from_firestore(teacher_id)
        if real_comments:
            bert_scores = [get_bert_sentiment_score(c) for c in real_comments[:10]]
            nlp_score = float(np.mean(bert_scores))
        else:
            teacher_sessions = df[df['teacher_id'] == teacher_id]
            if len(teacher_sessions) > 0:
                fake_comments = teacher_sessions['comment'].tolist()[:10]
                bert_scores = [get_bert_sentiment_score(c) for c in fake_comments]
                nlp_score = float(np.mean(bert_scores))
            else:
                nlp_score = 0.0

        teacher_style = get_teacher_style_from_firestore(teacher_id)
        if teacher_style == 'mixed':
            teacher_sessions = df[df['teacher_id'] == teacher_id]
            if len(teacher_sessions) > 0:
                teacher_style = teacher_sessions['teaching_style'].mode()[0]

        style_match = 1.0 if teacher_style == preferred_style else 0.0

        features = np.array([[svd_score, nlp_score, style_match]])
        final_score = float(scoring_model.predict(features)[0])

        basic_info = user_data.get('basicInfo', {})
        results.append({
            'teacher_id': teacher_id,
            'name': basic_info.get('fullName', 'Unknown'),
            'svd_score': round(svd_score, 3),
            'nlp_score': round(nlp_score, 3),
            'style_match': round(style_match, 2),
            'preferred_style': preferred_style,
            'final_score': round(final_score, 3)
        })

    results = sorted(results, key=lambda x: x['final_score'], reverse=True)[:top_n]
    return results, preferred_style

@app.route('/recommend', methods=['GET'])
def recommend():
    learner_id = request.args.get('learner_id')
    skill = request.args.get('skill')
    level = request.args.get('level')
    top_n = int(request.args.get('top_n', 5))

    if not learner_id or not skill or not level:
        return jsonify({"error": "missing parameters"}), 400

    results, preferred_style = recommend_teachers(learner_id, skill, level, top_n)
    return jsonify({
        "learner_id": learner_id,
        "skill": skill,
        "level": level,
        "preferred_style": preferred_style,
        "recommendations": results
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

@app.route('/analyze_comment', methods=['GET'])
def analyze_comment():
    comment = request.args.get('comment', '')
    if not comment:
        return jsonify({"error": "missing comment"}), 400
    score = get_bert_sentiment_score(comment)
    sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return jsonify({
        "comment": comment,
        "sentiment": sentiment,
        "score": round(score, 3)
    })
