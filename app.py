from flask import Flask, request, jsonify
from transformers import pipeline
import pickle
import numpy as np

app = Flask(__name__)

# Load everything from your saved pickle (SVD, encoders, factors, GB model, df)
with open('recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

svd_model = model_data['svd_model']
learner_enc = model_data['learner_enc']
teacher_enc = model_data['teacher_enc']
learner_factors = model_data['learner_factors']
teacher_factors = model_data['teacher_factors']
teacher_scores = model_data['teacher_scores']
scoring_model = model_data['scoring_model']
df = model_data['df']

# Load sentiment pipeline (downloads from HuggingFace, same as Colab)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def recommend_teachers(learner_id, skill, level, top_n=5):
    if learner_id in learner_enc.classes_:
        learner_idx = learner_enc.transform([learner_id])[0]
        learner_vector = learner_factors[learner_idx]
    else:
        learner_vector = learner_factors.mean(axis=0)

    learner_sessions = df[df['learner_id'] == learner_id]
    if len(learner_sessions) == 0:
        preferred_style = 'mixed'
    else:
        high_rated = learner_sessions[learner_sessions['rating'] >= 4]
        if len(high_rated) > 0:
            preferred_style = high_rated['teaching_style'].mode()[0]
        else:
            preferred_style = learner_sessions['teaching_style'].mode()[0]

    skill_teachers = df[
        (df['skill'].str.lower() == skill.lower()) &
        (df['level'].str.lower() == level.lower())
    ]['teacher_id'].unique()

    if len(skill_teachers) == 0:
        return [], preferred_style

    results = []
    for teacher_id in skill_teachers:
        if teacher_id not in teacher_enc.classes_:
            continue
        teacher_idx = teacher_enc.transform([teacher_id])[0]
        teacher_vector = teacher_factors[teacher_idx]
        svd_score = float(np.dot(learner_vector, teacher_vector))

        teacher_sessions = df[df['teacher_id'] == teacher_id]
        bert_score = float(teacher_sessions['bert_sentiment_score'].mean())
        style_match = 1.0 if teacher_sessions['teaching_style'].mode()[0] == preferred_style else 0.0

        features = np.array([[svd_score, bert_score, style_match]])
        final_score = float(scoring_model.predict(features)[0])

        results.append({
            'teacher_id': teacher_id,
            'svd_score': round(svd_score, 3),
            'bert_score': round(bert_score, 3),
            'style_match': round(style_match, 2),
            'preferred_style': preferred_style,
            'final_score': round(final_score, 3)
        })

    results = sorted(results, key=lambda x: x['final_score'], reverse=True)[:top_n]
    return results, preferred_style

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    learner_id = data.get('learner_id')
    skill = data.get('skill')
    level = data.get('level')

    results, style = recommend_teachers(learner_id, skill, level)
    return jsonify({'preferred_style': style, 'recommendations': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)