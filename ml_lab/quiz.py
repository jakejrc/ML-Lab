# -*- coding: utf-8 -*-
"""ML-Lab v3.9 - Interactive Quiz Generator"""

import random

def generate_quiz(model_name, task_type, eval_report, n_questions=3):
    pool = []

    pool.append({
        "question": "What describes overfitting?",
        "options": [
            "Low training score, low test score",
            "High training score, low test score",
            "Low training score, high test score",
            "High training score, high test score"
        ],
        "answer": 1,
        "explanation": "Overfitting: model performs well on training but poorly on test data."
    })
    pool.append({
        "question": "What is Cross-Validation's main purpose?",
        "options": [
            "Increase training data",
            "More reliable evaluation",
            "Speed up training",
            "Reduce features"
        ],
        "answer": 1,
        "explanation": "CV provides stable estimate of model performance on unseen data."
    })
    pool.append({
        "question": "When can Accuracy be misleading?",
        "options": [
            "Large dataset",
            "Class imbalance",
            "High dimensionality",
            "Complex model"
        ],
        "answer": 1,
        "explanation": "With 99% negative class, all-negative prediction gives 99% accuracy but is useless."
    })
    if task_type == "classification":
        pool.append({
            "question": "What does False Positive mean?",
            "options": [
                "Actual pos, predicted neg",
                "Actual neg, predicted pos",
                "Actual pos, predicted pos",
                "Actual neg, predicted neg"
            ],
            "answer": 1,
            "explanation": "False Positive: model incorrectly predicts negative as positive."
        })
        pool.append({
            "question": "Precision vs Recall relationship?",
            "options": [
                "Always change together",
                "Usually a trade-off",
                "No relationship",
                "Precision > Recall always"
            ],
            "answer": 1,
            "explanation": "Precision and Recall have inverse relationship. F1 is their harmonic mean."
        })
    elif task_type == "regression":
        pool.append({
            "question": "R2 range and meaning?",
            "options": [
                "[0,1], closer to 1 better",
                "[-1,1], closer to 1 better",
                "(-inf,1], closer to 1 better",
                "[0,inf), higher better"
            ],
            "answer": 2,
            "explanation": "R2 ranges (-inf,1]. R2=1 is perfect. R2<0 means worse than using mean."
        })
    elif task_type in ("unsupervised", "clustering"):
        pool.append({
            "question": "Silhouette Score meaning?",
            "options": [
                "[-1,1], near 1 = good clustering",
                "[0,1], near 1 = good",
                "[-1,0], near 0 = good",
                "(-inf,inf), higher = better"
            ],
            "answer": 0,
            "explanation": "Silhouette [-1,1]. Near 1: well-separated. Near -1: wrong cluster."
        })

    random.shuffle(pool)
    return pool[:min(n_questions, len(pool))]


def format_quiz_html(questions):
    if not questions:
        return '<div style="color:gray;padding:10px;">No questions available.</div>'

    html = ['<div style="font-family:sans-serif;">']
    for i, q in enumerate(questions):
        html.append('''
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin-bottom:12px;">
            <div style="font-weight:bold;color:#1e293b;margin-bottom:8px;">Q%d. %s</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        ''' % (i+1, q["question"]))
        for j, opt in enumerate(q["options"]):
            hl = "#e8f5e9" if j == q["answer"] else "#ffffff"
            tc = "#2e7d32" if j == q["answer"] else "#64748b"
            html.append('''
                <div style="background:%s;padding:6px 10px;border-radius:6px;font-size:13px;color:#334155;">
                    <span style="font-weight:bold;color:%s;">%s</span> %s
                </div>
            ''' % (hl, tc, chr(65+j) + ".", opt))
        html.append('''
            </div>
            <div style="margin-top:8px;padding:8px;background:#fff3e0;border-radius:6px;font-size:12px;color:#e65100;">
                <strong>Explanation:</strong> %s
            </div>
        </div>
        ''' % q["explanation"])
    html.append('</div>')
    return "\n".join(html)
