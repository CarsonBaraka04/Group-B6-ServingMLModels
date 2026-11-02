
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

# Create a separate API route for Group B6
group_b6_regression = Blueprint(
    "group_b6_regression", __name__
)

# Load the trained model
model_path = (
    Path(__file__).resolve().parent
    / "model"
    / "group_b6_student_exam_regression.joblib"
)

artifact = joblib.load(model_path)

model = artifact["model"]
target_transformer = artifact["target_transformer"]
feature_columns = list(artifact["feature_columns"])


@group_b6_regression.route(
    "/api/v1/models/group-b6-student-score/predictions",
    methods=["POST"]
)
def predict_student_score():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON object is required"
        }), 400

    missing = [
        feature for feature in feature_columns
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    try:
        student = pd.DataFrame([{
            feature: data[feature]
            for feature in feature_columns
        }])

        transformed_prediction = model.predict(student)

        prediction = target_transformer.inverse_transform(
            transformed_prediction.reshape(-1, 1)
        )[0, 0]

        return jsonify({
            "model": "Group B6 Student Exam Score Regression",
            "predicted_exam_score": round(
                float(prediction), 2
            )
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": "Invalid prediction input",
            "details": str(error)
        }), 400
