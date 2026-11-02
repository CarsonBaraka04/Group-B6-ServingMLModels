
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

# Create Group B6 Classification Blueprint
group_b6_classification = Blueprint(
    "group_b6_classification", __name__
)

# Load the trained classification model
model_path = (
    Path(__file__).resolve().parent
    / "model"
    / "group_b6_fraud_classifier.joblib"
)

model = joblib.load(model_path)

# Eight input features used for prediction
required_features = [
    "agent_flagged_history",
    "witnesses_present",
    "claim_type",
    "policy_type",
    "police_report_filed",
    "claim_amount_to_policy_limit_ratio",
    "prior_claims_count",
    "policy_tenure_years"
]


@group_b6_classification.route(
    "/api/v1/models/group-b6-fraud-classifier/predictions",
    methods=["POST"]
)
def predict_fraud():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON object is required"
        }), 400

    # Check for missing input fields
    missing = [
        feature for feature in required_features
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    try:
        # Convert the request into a DataFrame
        claim = pd.DataFrame([{
            feature: data[feature]
            for feature in required_features
        }])

        # Generate prediction
        prediction = model.predict(claim)[0]

        # Generate class probabilities
        probabilities = model.predict_proba(claim)[0]

        class_probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                model.classes_, probabilities
            )
        }

        return jsonify({
            "model": "Group B6 Insurance Fraud Classifier",
            "predicted_class": str(prediction),
            "class_probabilities": class_probabilities
        }), 200

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": "Invalid prediction input",
            "details": str(error)
        }), 400
