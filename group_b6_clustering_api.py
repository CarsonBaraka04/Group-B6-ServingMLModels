
from pathlib import Path

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

# Group B6 Clustering Blueprint
group_b6_clustering = Blueprint(
    "group_b6_clustering", __name__
)

# Load the trained K-Means model
model_path = (
    Path(__file__).resolve().parent
    / "model"
    / "group_b6_bank_clustering.joblib"
)

artifact = joblib.load(model_path)

model = artifact["model"]
preprocessor = artifact["preprocessor"]
required_features = list(artifact["raw_columns"])


@group_b6_clustering.route(
    "/api/v1/models/group-b6-bank-clustering/predictions",
    methods=["POST"]
)
def predict_customer_cluster():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON object is required"
        }), 400

    # Validate required input features
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
        customer = pd.DataFrame([{
            feature: data[feature]
            for feature in required_features
        }])

        # Apply missing-value handling
        for col, median in artifact["numeric_medians"].items():
            customer[col] = pd.to_numeric(
                customer[col], errors="coerce"
            ).fillna(median)

        for col, mode in artifact["categorical_modes"].items():
            customer[col] = customer[col].fillna(mode)

        # Apply outlier caps
        for col, (lower, upper) in artifact["numeric_caps"].items():
            customer[col] = customer[col].clip(
                lower, upper
            )

        # Recreate engineered features
        customer["balance_to_tenure"] = (
            customer["account_balance_kes"] /
            (customer["years_as_customer"] + 1)
        )

        customer["activity_score"] = (
            customer["transactions_per_month"] *
            customer["avg_transaction_amount_kes"]
        )

        customer["tx_per_year"] = (
            customer["transactions_per_month"] * 12
        )

        customer["high_digital"] = (
            customer["digital_banking_usage_score"] >= 70
        ).astype(int)

        # Apply saved preprocessing
        customer = customer[artifact["feature_columns"]]

        processed = preprocessor.transform(customer)

        # Generate cluster prediction
        cluster = model.predict(processed)[0]

        return jsonify({
            "model": "Group B6 Bank Customer Clustering",
            "predicted_cluster": int(cluster),
            "number_of_clusters": int(model.n_clusters)
        }), 200

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": "Invalid prediction input",
            "details": str(error)
        }), 400
