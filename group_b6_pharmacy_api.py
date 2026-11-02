
from pathlib import Path
import json

from flask import Blueprint, jsonify, request

# Group B6 Pharmacy Recommendation Blueprint
group_b6_pharmacy = Blueprint(
    "group_b6_pharmacy", __name__
)

# Load saved association rules
rules_path = (
    Path(__file__).resolve().parent
    / "model"
    / "group_b6_pharmacy_rules.json"
)

with open(rules_path, "r", encoding="utf-8") as file:
    artifact = json.load(file)

rules = artifact["rules"]


@group_b6_pharmacy.route(
    "/api/v1/models/group-b6-pharmacy-recommender/predictions",
    methods=["POST"]
)
def recommend_products():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON object is required"
        }), 400

    cart = data.get("cart")

    # Validate the shopping cart
    if (
        not isinstance(cart, list)
        or not cart
        or not all(
            isinstance(item, str) and item.strip()
            for item in cart
        )
    ):
        return jsonify({
            "error": "Cart must be a non-empty list of product names"
        }), 400

    # Normalize product names
    cart_items = {
        item.strip().casefold()
        for item in cart
    }

    recommendations = {}

    # Find matching association rules
    for rule in rules:

        antecedents = {
            item.casefold()
            for item in rule["antecedents"]
        }

        consequents = {
            item.casefold()
            for item in rule["consequents"]
        }

        # Cart must contain all antecedent products
        if not antecedents.issubset(cart_items):
            continue

        # Do not recommend products already in the cart
        if not consequents.isdisjoint(cart_items):
            continue

        key = tuple(sorted(consequents))

        recommendation = {
            "products": sorted(consequents),
            "based_on": sorted(antecedents),
            "confidence": round(float(rule["confidence"]), 4),
            "lift": round(float(rule["lift"]), 4),
            "support": round(float(rule["support"]), 4)
        }

        # Retain the highest-confidence rule per product set
        existing = recommendations.get(key)

        if (
            existing is None
            or recommendation["confidence"] > existing["confidence"]
        ):
            recommendations[key] = recommendation

    # Rank recommendations by confidence and lift
    ranked = sorted(
        recommendations.values(),
        key=lambda item: (
            item["confidence"],
            item["lift"]
        ),
        reverse=True
    )

    return jsonify({
        "model": "Group B6 Pharmacy Association Rules",
        "algorithm": "Apriori",
        "cart": sorted(cart_items),
        "recommendations": ranked[:5],
        "total_rules": len(rules)
    }), 200
