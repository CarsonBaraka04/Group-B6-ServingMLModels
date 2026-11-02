
from flask import Flask, jsonify
from flask_cors import CORS

from group_b6_regression_api import group_b6_regression
from group_b6_classification_api import group_b6_classification
from group_b6_clustering_api import group_b6_clustering
from group_b6_pharmacy_api import group_b6_pharmacy

# Create Group B6 Flask application
app = Flask(__name__)

# Enable cross-origin requests for the frontend
CORS(app)

# Register all four machine learning models
app.register_blueprint(group_b6_regression)
app.register_blueprint(group_b6_classification)
app.register_blueprint(group_b6_clustering)
app.register_blueprint(group_b6_pharmacy)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "project": "Group B6 ML Model Serving",
        "models_registered": 4
    }), 200


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
