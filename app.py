"""
PhysForensics - Web Dashboard
Showcases the PhysForensics deepfake detection project results.
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)

REAL_RESULTS_PATH = "outputs/real_data/logs/final_results.json"
REAL_TRAINING_LOG_PATH = "outputs/real_data/logs/training_log.json"
SYNTH_TRAINING_LOG_PATH = "outputs/logs/training_log.json"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@app.route("/")
def index():
    real_results = load_json(REAL_RESULTS_PATH)
    real_training = load_json(REAL_TRAINING_LOG_PATH)
    synth_training = load_json(SYNTH_TRAINING_LOG_PATH)
    return render_template(
        "index.html",
        real_results=real_results,
        real_training=real_training,
        synth_training=synth_training,
    )


@app.route("/api/results")
def api_results():
    return jsonify(load_json(REAL_RESULTS_PATH))


@app.route("/api/training/real")
def api_training_real():
    return jsonify(load_json(REAL_TRAINING_LOG_PATH))


@app.route("/api/training/synthetic")
def api_training_synthetic():
    return jsonify(load_json(SYNTH_TRAINING_LOG_PATH))


@app.route("/images/real/<path:filename>")
def real_images(filename):
    return send_from_directory("outputs/real_data/visualizations", filename)


@app.route("/images/synth/<path:filename>")
def synth_images(filename):
    return send_from_directory("outputs/visualizations", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
