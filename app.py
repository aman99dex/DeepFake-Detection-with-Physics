"""
PhysForensics - Web Dashboard
Showcases results, architecture, and provides a training guide.
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)

REAL_RESULTS_PATH = "outputs/real_data/logs/final_results.json"
REAL_TRAINING_LOG_PATH = "outputs/real_data/logs/training_log.json"
SYNTH_TRAINING_LOG_PATH = "outputs/logs/training_log.json"
V2_RESULTS_PATH = "outputs/v2_extended/logs/final_results.json"
V2_TRAINING_LOG_PATH = "outputs/v2_extended/logs/training_log.json"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def check_data_status():
    """Check what data and checkpoints are available."""
    status = {}
    # Datasets
    unified = Path("data/processed/unified")
    status["unified_dataset"] = unified.exists() and (unified / "metadata.json").exists()
    if status["unified_dataset"]:
        try:
            with open(unified / "metadata.json") as f:
                meta = json.load(f)
            status["unified_count"] = len(meta)
            status["unified_real"] = sum(1 for m in meta if m.get("label") == 0)
            status["unified_fake"] = sum(1 for m in meta if m.get("label") == 1)
        except Exception:
            status["unified_count"] = 0

    for ds in ["ff++", "celeb-df", "dfdc"]:
        d = Path(f"data/raw/{ds}")
        status[f"{ds}_available"] = d.exists() and (d / "real").exists()

    # Checkpoints
    status["v1_checkpoint"] = Path("outputs/real_data/checkpoints/best_model.pt").exists()
    status["v2_checkpoint"] = Path("outputs/v2_extended/checkpoints/best_model.pt").exists()

    return status


@app.route("/")
def index():
    real_results = load_json(REAL_RESULTS_PATH)
    real_training = load_json(REAL_TRAINING_LOG_PATH)
    synth_training = load_json(SYNTH_TRAINING_LOG_PATH)
    v2_results = load_json(V2_RESULTS_PATH)
    data_status = check_data_status()
    return render_template(
        "index.html",
        real_results=real_results,
        real_training=real_training,
        synth_training=synth_training,
        v2_results=v2_results,
        data_status=data_status,
    )


@app.route("/guide")
def guide():
    data_status = check_data_status()
    return render_template("guide.html", data_status=data_status)


@app.route("/api/results")
def api_results():
    return jsonify({
        "v1": load_json(REAL_RESULTS_PATH),
        "v2_extended": load_json(V2_RESULTS_PATH),
    })


@app.route("/api/training/real")
def api_training_real():
    return jsonify(load_json(REAL_TRAINING_LOG_PATH))


@app.route("/api/training/v2")
def api_training_v2():
    return jsonify(load_json(V2_TRAINING_LOG_PATH))


@app.route("/api/status")
def api_status():
    return jsonify(check_data_status())


@app.route("/images/real/<path:filename>")
def real_images(filename):
    return send_from_directory("outputs/real_data/visualizations", filename)


@app.route("/images/synth/<path:filename>")
def synth_images(filename):
    return send_from_directory("outputs/visualizations", filename)


@app.route("/images/v2/<path:filename>")
def v2_images(filename):
    return send_from_directory("outputs/v2_extended/visualizations", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
