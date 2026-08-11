"""
backend.py
Flask app: serves the frontend and exposes POST /analyze, which accepts an
uploaded blood work image and returns structured results from diet_pipeline.
"""

import os
import uuid
import logging

from flask import Flask, request, jsonify, render_template

from diet_pipeline import run_full_analysis, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB, matches Groq's limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file included in the request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        return jsonify({"error": f"Unsupported format '.{ext}'. Use: {sorted(SUPPORTED_FORMATS)}"}), 400

    # Unique filename so concurrent uploads never collide
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    try:
        result = run_full_analysis(saved_path)
        return jsonify(result)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Unexpected error during analysis")
        return jsonify({"error": "Something went wrong analyzing this report. Please try again."}), 500
    finally:
        # Clean up the uploaded file regardless of outcome
        try:
            os.remove(saved_path)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(debug=True, port=5000)
