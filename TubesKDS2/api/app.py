"""
Flask API for the Cell Cycle Intelligence System.
Serves the frontend and handles image classification requests.
"""

import os
import sys
import io
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    PHASES, IDX_TO_PHASE, PHASE_TO_IDX, IMG_SIZE, IMG_MEAN, IMG_STD,
    MODEL_DIR, DEVICE, PHASE_DURATIONS, TOTAL_CYCLE_HOURS
)
from src.model import CellCycleCNN

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, "frontend"))

# ── Model loading ─────────────────────────────────────────

model = None

def load_model():
    """Load the trained CNN model."""
    global model
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"  ⚠ Model not found at {model_path}")
        print(f"    Run main.py first to train the model.")
        return False

    model = CellCycleCNN(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  ✓ Model loaded from {model_path}")
    return True


# ── Image preprocessing ───────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Classify a cell image into a cell cycle phase."""
    if model is None:
        return jsonify({"error": "Model not loaded. Run main.py first to train."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    predicted_phase = IDX_TO_PHASE[predicted_idx]

    # Build response
    phase_probs = {phase: float(probs[i]) for i, phase in enumerate(PHASES)}

    return jsonify({
        "predicted_phase": predicted_phase,
        "confidence": float(probs[predicted_idx]),
        "probabilities": phase_probs,
        "phase_info": {
            "duration_hours": PHASE_DURATIONS[predicted_phase],
            "fraction_of_cycle": PHASE_DURATIONS[predicted_phase] / TOTAL_CYCLE_HOURS,
            "category": "Mitotic" if predicted_phase in ["Prophase", "Metaphase", "Anaphase", "Telophase"] else "Interphase",
        }
    })


@app.route("/api/status", methods=["GET"])
def status():
    """Check if the API and model are ready."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "phases": PHASES,
    })


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  Cell Cycle Intelligence — Web Interface")
    print("═" * 50)

    load_model()

    print(f"\n  Starting server at http://localhost:5000")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
