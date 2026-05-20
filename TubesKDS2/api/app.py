"""
Flask API — Cell Cycle Intelligence System
Dashboard endpoints use real training data. Inference is live.
"""

import os
import sys
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    PHASES, NUM_PHASES, IDX_TO_PHASE, PHASE_TO_IDX, IMG_SIZE, IMG_MEAN, IMG_STD,
    MODEL_DIR, DEVICE, PHASE_DURATIONS, TOTAL_CYCLE_HOURS,
    TIMELAPSE_NUM_FRAMES, TIMELAPSE_HOURS, NUM_EPOCHS,
    PHASE_COLORS, PHASE_COLOR_LIST
)
from src.model import CellCycleCNN
from src.gradcam import GradCAM, generate_gradcam_batch
from src.dataset import create_dataloaders
from src.ode_model import solve_cyclin_dynamics, compute_transition_matrix
from src.hmm import (
    BiologicalHMM, generate_ground_truth_sequence,
    simulate_timelapse_observations, inject_noise_into_sequence
)
from src.checkpoint_detector import run_checkpoint_analysis
from src.population_analysis import compute_population_summary

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, "frontend"))

# ── Global state ──────────────────────────────────────────
model = None
gradcam_engine = None
hmm_model = None
ode_results_cache = None
trans_matrix_cache = None
test_loader = None
train_loader = None
val_loader = None


def load_model():
    global model, gradcam_engine
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found at {model_path}. Run main.py first.")
        return False
    model = CellCycleCNN(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    gradcam_engine = GradCAM(model)
    print(f"  OK Model loaded")
    return True


def load_data():
    global train_loader, val_loader, test_loader
    try:
        train_loader, val_loader, test_loader = create_dataloaders()
        print(f"  OK Data loaded ({len(test_loader.dataset)} test samples)")
    except Exception as e:
        print(f"  WARNING: Could not load data: {e}")


def load_ode_hmm():
    global hmm_model, ode_results_cache, trans_matrix_cache
    ode_results_cache = solve_cyclin_dynamics()
    trans_matrix_cache, phase_fractions = compute_transition_matrix(ode_results_cache)
    hmm_model = BiologicalHMM(trans_matrix_cache, phase_fractions)
    print("  OK ODE + HMM ready")


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor="#080c14", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══ STATIC ROUTES ════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


# ═══ LIVE INFERENCE ═══════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run main.py first."}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image."}), 400

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    predicted_phase = IDX_TO_PHASE[predicted_idx]

    # Grad-CAM
    heatmap, _, _ = gradcam_engine.generate(img_tensor, target_class=predicted_idx)
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))
    fig.patch.set_facecolor("#080c14")
    original_np = np.array(img_resized)
    for ax in axes: ax.axis("off")
    axes[0].imshow(original_np); axes[0].set_title("Original", color="white", fontsize=9)
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Heatmap", color="white", fontsize=9)
    axes[2].imshow(original_np); axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay", color="white", fontsize=9)
    plt.tight_layout()
    gradcam_b64 = fig_to_base64(fig)

    return jsonify({
        "predicted_phase": predicted_phase,
        "confidence": float(probs[predicted_idx]),
        "probabilities": {p: float(probs[i]) for i, p in enumerate(PHASES)},
        "phase_info": {
            "duration_hours": PHASE_DURATIONS[predicted_phase],
            "fraction_of_cycle": PHASE_DURATIONS[predicted_phase] / TOTAL_CYCLE_HOURS,
            "category": "Mitotic" if predicted_phase in ["Prophase","Metaphase","Anaphase","Telophase"] else "Interphase",
        },
        "gradcam_image": gradcam_b64,
    })


# ═══ DASHBOARD: CNN EVALUATION (real test data) ═══════════

@app.route("/api/dashboard/cnn")
def dashboard_cnn():
    if model is None or test_loader is None:
        return jsonify({"error": "Model or data not loaded."}), 503

    from src.model import evaluate_test
    y_pred, y_true, y_probs = evaluate_test(model, test_loader)
    test_acc = float(np.mean(y_pred == y_true))

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_PHASES)))
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#0f1520")
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(NUM_PHASES)); ax.set_yticks(range(NUM_PHASES))
    ax.set_xticklabels(PHASES, rotation=45, ha="right", fontsize=7, color="white")
    ax.set_yticklabels(PHASES, fontsize=7, color="white")
    ax.set_xlabel("Predicted", color="white", fontsize=8)
    ax.set_ylabel("True", color="white", fontsize=8)
    ax.set_title("Confusion Matrix", color="white", fontsize=10)
    for i in range(NUM_PHASES):
        for j in range(NUM_PHASES):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "#aaa", fontsize=7)
    plt.tight_layout()
    cm_b64 = fig_to_base64(fig)

    # Training curves (from saved history if available, else placeholder)
    fig2, axes2 = plt.subplots(1, 2, figsize=(7, 3))
    fig2.patch.set_facecolor("#080c14")
    for ax in axes2: ax.set_facecolor("#0f1520")
    # We'll show per-class accuracy as a bar chart instead
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=PHASES, output_dict=True)
    per_class_acc = [report[p]["precision"] for p in PHASES]
    axes2[0].bar(PHASES, per_class_acc, color=PHASE_COLOR_LIST, edgecolor="none")
    axes2[0].set_ylim(0, 1); axes2[0].set_ylabel("Precision", color="white", fontsize=8)
    axes2[0].set_title("Per-Class Precision", color="white", fontsize=9)
    axes2[0].tick_params(colors="white"); axes2[0].set_xticklabels(PHASES, rotation=45, ha="right", fontsize=6)
    for s in axes2[0].spines.values(): s.set_color("#1a2540")

    per_class_recall = [report[p]["recall"] for p in PHASES]
    axes2[1].bar(PHASES, per_class_recall, color=PHASE_COLOR_LIST, edgecolor="none")
    axes2[1].set_ylim(0, 1); axes2[1].set_ylabel("Recall", color="white", fontsize=8)
    axes2[1].set_title("Per-Class Recall", color="white", fontsize=9)
    axes2[1].tick_params(colors="white"); axes2[1].set_xticklabels(PHASES, rotation=45, ha="right", fontsize=6)
    for s in axes2[1].spines.values(): s.set_color("#1a2540")
    plt.tight_layout()
    curves_b64 = fig_to_base64(fig2)

    total_params = sum(p.numel() for p in model.parameters())
    param_str = f"{total_params/1e6:.1f}M" if total_params > 1e6 else f"{total_params/1e3:.0f}K"

    return jsonify({
        "test_accuracy": test_acc,
        "total_samples": len(test_loader.dataset),
        "train_epochs": NUM_EPOCHS,
        "model_params": param_str,
        "confusion_matrix": cm_b64,
        "training_curves": curves_b64,
    })


# ═══ DASHBOARD: GRAD-CAM BATCH (real test data) ═══════════

@app.route("/api/dashboard/gradcam")
def dashboard_gradcam():
    if model is None or test_loader is None:
        return jsonify({"error": "Model or data not loaded."}), 503

    results = generate_gradcam_batch(model, test_loader.dataset, n_samples=7)

    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 5))
    fig.patch.set_facecolor("#080c14")

    for i, r in enumerate(results):
        axes[0, i].imshow(r["original"]); axes[0, i].axis("off")
        axes[0, i].set_title(f'True: {r["true_label"]}', fontsize=7, color="white")

        axes[1, i].imshow(r["heatmap"], cmap="jet"); axes[1, i].axis("off")

        axes[2, i].imshow(r["original"])
        axes[2, i].imshow(r["heatmap"], cmap="jet", alpha=0.5); axes[2, i].axis("off")
        pred_color = "#00E676" if r["predicted"] == r["true_label"] else "#FF1744"
        axes[2, i].set_title(f'{r["predicted"]} ({r["confidence"]:.0%})',
                             fontsize=7, color=pred_color)

    axes[0, 0].set_ylabel("Original", color="white", fontsize=8)
    axes[1, 0].set_ylabel("Heatmap", color="white", fontsize=8)
    axes[2, 0].set_ylabel("Overlay", color="white", fontsize=8)
    plt.tight_layout()
    img_b64 = fig_to_base64(fig)

    return jsonify({"gradcam_plot": img_b64})


# ═══ DASHBOARD: ODE ═══════════════════════════════════════

@app.route("/api/dashboard/ode")
def dashboard_ode():
    if ode_results_cache is None:
        return jsonify({"error": "ODE not computed."}), 503

    res = ode_results_cache
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#0f1520")
    ax.plot(res["t"], res["CycB"], color="#00E676", lw=1.5, label="Cyclin B")
    ax.plot(res["t"], res["CDK1"], color="#448AFF", lw=1.5, label="CDK1 (active)")
    ax.plot(res["t"], res["APC"], color="#E040FB", lw=1.5, label="APC/C (active)")
    ax.set_xlabel("Time (a.u.)", color="white", fontsize=8)
    ax.set_ylabel("Concentration", color="white", fontsize=8)
    ax.set_title("Tyson-Novak Cyclin-CDK ODE", color="white", fontsize=10)
    ax.legend(facecolor="#141c2e", edgecolor="#1a2540", labelcolor="white", fontsize=7)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#1a2540")
    ax.grid(True, alpha=0.15, color="#1a2540")
    plt.tight_layout()
    img_b64 = fig_to_base64(fig)

    return jsonify({
        "ode_plot": img_b64,
        "transition_matrix": trans_matrix_cache.tolist(),
        "phases": PHASES,
    })


# ═══ DASHBOARD: HMM (real phase durations) ════════════════

@app.route("/api/dashboard/hmm")
def dashboard_hmm():
    if hmm_model is None:
        return jsonify({"error": "HMM not ready."}), 503

    ground_truth = generate_ground_truth_sequence(
        n_frames=TIMELAPSE_NUM_FRAMES, total_hours=TIMELAPSE_HOURS)
    cnn_noisy = inject_noise_into_sequence(ground_truth, error_rate=0.18)
    noisy_obs = simulate_timelapse_observations(cnn_noisy, noise=0.15)
    hmm_path, _ = hmm_model.viterbi(noisy_obs)
    hmm_corrected = [IDX_TO_PHASE[idx] for idx in hmm_path]

    cnn_acc = sum(a == b for a, b in zip(ground_truth, cnn_noisy)) / len(ground_truth)
    hmm_acc = sum(a == b for a, b in zip(ground_truth, hmm_corrected)) / len(ground_truth)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.5), sharex=True)
    fig.patch.set_facecolor("#080c14")

    def plot_seq(ax, seq, title):
        ax.set_facecolor("#0f1520")
        for i, p in enumerate(seq):
            ax.bar(i, 1, bottom=PHASE_TO_IDX[p]-0.4, width=1.0,
                   color=PHASE_COLORS[p], linewidth=0)
        ax.set_yticks(range(NUM_PHASES))
        ax.set_yticklabels(PHASES, fontsize=6, color="white")
        ax.set_title(title, color="white", fontsize=8, loc="left")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#1a2540")

    plot_seq(axes[0], ground_truth, "Ground Truth")
    plot_seq(axes[1], cnn_noisy, f"CNN-Only ({cnn_acc:.0%})")
    plot_seq(axes[2], hmm_corrected, f"HMM-Corrected ({hmm_acc:.0%})")
    axes[2].set_xlabel("Frame", color="white", fontsize=8)
    plt.tight_layout()
    img_b64 = fig_to_base64(fig)

    return jsonify({
        "hmm_plot": img_b64,
        "cnn_accuracy": cnn_acc,
        "hmm_accuracy": hmm_acc,
        "improvement": hmm_acc - cnn_acc,
        "n_frames": TIMELAPSE_NUM_FRAMES,
        "total_hours": TIMELAPSE_HOURS,
    })


# ═══ DASHBOARD: CHECKPOINT ════════════════════════════════

@app.route("/api/dashboard/checkpoint")
def dashboard_checkpoint():
    if hmm_model is None:
        return jsonify({"error": "HMM not ready."}), 503

    ground_truth = generate_ground_truth_sequence(
        n_frames=TIMELAPSE_NUM_FRAMES, total_hours=TIMELAPSE_HOURS)
    cnn_noisy = inject_noise_into_sequence(ground_truth, error_rate=0.18)
    noisy_obs = simulate_timelapse_observations(cnn_noisy, noise=0.15)
    hmm_path, _ = hmm_model.viterbi(noisy_obs)
    hmm_corrected = [IDX_TO_PHASE[idx] for idx in hmm_path]

    hours_per_frame = TIMELAPSE_HOURS / TIMELAPSE_NUM_FRAMES
    _, anomalies_cnn = run_checkpoint_analysis(cnn_noisy, hours_per_frame=hours_per_frame)
    _, anomalies_hmm = run_checkpoint_analysis(hmm_corrected, hours_per_frame=hours_per_frame)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#0f1520")
    for i, p in enumerate(hmm_corrected):
        ax.bar(i, 1, color=PHASE_COLORS[p], width=1.0, linewidth=0)
    for a in anomalies_hmm:
        sc = {"low":"#FFD600","medium":"#FF9100","high":"#FF1744","critical":"#D500F9"}
        ax.axvspan(a.frame_start, a.frame_end, alpha=0.3, color=sc.get(a.severity,"#FF1744"))
    ax.set_xlabel("Frame", color="white", fontsize=8)
    ax.set_title("Checkpoint Anomaly Timeline", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#1a2540")
    plt.tight_layout()
    img_b64 = fig_to_base64(fig)

    def fmt(a):
        return {"checkpoint":a.checkpoint,"severity":a.severity,"frame_start":a.frame_start,
                "frame_end":a.frame_end,"phase":a.phase,"observed_value":float(a.observed_value),
                "expected_range":a.expected_range,"bio_interpretation":a.bio_interpretation}

    return jsonify({
        "checkpoint_plot": img_b64,
        "anomalies_cnn": [fmt(a) for a in anomalies_cnn],
        "anomalies_hmm": [fmt(a) for a in anomalies_hmm],
        "n_cnn_anomalies": len(anomalies_cnn),
        "n_hmm_anomalies": len(anomalies_hmm),
    })


# ═══ DASHBOARD: POPULATION (real test predictions) ════════

@app.route("/api/dashboard/population")
def dashboard_population():
    if model is None or test_loader is None:
        return jsonify({"error": "Model or data not loaded."}), 503

    from src.model import evaluate_test
    y_pred, _, _ = evaluate_test(model, test_loader)
    summary = compute_population_summary(y_pred)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor("#080c14")

    dist = summary["phase_distribution"]
    sizes = [dist[p] for p in PHASES]

    axes[0].set_facecolor("#080c14")
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=PHASES, colors=PHASE_COLOR_LIST,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "white", "fontsize": 7})
    for t in autotexts: t.set_fontsize(6); t.set_color("white")
    axes[0].set_title("Phase Distribution", color="white", fontsize=9)

    axes[1].set_facecolor("#0f1520")
    axes[1].bar(PHASES, sizes, color=PHASE_COLOR_LIST, edgecolor="none")
    axes[1].set_ylabel("Fraction", color="white", fontsize=8)
    axes[1].set_title("Phase Fractions", color="white", fontsize=9)
    axes[1].tick_params(colors="white")
    axes[1].set_xticklabels(PHASES, rotation=45, ha="right", fontsize=7)
    for s in axes[1].spines.values(): s.set_color("#1a2540")
    plt.tight_layout()
    img_b64 = fig_to_base64(fig)

    prolif = summary["proliferation"]
    return jsonify({
        "population_plot": img_b64,
        "total_cells": int(summary["total_cells_analyzed"]),
        "phase_distribution": summary["phase_distribution"],
        "mitotic_index": prolif["mitotic_index"],
        "growth_fraction": prolif["growth_fraction"],
        "doubling_time_hours": prolif["doubling_time_hours"],
        "status": prolif["status"],
        "clinical_significance": prolif["clinical_significance"],
    })


# ═══ STATUS ═══════════════════════════════════════════════

@app.route("/api/status")
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "ode_ready": ode_results_cache is not None,
        "hmm_ready": hmm_model is not None,
        "data_loaded": test_loader is not None,
        "device": str(DEVICE),
        "phases": PHASES,
    })


# ═══ MAIN ═════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Cell Cycle Intelligence — Web Server")
    print("=" * 50)

    load_model()
    load_data()
    load_ode_hmm()

    print(f"\n  http://localhost:5000")
    print(f"  Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
