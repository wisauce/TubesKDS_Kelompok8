"""
Visualization Module — Publication-quality dark-themed plots
=============================================================
Creates all visual outputs for the Cell Cycle Intelligence System:
  1. Training curves
  2. Confusion matrix
  3. Grad-CAM overlays
  4. Cyclin-CDK ODE dynamics
  5. HMM temporal correction comparison
  6. Checkpoint anomaly timeline
  7. Population analysis dashboard
  8. Final summary dashboard
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.config import (
    PHASES, PHASE_COLORS, PHASE_COLOR_LIST, NUM_PHASES,
    IDX_TO_PHASE, PLOTS_DIR, GRADCAM_DIR, DASHBOARD_DIR
)


# ── Global Dark Theme ─────────────────────────────────────────

def setup_style():
    """Set up a premium dark theme for all plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0D1117",
        "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": "#E6EDF3",
        "axes.grid": True,
        "grid.color": "#21262D",
        "grid.alpha": 0.6,
        "text.color": "#E6EDF3",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
        "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D",
        "legend.fontsize": 9,
        "savefig.facecolor": "#0D1117",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })

setup_style()

# Custom colormaps
NEON_CMAP = LinearSegmentedColormap.from_list(
    "neon", ["#0D1117", "#00E5FF", "#00E676", "#FFEA00", "#FF1744"], N=256
)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "bio_heat", ["#0D1117", "#1A237E", "#E040FB", "#FF1744", "#FFEA00"], N=256
)


def _save(fig, name, subdir=PLOTS_DIR):
    """Save figure and close."""
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════
# 1. TRAINING CURVES
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(history):
    """Plot loss and accuracy curves with gradient fills."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training Progress", fontsize=16, fontweight="bold", y=1.02)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], color="#448AFF", linewidth=2, label="Train")
    ax1.plot(epochs, history["val_loss"], color="#FF1744", linewidth=2, label="Validation")
    ax1.fill_between(epochs, history["train_loss"], alpha=0.15, color="#448AFF")
    ax1.fill_between(epochs, history["val_loss"], alpha=0.15, color="#FF1744")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history["train_acc"], color="#00E676", linewidth=2, label="Train")
    ax2.plot(epochs, history["val_acc"], color="#E040FB", linewidth=2, label="Validation")
    ax2.fill_between(epochs, history["train_acc"], alpha=0.15, color="#00E676")
    ax2.fill_between(epochs, history["val_acc"], alpha=0.15, color="#E040FB")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy")
    ax2.legend()

    fig.tight_layout()
    return _save(fig, "01_training_curves.png")


# ═══════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred):
    """Neon-styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_PHASES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — CNN Phase Classifier", fontsize=16, y=1.02)

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap=NEON_CMAP, ax=ax1,
                xticklabels=PHASES, yticklabels=PHASES,
                linewidths=0.5, linecolor="#30363D",
                cbar_kws={"label": "Count"})
    ax1.set_xlabel("Predicted Phase")
    ax1.set_ylabel("True Phase")
    ax1.set_title("Raw Counts")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=HEATMAP_CMAP, ax=ax2,
                xticklabels=PHASES, yticklabels=PHASES,
                linewidths=0.5, linecolor="#30363D",
                vmin=0, vmax=1, cbar_kws={"label": "Fraction"})
    ax2.set_xlabel("Predicted Phase")
    ax2.set_ylabel("True Phase")
    ax2.set_title("Normalized")

    fig.tight_layout()

    # Print classification report
    report = classification_report(y_true, y_pred,
                                   target_names=PHASES, zero_division=0)
    print(f"\n  Classification Report:\n{report}")

    return _save(fig, "02_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════
# 3. GRAD-CAM VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

def plot_gradcam_results(gradcam_results):
    """Grid of Grad-CAM overlays showing what the CNN focuses on."""
    n = len(gradcam_results)
    if n == 0:
        return None

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Grad-CAM — What the CNN Sees", fontsize=16, y=1.02)

    if n == 1:
        axes = axes.reshape(2, 1)

    for i, res in enumerate(gradcam_results):
        # Original image
        axes[0, i].imshow(res["original"])
        axes[0, i].set_title(f"True: {res['true_label']}", fontsize=10,
                             color=PHASE_COLORS.get(res["true_label"], "#FFF"))
        axes[0, i].axis("off")

        # Grad-CAM overlay
        axes[1, i].imshow(res["original"])
        axes[1, i].imshow(res["heatmap"], cmap="jet", alpha=0.5)
        pred_color = "#00E676" if res["predicted"] == res["true_label"] else "#FF1744"
        axes[1, i].set_title(
            f"Pred: {res['predicted']} ({res['confidence']:.0%})",
            fontsize=10, color=pred_color
        )
        axes[1, i].axis("off")

    fig.tight_layout()
    return _save(fig, "03_gradcam_analysis.png", GRADCAM_DIR)


# ═══════════════════════════════════════════════════════════════
# 4. CYCLIN-CDK ODE DYNAMICS
# ═══════════════════════════════════════════════════════════════

def plot_ode_dynamics(ode_results):
    """Plot cyclin B / CDK1 / APC oscillations with phase annotations."""
    t = ode_results["t"]
    CycB = ode_results["CycB"]
    CDK1 = ode_results["CDK1"]
    APC = ode_results["APC"]
    phases = ode_results["phase_boundaries"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    fig.suptitle("Cyclin-CDK Oscillator — Biology Driving the Algorithm",
                 fontsize=16, y=1.02)

    # Concentration dynamics
    ax1.plot(t, CycB, color="#FF1744", linewidth=2.5, label="Cyclin B", zorder=3)
    ax1.plot(t, CDK1, color="#00E676", linewidth=2.5, label="CDK1 (active)", zorder=3)
    ax1.plot(t, APC, color="#448AFF", linewidth=2.5, label="APC/C (active)", zorder=3)

    ax1.fill_between(t, CycB, alpha=0.08, color="#FF1744")
    ax1.fill_between(t, CDK1, alpha=0.08, color="#00E676")
    ax1.fill_between(t, APC, alpha=0.08, color="#448AFF")

    ax1.set_ylabel("Concentration / Activity")
    ax1.set_title("Molecular Dynamics of Cell Cycle Regulators")
    ax1.legend(loc="upper right", framealpha=0.8)
    ax1.set_xlim(t[0], t[-1])

    # Phase strip
    phase_colors_arr = [PHASE_COLORS.get(p, "#333") for p in phases]
    for i in range(len(t) - 1):
        ax2.axvspan(t[i], t[i + 1], color=phase_colors_arr[i], alpha=0.8)

    ax2.set_xlabel("Time (arbitrary units)")
    ax2.set_ylabel("Phase")
    ax2.set_yticks([])
    ax2.set_title("Cell Cycle Phase (derived from molecular state)")
    ax2.set_xlim(t[0], t[-1])

    # Add phase legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=PHASE_COLORS[p], label=p) for p in PHASES]
    ax2.legend(handles=legend_patches, loc="upper center",
               ncol=len(PHASES), fontsize=8, framealpha=0.8)

    fig.tight_layout()
    return _save(fig, "04_ode_cyclin_dynamics.png")


# ═══════════════════════════════════════════════════════════════
# 5. HMM TEMPORAL CORRECTION
# ═══════════════════════════════════════════════════════════════

def plot_hmm_comparison(ground_truth, cnn_only, hmm_corrected, frames=None):
    """Compare CNN-only vs HMM-corrected phase sequences."""
    n = len(ground_truth)
    if frames is None:
        frames = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    fig.suptitle("HMM Temporal Correction — Biology Constrains the Prediction",
                 fontsize=16, y=1.02)

    titles = ["Ground Truth", "CNN-Only (noisy)", "CNN + ODE-HMM (corrected)"]
    sequences = [ground_truth, cnn_only, hmm_corrected]

    from src.config import PHASE_TO_IDX

    for ax, title, seq in zip(axes, titles, sequences):
        colors = [PHASE_COLORS.get(p, "#333") for p in seq]
        indices = [PHASE_TO_IDX.get(p, 0) for p in seq]

        for i in range(len(seq)):
            ax.bar(i, 1, color=colors[i], width=1.0, edgecolor="none")

        ax.set_ylabel(title, fontsize=10, fontweight="bold", rotation=0,
                      ha="right", va="center")
        ax.set_yticks([])
        ax.set_xlim(0, n)

    axes[-1].set_xlabel("Frame")

    # Accuracy comparison
    gt_indices = [PHASE_TO_IDX[p] for p in ground_truth]
    cnn_indices = [PHASE_TO_IDX[p] for p in cnn_only]
    hmm_indices = [PHASE_TO_IDX[p] for p in hmm_corrected]

    cnn_acc = sum(1 for a, b in zip(gt_indices, cnn_indices) if a == b) / n
    hmm_acc = sum(1 for a, b in zip(gt_indices, hmm_indices) if a == b) / n

    fig.text(0.5, -0.02,
             f"CNN-Only Accuracy: {cnn_acc:.1%}    │    "
             f"CNN + ODE-HMM Accuracy: {hmm_acc:.1%}    │    "
             f"Improvement: +{(hmm_acc - cnn_acc):.1%}",
             ha="center", fontsize=13, fontweight="bold",
             color="#00E676" if hmm_acc > cnn_acc else "#FF1744")

    # Phase legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=PHASE_COLORS[p], label=p) for p in PHASES]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=len(PHASES), fontsize=9, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return _save(fig, "05_hmm_correction.png")


# ═══════════════════════════════════════════════════════════════
# 6. CHECKPOINT ANOMALY TIMELINE
# ═══════════════════════════════════════════════════════════════

def plot_anomaly_timeline(phase_sequence, anomalies, hours_per_frame=0.2):
    """Timeline of detected checkpoint anomalies."""
    n = len(phase_sequence)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7), height_ratios=[2, 1],
                                   sharex=True)
    fig.suptitle("Layer 3 — Biological Checkpoint Anomaly Detection",
                 fontsize=16, y=1.02)

    # Phase timeline
    from src.config import PHASE_TO_IDX
    for i in range(n):
        color = PHASE_COLORS.get(phase_sequence[i], "#333")
        ax1.bar(i, 1, color=color, width=1.0, edgecolor="none")

    ax1.set_ylabel("Phase")
    ax1.set_yticks([])
    ax1.set_title("Cell Phase Sequence")

    # Anomaly markers
    severity_colors = {
        "low": "#FFEA00",
        "medium": "#FF9100",
        "high": "#FF1744",
        "critical": "#D50000",
    }
    severity_heights = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}

    if anomalies:
        for anom in anomalies:
            color = severity_colors.get(anom.severity, "#FFF")
            ax2.barh(0.5, anom.frame_end - anom.frame_start + 1,
                     left=anom.frame_start, height=0.6,
                     color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
            # Label
            mid = (anom.frame_start + anom.frame_end) / 2
            ax2.text(mid, 0.5, anom.checkpoint.split("(")[0].strip()[:15],
                     ha="center", va="center", fontsize=7,
                     color="#0D1117", fontweight="bold")

        # Severity legend
        from matplotlib.patches import Patch
        sev_patches = [Patch(facecolor=c, label=s.capitalize())
                       for s, c in severity_colors.items()]
        ax2.legend(handles=sev_patches, loc="upper right", fontsize=8)
    else:
        ax2.text(n / 2, 0.5, "✓ No anomalies detected — all checkpoints passed",
                 ha="center", va="center", fontsize=12, color="#00E676")

    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Anomalies")
    ax2.set_yticks([])
    ax2.set_xlim(0, n)
    ax2.set_title(f"Detected Anomalies: {len(anomalies)}")

    fig.tight_layout()
    return _save(fig, "06_checkpoint_anomalies.png")


# ═══════════════════════════════════════════════════════════════
# 7. POPULATION ANALYSIS DASHBOARD
# ═══════════════════════════════════════════════════════════════

def plot_population_dashboard(pop_summary):
    """Multi-panel population analysis dashboard."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Population Analysis Dashboard", fontsize=18, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Phase distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    dist = pop_summary["phase_distribution"]
    sizes = [dist[p] for p in PHASES]
    colors = PHASE_COLOR_LIST
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=PHASES, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "#0D1117", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("#0D1117")
        t.set_fontweight("bold")
    ax1.set_title("Phase Distribution")

    # 2. Phase distribution bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(PHASES, sizes, color=colors, edgecolor="#30363D", linewidth=0.5)
    ax2.set_ylabel("Fraction")
    ax2.set_title("Phase Fractions")
    ax2.set_xticklabels(PHASES, rotation=45, ha="right", fontsize=9)
    for bar, val in zip(bars, sizes):
        if val > 0.01:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=8, color="#E6EDF3")

    # 3. Interphase vs Mitotic
    ax3 = fig.add_subplot(gs[0, 2])
    inter = pop_summary["interphase_fraction"]
    mito = pop_summary["mitotic_fraction"]
    ax3.bar(["Interphase", "Mitotic"], [inter, mito],
            color=["#448AFF", "#FF1744"], edgecolor="#30363D")
    ax3.set_ylabel("Fraction")
    ax3.set_title("Interphase vs Mitotic")
    for i, val in enumerate([inter, mito]):
        ax3.text(i, val + 0.01, f"{val:.1%}", ha="center", fontsize=11,
                 fontweight="bold", color="#E6EDF3")

    # 4. Proliferation status panel (text)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    prolif = pop_summary["proliferation"]
    info_text = (
        f"Status: {prolif['status']}\n"
        f"─────────────────────────\n"
        f"Mitotic Index: {prolif['mitotic_index']:.2%}\n"
        f"Growth Fraction: {prolif['growth_fraction']:.2%}\n"
        f"Doubling Time: {prolif['doubling_time_hours']:.1f}h\n"
        f"─────────────────────────\n"
        f"{prolif['clinical_significance']}"
    )
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             color="#00E676",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#161B22",
                       edgecolor="#00E676", alpha=0.9))
    ax4.set_title("Proliferation Assessment")

    # 5. Expected vs Observed durations
    ax5 = fig.add_subplot(gs[1, 1:])
    est_dur = pop_summary["estimated_phase_durations"]
    from src.config import PHASE_DURATIONS
    x = np.arange(len(PHASES))
    w = 0.35
    expected = [PHASE_DURATIONS[p] for p in PHASES]
    observed = [est_dur.get(p, 0) for p in PHASES]
    ax5.bar(x - w / 2, expected, w, label="Expected (literature)", color="#448AFF",
            edgecolor="#30363D", alpha=0.8)
    ax5.bar(x + w / 2, observed, w, label="Estimated (from data)", color="#00E676",
            edgecolor="#30363D", alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(PHASES, rotation=45, ha="right")
    ax5.set_ylabel("Duration (hours)")
    ax5.set_title("Expected vs Estimated Phase Durations")
    ax5.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "07_population_dashboard.png", DASHBOARD_DIR)


# ═══════════════════════════════════════════════════════════════
# 8. TRANSITION MATRIX HEATMAP
# ═══════════════════════════════════════════════════════════════

def plot_transition_matrix(trans_matrix):
    """Visualize the ODE-derived transition probability matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("ODE-Derived Phase Transition Probabilities",
                 fontsize=14, y=1.02)

    sns.heatmap(trans_matrix, annot=True, fmt=".3f", cmap=NEON_CMAP,
                xticklabels=PHASES, yticklabels=PHASES,
                linewidths=1, linecolor="#30363D",
                ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "P(transition)"})
    ax.set_xlabel("To Phase")
    ax.set_ylabel("From Phase")
    ax.set_title("Biology-derived transition matrix used by HMM")

    fig.tight_layout()
    return _save(fig, "08_transition_matrix.png")


# ═══════════════════════════════════════════════════════════════
# 9. FINAL SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════

def plot_summary_dashboard(test_acc, cnn_seq_acc, hmm_seq_acc, n_anomalies, pop_summary):
    """Single summary figure for presentations."""
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Cell Cycle Intelligence System — Results Summary",
                 fontsize=18, fontweight="bold", y=1.05)

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    metrics = [
        ("CNN Test\nAccuracy", f"{test_acc:.1%}", "#448AFF"),
        ("CNN Sequence\nAccuracy", f"{cnn_seq_acc:.1%}", "#FF9100"),
        ("HMM-Corrected\nAccuracy", f"{hmm_seq_acc:.1%}", "#00E676"),
        ("Anomalies\nDetected", str(n_anomalies), "#FF1744"),
    ]

    for i, (label, value, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")

        # Glowing card
        card = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor="#161B22", edgecolor=color,
                              linewidth=3, transform=ax.transAxes)
        ax.add_patch(card)

        ax.text(0.5, 0.65, value, transform=ax.transAxes,
                fontsize=32, fontweight="bold", ha="center", va="center",
                color=color)
        ax.text(0.5, 0.25, label, transform=ax.transAxes,
                fontsize=11, ha="center", va="center", color="#8B949E")

    fig.tight_layout()
    return _save(fig, "09_summary.png", DASHBOARD_DIR)
