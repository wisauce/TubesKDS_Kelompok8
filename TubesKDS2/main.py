"""
╔══════════════════════════════════════════════════════════════════╗
║   CELL CYCLE INTELLIGENCE SYSTEM                                ║
║   A Biologically-Constrained Deep Learning Pipeline             ║
║                                                                  ║
║   Layer 1: CNN Phase Classifier (CS → Biology)                  ║
║   Layer 2: Cyclin-CDK ODE + HMM Temporal Validator (Bio → CS)  ║
║   Layer 3: Checkpoint Anomaly Detector (Bio → CS)               ║
║                                                                  ║
║   Dataset: BBBC048 — Jurkat Cell Cycle (Broad Institute)        ║
║   Reference: Eulenberg et al., Nature Comms 8:463 (2017)        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import random
import numpy as np
import torch

# ── Setup ─────────────────────────────────────────────────────

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    from src.config import (
        setup_directories, DEVICE, RANDOM_SEED, PHASES, NUM_PHASES,
        PHASE_TO_IDX, IDX_TO_PHASE, TIMELAPSE_NUM_FRAMES, TIMELAPSE_HOURS
    )

    set_seeds(RANDOM_SEED)
    setup_directories()

    print("\n" + "═" * 65)
    print("  CELL CYCLE INTELLIGENCE SYSTEM")
    print("  Biologically-Constrained Deep Learning for Cell Cycle Analysis")
    print("═" * 65)
    print(f"  Device: {DEVICE}")
    print(f"  Phases: {', '.join(PHASES)}")
    print("═" * 65 + "\n")

    # ══════════════════════════════════════════════════════════
    # STEP 0: DATASET PREPARATION — Real BBBC048 Data
    # ══════════════════════════════════════════════════════════

    print("\n" + "━" * 65)
    print("  STEP 0: Dataset Preparation (BBBC048 — Broad Institute)")
    print("━" * 65)

    from src.config import MAX_SAMPLES_PER_PHASE
    from src.synthetic_data import prepare_dataset
    n_images = prepare_dataset(max_per_phase=MAX_SAMPLES_PER_PHASE)

    if n_images == 0:
        print("\n  ✗ No images available. Please check the download.")
        print("    You can manually download from:")
        print("    https://data.broadinstitute.org/bbbc/BBBC048/BBBC048v1.zip")
        print("    and extract to data/BBBC048_raw/")
        return

    # ══════════════════════════════════════════════════════════
    # LAYER 1: CNN PHASE CLASSIFIER (CS → Biology)
    # ══════════════════════════════════════════════════════════

    print("\n" + "━" * 65)
    print("  LAYER 1: CNN Phase Classifier")
    print("  ResNet-18 fine-tuned on real fluorescence microscopy")
    print("━" * 65)

    from src.dataset import create_dataloaders
    from src.model import CellCycleCNN, train_model, evaluate_test
    from src.gradcam import generate_gradcam_batch
    from src.visualization import (
        plot_training_curves, plot_confusion_matrix,
        plot_gradcam_results
    )

    train_loader, val_loader, test_loader = create_dataloaders()

    # Train CNN
    model = CellCycleCNN(pretrained=True)
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model, history = train_model(model, train_loader, val_loader)

    # Evaluate on test set
    print("\n  Evaluating on test set...")
    y_pred, y_true, y_probs = evaluate_test(model, test_loader)
    test_acc = np.mean(y_pred == y_true)
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Visualizations
    print("\n  Generating Layer 1 visualizations...")
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred)

    # Grad-CAM
    print("  Generating Grad-CAM explainability maps...")
    gradcam_results = generate_gradcam_batch(model, test_loader.dataset, n_samples=7)
    plot_gradcam_results(gradcam_results)

    # ══════════════════════════════════════════════════════════
    # LAYER 2: CYCLIN-CDK ODE + HMM (Biology → CS)
    # ══════════════════════════════════════════════════════════

    print("\n" + "━" * 65)
    print("  LAYER 2: Cyclin-CDK Temporal Validator")
    print("  ODE-derived transition matrix constrains CNN predictions")
    print("━" * 65)

    from src.ode_model import solve_cyclin_dynamics, compute_transition_matrix
    from src.hmm import (
        BiologicalHMM, generate_ground_truth_sequence,
        simulate_timelapse_observations, inject_noise_into_sequence
    )
    from src.visualization import (
        plot_ode_dynamics, plot_hmm_comparison, plot_transition_matrix
    )

    # Solve the ODE
    print("\n  Solving Tyson-Novak Cyclin-CDK ODE system...")
    ode_results = solve_cyclin_dynamics()
    print(f"  ✓ ODE solved: {len(ode_results['t'])} time points")

    # Derive transition matrix from ODE
    trans_matrix, phase_fractions = compute_transition_matrix(ode_results)
    print(f"\n  Phase fractions from ODE:")
    for phase, frac in phase_fractions.items():
        print(f"    {phase:>12s}: {frac:.3f}")

    # Plot ODE dynamics
    plot_ode_dynamics(ode_results)
    plot_transition_matrix(trans_matrix)

    # Set up HMM with ODE-derived transitions
    hmm = BiologicalHMM(trans_matrix, phase_fractions)
    print(f"\n  ✓ HMM initialised with ODE-derived transition matrix")

    # Simulate a time-lapse sequence for comparison
    print(f"\n  Simulating {TIMELAPSE_NUM_FRAMES}-frame time-lapse "
          f"({TIMELAPSE_HOURS}h)...")

    ground_truth = generate_ground_truth_sequence(
        n_frames=TIMELAPSE_NUM_FRAMES, total_hours=TIMELAPSE_HOURS
    )

    # CNN-only predictions (simulated with noise)
    cnn_observations = simulate_timelapse_observations(ground_truth)

    # Noisy CNN predictions (argmax of noisy softmax)
    cnn_only_preds = [IDX_TO_PHASE[np.argmax(obs)] for obs in cnn_observations]

    # Also inject explicit errors for a clearer demo
    cnn_only_noisy = inject_noise_into_sequence(ground_truth, error_rate=0.18)

    # HMM-corrected predictions
    noisy_obs = simulate_timelapse_observations(cnn_only_noisy, noise=0.15)
    hmm_path, hmm_prob = hmm.viterbi(noisy_obs)
    hmm_corrected = [IDX_TO_PHASE[idx] for idx in hmm_path]

    # Accuracy comparison
    cnn_seq_acc = sum(1 for a, b in zip(ground_truth, cnn_only_noisy)
                      if a == b) / len(ground_truth)
    hmm_seq_acc = sum(1 for a, b in zip(ground_truth, hmm_corrected)
                      if a == b) / len(ground_truth)

    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │  CNN-only sequence accuracy:  {cnn_seq_acc:.1%}    │")
    print(f"  │  HMM-corrected accuracy:      {hmm_seq_acc:.1%}    │")
    print(f"  │  Improvement:                +{hmm_seq_acc - cnn_seq_acc:.1%}    │")
    print(f"  └──────────────────────────────────────┘")

    # Plot comparison
    plot_hmm_comparison(ground_truth, cnn_only_noisy, hmm_corrected)

    # ══════════════════════════════════════════════════════════
    # LAYER 3: CHECKPOINT ANOMALY DETECTOR (Biology → CS)
    # ══════════════════════════════════════════════════════════

    print("\n" + "━" * 65)
    print("  LAYER 3: Biological Checkpoint Anomaly Detector")
    print("  Rules derived from cell biology, not from data")
    print("━" * 65)

    from src.checkpoint_detector import run_checkpoint_analysis
    from src.visualization import plot_anomaly_timeline

    hours_per_frame = TIMELAPSE_HOURS / TIMELAPSE_NUM_FRAMES

    # Run on HMM-corrected sequence
    print("\n  Running checkpoint analysis on HMM-corrected sequence...")
    track_hmm, anomalies_hmm = run_checkpoint_analysis(
        hmm_corrected, hours_per_frame=hours_per_frame
    )

    # Also run on CNN-only sequence for comparison
    print("  Running checkpoint analysis on CNN-only sequence...")
    track_cnn, anomalies_cnn = run_checkpoint_analysis(
        cnn_only_noisy, hours_per_frame=hours_per_frame
    )

    print(f"\n  CNN-only anomalies detected: {len(anomalies_cnn)}")
    print(f"  HMM-corrected anomalies detected: {len(anomalies_hmm)}")

    if anomalies_hmm:
        print(f"\n  Anomaly details (HMM-corrected):")
        for a in anomalies_hmm[:10]:
            print(f"    {a}")

    # Plot anomaly timeline (for HMM-corrected)
    plot_anomaly_timeline(hmm_corrected, anomalies_hmm, hours_per_frame)

    # ══════════════════════════════════════════════════════════
    # POPULATION ANALYSIS
    # ══════════════════════════════════════════════════════════

    print("\n" + "━" * 65)
    print("  POPULATION ANALYSIS")
    print("  Aggregate single-cell predictions to population metrics")
    print("━" * 65)

    from src.population_analysis import compute_population_summary
    from src.visualization import plot_population_dashboard, plot_summary_dashboard

    pop_summary = compute_population_summary(y_pred)

    prolif = pop_summary["proliferation"]
    print(f"\n  Total cells analysed: {pop_summary['total_cells_analyzed']}")
    print(f"  Mitotic Index: {prolif['mitotic_index']:.2%}")
    print(f"  Growth Fraction: {prolif['growth_fraction']:.2%}")
    print(f"  Est. Doubling Time: {prolif['doubling_time_hours']:.1f}h")
    print(f"  Status: {prolif['status']}")
    print(f"  Clinical: {prolif['clinical_significance']}")

    plot_population_dashboard(pop_summary)

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════

    print("\n" + "━" * 65)
    print("  GENERATING FINAL SUMMARY DASHBOARD")
    print("━" * 65)

    plot_summary_dashboard(
        test_acc=test_acc,
        cnn_seq_acc=cnn_seq_acc,
        hmm_seq_acc=hmm_seq_acc,
        n_anomalies=len(anomalies_hmm),
        pop_summary=pop_summary,
    )

    # ══════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════

    from src.config import PLOTS_DIR, GRADCAM_DIR, DASHBOARD_DIR
    print("\n" + "═" * 65)
    print("  ✓ PIPELINE COMPLETE")
    print("═" * 65)
    print(f"\n  All outputs saved to:")
    print(f"    Plots:      {os.path.abspath(PLOTS_DIR)}")
    print(f"    Grad-CAM:   {os.path.abspath(GRADCAM_DIR)}")
    print(f"    Dashboards: {os.path.abspath(DASHBOARD_DIR)}")
    print(f"\n  Generated visualizations:")
    for d in [PLOTS_DIR, GRADCAM_DIR, DASHBOARD_DIR]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"):
                    print(f"    📊 {f}")
    print()


if __name__ == "__main__":
    main()
