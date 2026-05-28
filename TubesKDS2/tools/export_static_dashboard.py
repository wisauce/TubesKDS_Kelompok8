"""Export static dashboard assets.

Copies key plots from `output/plots/` into `api/static/plots/` and writes
`manifest.json` with metrics so the frontend can render without loading the test set.

Run (from `TubesKDS2/`):
  python tools/export_static_dashboard.py

Optional flags:
  --skip-metrics   Only copy images and keep existing manifest (if any)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUTPUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "output", "plots")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "api", "static", "plots")
MANIFEST_PATH = os.path.join(ASSETS_DIR, "manifest.json")


FILES_TO_COPY: list[tuple[str, str]] = [
    (os.path.join(OUTPUT_PLOTS_DIR, "02_confusion_matrix.png"), "02_confusion_matrix.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "01_training_curves.png"), "01_training_curves.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "gradcam", "03_gradcam_analysis.png"), "03_gradcam_analysis.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "dashboard", "07_population_dashboard.png"), "07_population_dashboard.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "dashboard", "09_summary.png"), "09_summary.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "04_ode_cyclin_dynamics.png"), "04_ode_cyclin_dynamics.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "05_hmm_correction.png"), "05_hmm_correction.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "06_checkpoint_anomalies.png"), "06_checkpoint_anomalies.png"),
    (os.path.join(OUTPUT_PLOTS_DIR, "08_transition_matrix.png"), "08_transition_matrix.png"),
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_assets() -> list[str]:
    missing: list[str] = []
    for src, dst_name in FILES_TO_COPY:
        dst = os.path.join(ASSETS_DIR, dst_name)
        if not os.path.exists(src):
            missing.append(src)
            continue
        shutil.copy2(src, dst)
    return missing


def _fmt_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.0f}K"
    return str(n)


def _build_manifest() -> dict:
    # Local imports to keep script lightweight when using --skip-metrics
    import torch

    from src.config import DEVICE, MODEL_DIR, NUM_EPOCHS
    from src.model import CellCycleCNN, evaluate_test
    from src.dataset import create_dataloaders
    from src.population_analysis import compute_population_summary

    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path} (run main.py first)")

    _, _, test_loader = create_dataloaders()

    model = CellCycleCNN(pretrained=False)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    y_pred, y_true, _ = evaluate_test(model, test_loader)

    test_accuracy = float((y_pred == y_true).mean())
    total_samples = int(len(test_loader.dataset))

    total_params = sum(p.numel() for p in model.parameters())

    pop = compute_population_summary(y_pred.tolist())
    prolif = pop["proliferation"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "weights": os.path.relpath(model_path, PROJECT_ROOT),
            "train_epochs": int(NUM_EPOCHS),
            "device": str(DEVICE),
            "params": _fmt_param_count(int(total_params)),
        },
        "cnn": {
            "test_accuracy": test_accuracy,
            "total_samples": total_samples,
            "train_epochs": int(NUM_EPOCHS),
            "model_params": _fmt_param_count(int(total_params)),
            "images": {
                "confusion": "02_confusion_matrix.png",
                "curves": "01_training_curves.png",
            },
        },
        "gradcam": {"images": {"batch": "03_gradcam_analysis.png"}},
        "population": {
            "total_cells": int(pop["total_cells_analyzed"]),
            "mitotic_index": float(prolif["mitotic_index"]),
            "growth_fraction": float(prolif["growth_fraction"]),
            "doubling_time_hours": float(prolif["doubling_time_hours"]),
            "status": prolif["status"],
            "clinical_significance": prolif["clinical_significance"],
            "images": {"plot": "07_population_dashboard.png"},
        },
        "simulation": {
            "images": {
                "ode": "04_ode_cyclin_dynamics.png",
                "hmm": "05_hmm_correction.png",
                "checkpoint": "06_checkpoint_anomalies.png",
                "transition_matrix": "08_transition_matrix.png",
                "summary": "09_summary.png",
            }
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-metrics", action="store_true", help="Only copy images; do not recompute manifest")
    args = parser.parse_args()

    _ensure_dir(ASSETS_DIR)

    missing = _copy_assets()
    if missing:
        print("WARNING: Some plot files are missing; re-run main.py to regenerate:")
        for m in missing:
            print(f"  - {m}")

    if not args.skip_metrics:
        manifest = _build_manifest()
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"OK wrote {os.path.relpath(MANIFEST_PATH, PROJECT_ROOT)}")
    else:
        print("OK copied images (metrics skipped)")

    print(f"OK assets dir: {os.path.relpath(ASSETS_DIR, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
