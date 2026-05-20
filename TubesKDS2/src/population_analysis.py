"""
Population Analysis Module
===========================
Aggregate single-cell classifications to compute population-level metrics:
  - Mitotic index
  - Phase distribution
  - Estimated doubling time
  - Growth fraction
"""

import numpy as np
from collections import Counter
from src.config import PHASES, PHASE_TO_IDX, INTERPHASE, MITOTIC, PHASE_DURATIONS, TOTAL_CYCLE_HOURS


def compute_phase_distribution(predictions):
    """
    Compute the fraction of cells in each phase.

    Args:
        predictions: array of phase indices or list of phase names

    Returns:
        dict: {phase_name: fraction}
    """
    if len(predictions) == 0:
        return {p: 0.0 for p in PHASES}

    # Convert indices to names if needed
    if isinstance(predictions[0], (int, np.integer)):
        from src.config import IDX_TO_PHASE
        names = [IDX_TO_PHASE[int(p)] for p in predictions]
    else:
        names = list(predictions)

    counts = Counter(names)
    total = len(names)

    distribution = {}
    for phase in PHASES:
        distribution[phase] = counts.get(phase, 0) / total

    return distribution


def compute_mitotic_index(predictions):
    """
    Compute the mitotic index: fraction of cells in mitotic phases.

    Biological significance:
      - Normal tissue: MI ≈ 1-3%
      - Rapidly dividing (e.g. bone marrow): MI ≈ 5-10%
      - Tumor tissue: MI > 10% (indicates uncontrolled proliferation)
    """
    dist = compute_phase_distribution(predictions)
    mi = sum(dist.get(p, 0) for p in MITOTIC)
    return mi


def compute_growth_fraction(predictions):
    """
    Compute the growth fraction: fraction of cells actively cycling
    (not in quiescence/G0). Since our model doesn't detect G0,
    we use S+G2+M as a proxy for actively cycling cells.
    """
    dist = compute_phase_distribution(predictions)
    active = sum(dist.get(p, 0) for p in ["S", "G2"] + MITOTIC)
    return active


def estimate_doubling_time(predictions):
    """
    Estimate population doubling time from phase distribution.

    Uses the relationship between phase fraction and phase duration:
      T_total / T_phase = 1 / fraction_in_phase

    We use the S-phase fraction (most reliable indicator of proliferation rate).
    """
    dist = compute_phase_distribution(predictions)
    s_fraction = dist.get("S", 0)

    if s_fraction < 0.01:
        return float('inf')  # Effectively non-dividing

    # T_total = T_s / fraction_s
    t_s = PHASE_DURATIONS["S"]  # typical S phase duration in hours
    doubling_time = t_s / s_fraction

    return doubling_time


def classify_proliferation_status(predictions):
    """
    Classify the population's proliferation status based on mitotic index.

    Returns:
        dict with status, description, and clinical significance
    """
    mi = compute_mitotic_index(predictions)
    gf = compute_growth_fraction(predictions)
    dt = estimate_doubling_time(predictions)

    if mi < 0.03:
        status = "Quiescent"
        desc = "Low proliferative activity"
        clinical = "Normal differentiated tissue pattern"
    elif mi < 0.10:
        status = "Normally Proliferating"
        desc = "Moderate proliferative activity"
        clinical = "Consistent with healthy dividing tissue (e.g. epithelium, bone marrow)"
    elif mi < 0.20:
        status = "Highly Proliferative"
        desc = "Elevated mitotic activity"
        clinical = "May indicate hyperplasia or early neoplastic transformation"
    else:
        status = "Abnormally Proliferative"
        desc = "Very high mitotic activity"
        clinical = "Consistent with aggressive tumor phenotype — further analysis recommended"

    return {
        "status": status,
        "description": desc,
        "clinical_significance": clinical,
        "mitotic_index": mi,
        "growth_fraction": gf,
        "doubling_time_hours": dt,
    }


def compute_population_summary(predictions):
    """
    Generate a complete population analysis summary.

    Returns:
        dict with all population-level metrics
    """
    dist = compute_phase_distribution(predictions)
    prolif = classify_proliferation_status(predictions)

    # Phase duration estimates
    phase_durations_est = {}
    for phase in PHASES:
        frac = dist.get(phase, 0)
        if frac > 0:
            # Estimated duration = fraction × total cycle time
            phase_durations_est[phase] = frac * prolif["doubling_time_hours"]
        else:
            phase_durations_est[phase] = 0.0

    return {
        "phase_distribution": dist,
        "proliferation": prolif,
        "estimated_phase_durations": phase_durations_est,
        "total_cells_analyzed": len(predictions),
        "interphase_fraction": sum(dist.get(p, 0) for p in INTERPHASE),
        "mitotic_fraction": sum(dist.get(p, 0) for p in MITOTIC),
    }
