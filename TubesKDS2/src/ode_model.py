"""
Cyclin-CDK ODE Model — Layer 2a of the Cell Cycle Intelligence System
=====================================================================
Implements a simplified Tyson-Novak model of the cyclin B / CDK1 / APC
oscillator that drives cell cycle progression.

The ODE solution provides:
  1. Cyclin concentration dynamics over time
  2. Phase boundaries (when specific thresholds are crossed)
  3. Transition probabilities for the HMM
"""

import numpy as np
from scipy.integrate import solve_ivp

from src.config import ODE_PARAMS, ODE_T_SPAN, ODE_T_POINTS, PHASES, PHASE_DURATIONS


# ── Goldbeter-style Michaelis function ──────────────────────

def _goldbeter_koshland(v1, v2, J1, J2):
    """
    Goldbeter-Koshland ultrasensitive switch function.
    Models the bistable behaviour of CDK1/APC activation.
    """
    B = v2 - v1 + J1 * v2 + J2 * v1
    disc = B ** 2 - 4 * (v2 - v1) * J2 * v1
    disc = max(disc, 0)  # numerical safety
    return (2 * J2 * v1) / (B + np.sqrt(disc) + 1e-12)


# ── ODE system ────────────────────────────────────────────────

def cyclin_cdk_ode(t, y, params):
    """
    Simplified Tyson-Novak ODE system:
        y[0] = [CycB]   — total Cyclin B concentration
        y[1] = [CDK1*]  — fraction of active CDK1
        y[2] = [APC*]   — fraction of active APC/C
    """
    CycB, CDK1, APC = y
    p = params

    # Cyclin B dynamics: synthesis – (basal + APC-mediated) degradation
    dCycB = p["k_syn"] - (p["k_deg_basal"] + p["k_deg_apc"] * APC) * CycB

    # CDK1 activation via Cdc25 (positive feedback) / inactivation via Wee1
    # Using Goldbeter-Koshland for ultrasensitivity
    cdc25_rate = p["k_act"] * CycB  # Cdc25 activated by Cyclin B
    wee1_rate = p["k_inact"]
    CDK1_ss = _goldbeter_koshland(cdc25_rate, wee1_rate, p["J_act"], p["J_inact"])
    dCDK1 = (CDK1_ss - CDK1) * 2.0  # relaxation towards steady state

    # APC activation by active CDK1 (negative feedback loop)
    apc_on_rate = p["k_apc_on"] * CDK1
    apc_off_rate = p["k_apc_off"]
    APC_ss = _goldbeter_koshland(apc_on_rate, apc_off_rate, p["J_apc_on"], p["J_apc_off"])
    dAPC = (APC_ss - APC) * 1.5

    return [dCycB, dCDK1, dAPC]


def solve_cyclin_dynamics(params=None):
    """
    Solve the cyclin-CDK ODE system.

    Returns:
        t: time array
        CycB: Cyclin B concentration over time
        CDK1: active CDK1 fraction over time
        APC: active APC fraction over time
        phase_boundaries: list of (time, phase_name) tuples
    """
    if params is None:
        params = ODE_PARAMS

    # Initial conditions: start in G1 (low CycB, low CDK1, high APC)
    y0 = [0.05, 0.01, 0.9]

    t_eval = np.linspace(ODE_T_SPAN[0], ODE_T_SPAN[1], ODE_T_POINTS)
    sol = solve_ivp(cyclin_cdk_ode, ODE_T_SPAN, y0,
                    args=(params,), t_eval=t_eval,
                    method="RK45", max_step=0.1)

    t = sol.t
    CycB = sol.y[0]
    CDK1 = sol.y[1]
    APC  = sol.y[2]

    # Determine phase boundaries from cyclin/CDK dynamics
    phase_boundaries = _assign_phases(t, CycB, CDK1, APC)

    return {
        "t": t,
        "CycB": CycB,
        "CDK1": CDK1,
        "APC": APC,
        "phase_boundaries": phase_boundaries,
    }


def _assign_phases(t, CycB, CDK1, APC):
    """
    Assign cell cycle phases based on cyclin-CDK state.

    Biological logic:
        G1:        low CycB, low CDK1, high APC
        S:         rising CycB, low CDK1
        G2:        high CycB, low CDK1 (pre-activation)
        Prophase:  CDK1 activating (switch flipping)
        Metaphase: CDK1 fully active, APC not yet active
        Anaphase:  APC activating (destroying CycB)
        Telophase: CycB falling, CDK1 inactivating
    """
    n = len(t)
    phases = np.empty(n, dtype=object)

    for i in range(n):
        cb, cdk, apc = CycB[i], CDK1[i], APC[i]

        if cb < 0.15 and cdk < 0.2:
            phases[i] = "G1"
        elif cb < 0.35 and cdk < 0.2:
            phases[i] = "S"
        elif cb >= 0.35 and cdk < 0.5:
            phases[i] = "G2"
        elif 0.5 <= cdk < 0.8 and apc < 0.4:
            phases[i] = "Prophase"
        elif cdk >= 0.8 and apc < 0.5:
            phases[i] = "Metaphase"
        elif apc >= 0.5 and cb > 0.15:
            phases[i] = "Anaphase"
        elif apc >= 0.5 and cb <= 0.15:
            phases[i] = "Telophase"
        else:
            phases[i] = "G2"  # fallback

    return phases


def compute_transition_matrix(ode_results):
    """
    Derive empirical phase transition probabilities from the ODE solution.
    This becomes the transition matrix for the HMM.

    Returns:
        trans_matrix: (NUM_PHASES, NUM_PHASES) numpy array
        phase_fractions: dict of phase → fraction of total cycle
    """
    from src.config import PHASE_TO_IDX, NUM_PHASES

    phases = ode_results["phase_boundaries"]
    n = len(phases)

    # Count transitions
    counts = np.zeros((NUM_PHASES, NUM_PHASES), dtype=np.float64)
    for i in range(n - 1):
        p_from = PHASE_TO_IDX.get(phases[i])
        p_to = PHASE_TO_IDX.get(phases[i + 1])
        if p_from is not None and p_to is not None:
            counts[p_from, p_to] += 1

    # Normalize rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    trans_matrix = counts / row_sums

    # Phase fractions
    phase_fractions = {}
    for phase in PHASES:
        phase_fractions[phase] = np.sum(phases == phase) / n

    return trans_matrix, phase_fractions
