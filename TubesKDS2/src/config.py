"""
Configuration module for the Cell Cycle Intelligence System.
Contains all hyperparameters, biological constants, and project paths.
"""

import os
import torch

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
GRADCAM_DIR = os.path.join(PLOTS_DIR, "gradcam")
DASHBOARD_DIR = os.path.join(PLOTS_DIR, "dashboard")

# ============================================================
# CELL CYCLE PHASES
# ============================================================
PHASES = ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase"]
NUM_PHASES = len(PHASES)
PHASE_TO_IDX = {phase: idx for idx, phase in enumerate(PHASES)}
IDX_TO_PHASE = {idx: phase for idx, phase in enumerate(PHASES)}

INTERPHASE = ["G1", "S", "G2"]
MITOTIC = ["Prophase", "Metaphase", "Anaphase", "Telophase"]

# Vibrant neon palette for dark-themed visualizations
PHASE_COLORS = {
    "G1":        "#00E676",  # Neon Green
    "S":         "#448AFF",  # Bright Blue
    "G2":        "#E040FB",  # Neon Purple
    "Prophase":  "#FF9100",  # Vivid Orange
    "Metaphase": "#FF1744",  # Vivid Red
    "Anaphase":  "#F50057",  # Hot Pink
    "Telophase": "#00E5FF",  # Cyan
}
PHASE_COLOR_LIST = [PHASE_COLORS[p] for p in PHASES]

# ============================================================
# IMAGE SETTINGS
# ============================================================
IMG_SIZE = 128
IMG_CHANNELS = 3
IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet normalization
IMG_STD  = [0.229, 0.224, 0.225]

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 5            # Increase to 20+ with GPU for best results
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
NUM_SAMPLES_PER_PHASE = 200   # (synthetic fallback only)
MAX_SAMPLES_PER_PHASE = 800   # Cap per phase from BBBC048 (None = use all)

# ============================================================
# BIOLOGICAL CONSTANTS — Cell Cycle Durations (hours)
# ============================================================
PHASE_DURATIONS = {
    "G1":        10.0,
    "S":          7.0,
    "G2":         4.0,
    "Prophase":   0.50,
    "Metaphase":  0.33,
    "Anaphase":   0.08,
    "Telophase":  0.25,
}
TOTAL_CYCLE_HOURS = sum(PHASE_DURATIONS.values())

# ============================================================
# ODE PARAMETERS — Simplified Tyson-Novak Cyclin-CDK Model
# ============================================================
ODE_PARAMS = {
    "k_syn":       0.04,    # Cyclin B synthesis rate
    "k_deg_basal": 0.04,    # Basal Cyclin B degradation
    "k_deg_apc":   1.0,     # APC-mediated Cyclin B degradation
    "k_act":       1.0,     # CDK1 activation rate (Cdc25)
    "k_inact":     0.5,     # CDK1 inactivation rate (Wee1)
    "k_apc_on":    0.1,     # APC activation by active CDK1
    "k_apc_off":   0.05,    # APC inactivation
    "J_act":       0.05,    # Michaelis constant for CDK1 activation
    "J_inact":     0.05,    # Michaelis constant for CDK1 inactivation
    "J_apc_on":    0.05,    # Michaelis constant for APC activation
    "J_apc_off":   0.05,    # Michaelis constant for APC inactivation
    "n":           4,       # Hill coefficient for ultrasensitivity
}
ODE_T_SPAN = (0, 100)       # Simulation time (arbitrary units)
ODE_T_POINTS = 5000         # Number of time evaluation points

# ============================================================
# HMM SETTINGS
# ============================================================
HMM_EMISSION_NOISE = 0.10   # Noise added to CNN softmax for realism
TIMELAPSE_NUM_FRAMES = 150   # Simulated time-lapse length
TIMELAPSE_HOURS = 30.0       # Total hours covered by time-lapse

# ============================================================
# CHECKPOINT ANOMALY THRESHOLDS
# ============================================================
CHECKPOINT_THRESHOLDS = {
    "G1_min_hours":       6.0,
    "G1_max_hours":      14.0,
    "S_min_hours":        4.0,
    "S_max_hours":       10.0,
    "G2_min_hours":       2.0,
    "G2_max_hours":       6.0,
    "metaphase_max_min": 40.0,   # minutes
    "expected_daughters":  2,
}

# ============================================================
# HELPER — create all output directories
# ============================================================
def setup_directories():
    """Create all necessary output directories."""
    for d in [DATA_DIR, SYNTHETIC_DIR, OUTPUT_DIR, MODEL_DIR,
              PLOTS_DIR, GRADCAM_DIR, DASHBOARD_DIR]:
        os.makedirs(d, exist_ok=True)
        for phase in PHASES:
            phase_dir = os.path.join(SYNTHETIC_DIR, phase)
            os.makedirs(phase_dir, exist_ok=True)
