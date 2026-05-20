"""
BBBC048 Dataset Loader
=======================
Downloads and extracts the Broad Bioimage Benchmark Collection BBBC048:
  - 32,266 real fluorescence microscopy images of Jurkat cells
  - Already labeled by phase: G1, S, G2, Prophase, Metaphase, Anaphase, Telophase
  - Imaged on ImageStream (PI + MPM2 staining)

The zip contains pre-organized folders:
    CellCycle/<Phase>/<id>_merged.jpg

Reference:
  Eulenberg et al. "Reconstructing cell cycle and disease progression
  using deep learning." Nature Communications 8, 463 (2017).
License: CC BY-NC-SA 3.0
"""

import os
import sys
import zipfile
import urllib.request
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import DATA_DIR, SYNTHETIC_DIR as DATASET_DIR, PHASES, IMG_SIZE

# ── URLs ──────────────────────────────────────────────────────
IMAGES_URL = "https://data.broadinstitute.org/bbbc/BBBC048/BBBC048v1.zip"
IMAGES_ZIP = os.path.join(DATA_DIR, "BBBC048v1.zip")
CELLCYCLE_ZIP = None  # discovered at runtime


class _DownloadProgress:
    """Progress bar for urllib downloads."""
    def __init__(self, desc="Downloading"):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit="B", unit_scale=True,
                             desc=f"  {self.desc}", ncols=80)
        self.pbar.update(block_size)
        if block_num * block_size >= total_size and self.pbar:
            self.pbar.close()


def _find_cellcycle_zip():
    """Locate CellCycle.zip inside the extracted BBBC048 archive."""
    raw_dir = os.path.join(DATA_DIR, "BBBC048_raw")
    candidate = os.path.join(raw_dir, "CellCycle.zip")
    if os.path.exists(candidate):
        return candidate
    # Search recursively
    for dirpath, _, filenames in os.walk(raw_dir):
        for fn in filenames:
            if fn == "CellCycle.zip":
                return os.path.join(dirpath, fn)
    return None


def download_dataset():
    """Download BBBC048 images (~1.5 GB). Skips if already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_dir = os.path.join(DATA_DIR, "BBBC048_raw")

    if not os.path.exists(IMAGES_ZIP) and not os.path.exists(raw_dir):
        print(f"\n  Downloading BBBC048 images (~1.5 GB)...")
        print(f"  Source: {IMAGES_URL}")
        urllib.request.urlretrieve(IMAGES_URL, IMAGES_ZIP,
                                  _DownloadProgress("BBBC048v1.zip"))
        print(f"  Saved to {IMAGES_ZIP}")

    # Extract outer zip
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
        if os.path.exists(IMAGES_ZIP):
            print(f"\n  Extracting outer archive...")
            os.makedirs(raw_dir, exist_ok=True)
            with zipfile.ZipFile(IMAGES_ZIP, 'r') as zf:
                zf.extractall(raw_dir)
            print(f"  Extracted to {raw_dir}")
    else:
        print(f"  Archive already extracted: {raw_dir}")


def organize_by_phase(max_per_phase=None):
    """
    Extract merged images from CellCycle.zip into phase-labeled directories:
        DATASET_DIR/<Phase>/img_XXXXX.png

    Args:
        max_per_phase: max images per phase (None = all). Use e.g. 500 for fast runs.
    """
    cellcycle_zip_path = _find_cellcycle_zip()
    if cellcycle_zip_path is None:
        print("  ERROR: CellCycle.zip not found in extracted data.")
        return 0

    print(f"\n  Extracting images from: {cellcycle_zip_path}")
    print(f"  Using merged channel images (_merged.jpg)")

    zf = zipfile.ZipFile(cellcycle_zip_path, 'r')
    members = zf.namelist()

    # Create phase dirs
    for phase in PHASES:
        os.makedirs(os.path.join(DATASET_DIR, phase), exist_ok=True)

    # Collect merged images per phase
    phase_files = {p: [] for p in PHASES}
    for m in members:
        if "_merged.jpg" not in m:
            continue
        parts = m.replace("\\", "/").split("/")
        # Expected: CellCycle/<Phase>/<id>_merged.jpg
        if len(parts) >= 3:
            phase_name = parts[-2]  # directory name
            if phase_name in PHASES:
                phase_files[phase_name].append(m)

    # Print distribution
    print(f"\n  Dataset distribution (merged images):")
    for phase in PHASES:
        count = len(phase_files[phase])
        bar = chr(9608) * min(count // 100, 80)
        print(f"    {phase:>12s}: {count:>6d}  {bar}")

    total_available = sum(len(v) for v in phase_files.values())
    print(f"    {'Total':>12s}: {total_available:>6d}")

    # Extract and convert
    counts = {p: 0 for p in PHASES}
    total_to_process = sum(
        min(len(phase_files[p]), max_per_phase) if max_per_phase
        else len(phase_files[p])
        for p in PHASES
    )

    pbar = tqdm(total=total_to_process, desc="  Extracting", ncols=80)

    for phase in PHASES:
        files = phase_files[phase]
        if max_per_phase:
            files = files[:max_per_phase]

        for member in files:
            try:
                with zf.open(member) as f:
                    img = Image.open(f)
                    img = img.convert("RGB")
                    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                    dst = os.path.join(DATASET_DIR, phase,
                                       f"img_{counts[phase]:05d}.png")
                    img.save(dst)
                    counts[phase] += 1
            except Exception:
                pass
            pbar.update(1)

    pbar.close()
    zf.close()

    total = sum(counts.values())
    print(f"\n  Organized {total} images into {DATASET_DIR}")
    for phase in PHASES:
        print(f"    {phase:>12s}: {counts[phase]:>5d} images")

    return total


def prepare_dataset(max_per_phase=None):
    """
    Full pipeline: Download -> Extract -> Organize.
    """
    print(f"\n{'='*60}")
    print(f"  BBBC048 DATASET PREPARATION")
    print(f"  Jurkat Cell Cycle Fluorescence Microscopy")
    print(f"{'='*60}\n")

    # Check if already organized
    existing = 0
    for phase in PHASES:
        phase_dir = os.path.join(DATASET_DIR, phase)
        if os.path.isdir(phase_dir):
            existing += len([f for f in os.listdir(phase_dir) if f.endswith('.png')])

    if existing > 100:
        print(f"  Dataset already prepared ({existing} images found)")
        print(f"  Location: {DATASET_DIR}")
        return existing

    download_dataset()
    return organize_by_phase(max_per_phase=max_per_phase)
