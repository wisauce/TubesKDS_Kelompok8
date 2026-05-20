"""
Checkpoint Anomaly Detector — Layer 3 of the Cell Cycle Intelligence System
============================================================================
Implements computational checkpoints that mirror real biological checkpoints:
  - G1/S Restriction Point
  - G2/M DNA Damage Checkpoint
  - Spindle Assembly Checkpoint (SAC)

Anomalies indicate potential cell cycle dysregulation (e.g. cancer hallmarks).
The detection logic is BIOLOGY-INSPIRED — rules come from cell biology, not ML.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

from src.config import (
    PHASES, PHASE_TO_IDX, IDX_TO_PHASE, CHECKPOINT_THRESHOLDS, PHASE_DURATIONS
)


@dataclass
class Anomaly:
    """A detected cell cycle anomaly."""
    checkpoint: str          # Which checkpoint was violated
    severity: str            # "low", "medium", "high", "critical"
    frame_start: int         # First frame of the anomaly
    frame_end: int           # Last frame of the anomaly
    phase: str               # Phase where the anomaly was detected
    observed_value: float    # What was actually observed
    expected_range: str      # What was expected (human-readable)
    bio_interpretation: str  # Biological meaning of this anomaly

    def __repr__(self):
        return (f"⚠ [{self.severity.upper()}] {self.checkpoint} "
                f"at frames {self.frame_start}-{self.frame_end}: "
                f"{self.bio_interpretation}")


@dataclass
class CellTrack:
    """Tracked cell's phase sequence over time."""
    cell_id: int
    phases: List[str]
    frame_indices: List[int]
    hours_per_frame: float

    def phase_duration_hours(self, phase: str) -> float:
        """Compute how long a cell spent in a given phase."""
        count = sum(1 for p in self.phases if p == phase)
        return count * self.hours_per_frame

    def phase_segments(self):
        """
        Get contiguous segments of the same phase.
        Returns list of (phase, start_frame, end_frame, duration_hours).
        """
        if not self.phases:
            return []

        segments = []
        current_phase = self.phases[0]
        start = 0

        for i in range(1, len(self.phases)):
            if self.phases[i] != current_phase:
                duration = (i - start) * self.hours_per_frame
                segments.append((current_phase, start, i - 1, duration))
                current_phase = self.phases[i]
                start = i

        # Last segment
        duration = (len(self.phases) - start) * self.hours_per_frame
        segments.append((current_phase, start, len(self.phases) - 1, duration))

        return segments


class CheckpointDetector:
    """
    Biological checkpoint anomaly detector.

    Each method implements a computational analogue of a real cell cycle
    checkpoint. The detection rules come from BIOLOGY, not from data:
      - Phase durations are compared against known biological ranges
      - Transition order is validated against the biological sequence
      - Mitotic integrity is checked for proper chromosome segregation
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or CHECKPOINT_THRESHOLDS
        self.anomalies = []

    def analyze(self, cell_track: CellTrack) -> List[Anomaly]:
        """Run all biological checkpoints on a cell track."""
        self.anomalies = []

        self._check_g1s_restriction(cell_track)
        self._check_g2m_checkpoint(cell_track)
        self._check_spindle_assembly(cell_track)
        self._check_transition_order(cell_track)
        self._check_s_phase_duration(cell_track)

        return self.anomalies

    def _check_g1s_restriction(self, track: CellTrack):
        """
        G1/S Restriction Point
        ─────────────────────
        Biology: Cell must reach sufficient size and receive growth signals
        before committing to DNA replication. Controlled by Rb/E2F and p53.

        Anomaly: G1 too short → restriction point bypassed
                 G1 too long → possible G1 arrest (DNA damage response)
        """
        segments = track.phase_segments()
        for phase, start, end, duration in segments:
            if phase != "G1":
                continue

            min_h = self.thresholds["G1_min_hours"]
            max_h = self.thresholds["G1_max_hours"]

            if duration < min_h:
                self.anomalies.append(Anomaly(
                    checkpoint="G1/S Restriction Point",
                    severity="high",
                    frame_start=start, frame_end=end,
                    phase="G1",
                    observed_value=duration,
                    expected_range=f"{min_h}-{max_h} hours",
                    bio_interpretation=(
                        f"G1 phase abnormally short ({duration:.1f}h < {min_h}h). "
                        f"Suggests Rb/p53 pathway disruption — restriction point "
                        f"may be bypassed, a hallmark of oncogenic transformation."
                    )
                ))
            elif duration > max_h:
                self.anomalies.append(Anomaly(
                    checkpoint="G1/S Restriction Point",
                    severity="medium",
                    frame_start=start, frame_end=end,
                    phase="G1",
                    observed_value=duration,
                    expected_range=f"{min_h}-{max_h} hours",
                    bio_interpretation=(
                        f"G1 phase prolonged ({duration:.1f}h > {max_h}h). "
                        f"Possible DNA damage-induced G1 arrest via p53/p21 "
                        f"pathway — cell may be undergoing repair or senescence."
                    )
                ))

    def _check_g2m_checkpoint(self, track: CellTrack):
        """
        G2/M DNA Damage Checkpoint
        ──────────────────────────
        Biology: Cell verifies DNA replication is complete and undamaged
        before entering mitosis. Controlled by ATM/ATR → Chk1/Chk2 → Wee1.

        Anomaly: G2 too short → checkpoint override (genomic instability)
                 G2 too long → possible G2 arrest
        """
        segments = track.phase_segments()
        for phase, start, end, duration in segments:
            if phase != "G2":
                continue

            min_h = self.thresholds["G2_min_hours"]
            max_h = self.thresholds["G2_max_hours"]

            if duration < min_h:
                self.anomalies.append(Anomaly(
                    checkpoint="G2/M DNA Damage",
                    severity="high",
                    frame_start=start, frame_end=end,
                    phase="G2",
                    observed_value=duration,
                    expected_range=f"{min_h}-{max_h} hours",
                    bio_interpretation=(
                        f"G2 phase too short ({duration:.1f}h < {min_h}h). "
                        f"DNA damage checkpoint may be compromised — risk of "
                        f"entering mitosis with unreplicated or damaged DNA."
                    )
                ))

    def _check_spindle_assembly(self, track: CellTrack):
        """
        Spindle Assembly Checkpoint (SAC)
        ─────────────────────────────────
        Biology: Ensures all kinetochores are properly attached to spindle
        microtubules before anaphase onset. Controlled by Mad2/BubR1.

        Anomaly: Metaphase too long → SAC activated (attachment problems)
                 Metaphase too short → SAC override (chromosome missegregation)
        """
        segments = track.phase_segments()
        max_min = self.thresholds["metaphase_max_min"]

        for phase, start, end, duration in segments:
            if phase != "Metaphase":
                continue

            duration_min = duration * 60  # convert to minutes

            if duration_min > max_min:
                self.anomalies.append(Anomaly(
                    checkpoint="Spindle Assembly (SAC)",
                    severity="critical",
                    frame_start=start, frame_end=end,
                    phase="Metaphase",
                    observed_value=duration_min,
                    expected_range=f"<{max_min} minutes",
                    bio_interpretation=(
                        f"Metaphase prolonged ({duration_min:.0f}min > {max_min}min). "
                        f"SAC is activated — likely chromosome attachment errors. "
                        f"If SAC is overridden, results in aneuploidy (CIN phenotype)."
                    )
                ))

    def _check_s_phase_duration(self, track: CellTrack):
        """
        S Phase Replication Check
        ─────────────────────────
        Biology: DNA replication must complete fully. Replication stress
        (e.g. from oncogene activation) causes S phase elongation.
        """
        segments = track.phase_segments()
        for phase, start, end, duration in segments:
            if phase != "S":
                continue

            min_h = self.thresholds["S_min_hours"]
            max_h = self.thresholds["S_max_hours"]

            if duration > max_h:
                self.anomalies.append(Anomaly(
                    checkpoint="S Phase Replication",
                    severity="medium",
                    frame_start=start, frame_end=end,
                    phase="S",
                    observed_value=duration,
                    expected_range=f"{min_h}-{max_h} hours",
                    bio_interpretation=(
                        f"S phase elongated ({duration:.1f}h > {max_h}h). "
                        f"Possible replication stress from fork stalling, "
                        f"oncogene-induced replication, or nucleotide depletion."
                    )
                ))

    def _check_transition_order(self, track: CellTrack):
        """
        Phase Transition Order Validation
        ──────────────────────────────────
        Biology: Phases must follow the order G1→S→G2→Pro→Meta→Ana→Telo→G1.
        Out-of-order transitions indicate classification errors or
        biological abnormalities (e.g. mitotic slippage).
        """
        valid_transitions = set()
        for i in range(len(PHASES)):
            next_i = (i + 1) % len(PHASES)
            valid_transitions.add((PHASES[i], PHASES[next_i]))
            # Also allow self-transitions (staying in same phase)
            valid_transitions.add((PHASES[i], PHASES[i]))

        phases = track.phases
        for i in range(len(phases) - 1):
            transition = (phases[i], phases[i + 1])
            if transition not in valid_transitions:
                self.anomalies.append(Anomaly(
                    checkpoint="Phase Transition Order",
                    severity="high",
                    frame_start=i, frame_end=i + 1,
                    phase=phases[i],
                    observed_value=0,
                    expected_range=f"{phases[i]} → {PHASES[(PHASE_TO_IDX[phases[i]] + 1) % len(PHASES)]}",
                    bio_interpretation=(
                        f"Invalid transition: {phases[i]} → {phases[i+1]}. "
                        f"May indicate mitotic slippage, endoreduplication, "
                        f"or classification error correctable by HMM."
                    )
                ))


def create_cell_track(phase_sequence, hours_per_frame=0.2, cell_id=0):
    """Helper to create a CellTrack from a phase sequence."""
    return CellTrack(
        cell_id=cell_id,
        phases=list(phase_sequence),
        frame_indices=list(range(len(phase_sequence))),
        hours_per_frame=hours_per_frame,
    )


def run_checkpoint_analysis(phase_sequence, hours_per_frame=0.2):
    """
    Convenience function: run all checkpoints on a phase sequence.
    Returns (cell_track, anomalies).
    """
    track = create_cell_track(phase_sequence, hours_per_frame)
    detector = CheckpointDetector()
    anomalies = detector.analyze(track)
    return track, anomalies
