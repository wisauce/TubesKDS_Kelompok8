"""
Hidden Markov Model — Layer 2b of the Cell Cycle Intelligence System
====================================================================
Uses ODE-derived transition probabilities to constrain CNN predictions
over time-lapse sequences.

The key insight: transition probabilities come from BIOLOGY (cyclin-CDK dynamics),
not from data — the algorithm is biologically informed.
"""

import numpy as np
from src.config import PHASES, NUM_PHASES, PHASE_TO_IDX, IDX_TO_PHASE, HMM_EMISSION_NOISE


class BiologicalHMM:
    """
    Hidden Markov Model where:
      - Hidden states = true cell cycle phases
      - Observations  = CNN softmax probability vectors
      - Transition matrix = derived from Cyclin-CDK ODE model
      - Emission model  = CNN classifier output (treated as observation likelihood)
    """

    def __init__(self, transition_matrix: np.ndarray, phase_fractions: dict = None):
        """
        Args:
            transition_matrix: (N, N) ODE-derived transition probabilities
            phase_fractions: dict of phase → initial probability
        """
        self.n_states = NUM_PHASES
        self.trans = self._smooth(transition_matrix)

        # Initial state distribution — from ODE phase fractions or uniform
        if phase_fractions is not None:
            self.pi = np.array([phase_fractions.get(p, 1.0 / self.n_states) for p in PHASES])
        else:
            self.pi = np.ones(self.n_states) / self.n_states
        self.pi /= self.pi.sum()

    @staticmethod
    def _smooth(matrix, epsilon=1e-4):
        """Add small floor probability to prevent zero-probability transitions."""
        smoothed = matrix + epsilon
        smoothed /= smoothed.sum(axis=1, keepdims=True)
        return smoothed

    def viterbi(self, observations: np.ndarray):
        """
        Viterbi algorithm — find the most likely phase sequence given
        CNN softmax outputs and ODE-derived transition constraints.

        Args:
            observations: (T, N) array of CNN softmax vectors per frame

        Returns:
            best_path: list of phase indices (length T)
            path_prob: log-probability of the best path
        """
        T = len(observations)
        N = self.n_states

        # Log domain for numerical stability
        log_trans = np.log(self.trans + 1e-12)
        log_pi = np.log(self.pi + 1e-12)

        # Viterbi tables
        V = np.full((T, N), -np.inf)
        backptr = np.zeros((T, N), dtype=int)

        # Emission log-likelihood = log(CNN softmax probability)
        log_emit = np.log(observations + 1e-12)

        # Initialisation
        V[0] = log_pi + log_emit[0]

        # Forward pass
        for t in range(1, T):
            for j in range(N):
                scores = V[t - 1] + log_trans[:, j]
                backptr[t, j] = np.argmax(scores)
                V[t, j] = scores[backptr[t, j]] + log_emit[t, j]

        # Backtrack
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(V[-1])
        path_prob = V[-1, best_path[-1]]

        for t in range(T - 2, -1, -1):
            best_path[t] = backptr[t + 1, best_path[t + 1]]

        return best_path.tolist(), path_prob

    def forward_backward(self, observations: np.ndarray):
        """
        Forward-backward algorithm — compute marginal posterior P(state_t | all obs).

        Returns:
            gamma: (T, N) posterior probabilities
        """
        T = len(observations)
        N = self.n_states

        # Forward
        alpha = np.zeros((T, N))
        alpha[0] = self.pi * observations[0]
        alpha[0] /= alpha[0].sum() + 1e-12

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = observations[t, j] * np.sum(alpha[t - 1] * self.trans[:, j])
            alpha[t] /= alpha[t].sum() + 1e-12

        # Backward
        beta = np.zeros((T, N))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(self.trans[i, :] * observations[t + 1] * beta[t + 1])
            beta[t] /= beta[t].sum() + 1e-12

        # Posterior
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12

        return gamma


def simulate_timelapse_observations(true_phases, model=None, noise=HMM_EMISSION_NOISE):
    """
    Simulate CNN softmax outputs for a known phase sequence.
    Used for testing the HMM pipeline without requiring actual images.

    Args:
        true_phases: list of phase names (ground truth sequence)
        model: (unused, for interface compatibility)
        noise: amount of noise to add to the "perfect" prediction

    Returns:
        observations: (T, NUM_PHASES) softmax-like probability matrix
    """
    T = len(true_phases)
    obs = np.zeros((T, NUM_PHASES))

    for t, phase in enumerate(true_phases):
        idx = PHASE_TO_IDX[phase]
        # Create a soft one-hot with noise
        obs[t] = np.random.dirichlet(np.ones(NUM_PHASES) * noise)
        obs[t, idx] += np.random.uniform(0.6, 0.85)  # boost true class
        obs[t] /= obs[t].sum()

    return obs


def generate_ground_truth_sequence(n_frames=150, total_hours=30.0):
    """
    Generate a biologically-valid ground truth phase sequence.

    Uses known phase durations to create a realistic time-lapse sequence
    spanning multiple cell cycles.
    """
    from src.config import PHASE_DURATIONS, TOTAL_CYCLE_HOURS

    hours_per_frame = total_hours / n_frames
    sequence = []
    current_hour = 0.0

    while len(sequence) < n_frames:
        for phase in PHASES:
            duration_hours = PHASE_DURATIONS[phase]
            n_phase_frames = max(1, int(duration_hours / hours_per_frame))
            sequence.extend([phase] * n_phase_frames)
            if len(sequence) >= n_frames:
                break

    return sequence[:n_frames]


def inject_noise_into_sequence(sequence, error_rate=0.15, seed=42):
    """
    Inject random classification errors into a phase sequence.
    Simulates what a noisy CNN would produce without HMM correction.
    """
    rng = np.random.RandomState(seed)
    noisy = list(sequence)

    for i in range(len(noisy)):
        if rng.random() < error_rate:
            noisy[i] = PHASES[rng.randint(0, NUM_PHASES)]

    return noisy
