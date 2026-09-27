"""Block-diagonal MMD^2 with delta kernel + permutation / simulation null.

A sample is a row in a (N, D) one-hot matrix where D = sum(block_sizes).
Each block is a categorical option-set for one probe template; the row's
entries within a block sum to 1 (one-hot) and the rows are independent draws.

Under the delta kernel k(a, b) = 1[a == b], per-block MMD^2 reduces to
||p^R_b - p^S_b||^22. The total MMD^2 is a weighted sum over blocks:

    MMD^2_total(p^R, p^S) = sum_b  w_b * ||p^R_b - p^S_b||^22

With uniform weights w_b = 1/K, this is exactly the squared L2 distance
between the stacked fingerprint vectors, equipped with a permutation null
distribution for false-positive-rate control.
"""

from __future__ import annotations

import numpy as np


def _block_slices(block_sizes) -> list[slice]:
    """Build cumulative slices into a flat D-vector for each block."""
    slices = []
    start = 0
    for size in block_sizes:
        slices.append(slice(start, start + size))
        start += size
    return slices


def _samples_to_probs(samples: np.ndarray, slices: list[slice]) -> np.ndarray:
    """Average (N, D) one-hot rows into a flat D-vector of per-block probabilities.

    Each block's entries are a probability distribution (sum to 1 if any row
    has a 1 in that block; 0 if the block was all-zero across samples, e.g.
    when a template had no samples collected).
    """
    if samples.shape[0] == 0:
        return np.zeros(samples.shape[1], dtype=float)
    # Uniform average: each row is one observation; block-wise this gives p_b.
    return samples.mean(axis=0)


def delta_kernel_mmd2_blockwise(
    p_ref: np.ndarray,
    p_sus: np.ndarray,
    block_sizes,
) -> np.ndarray:
    """Per-block MMD^2 under delta kernel.

    Args:
        p_ref, p_sus: flat (D,) probability vectors, D = sum(block_sizes).
        block_sizes: sequence of per-block cardinalities.

    Returns:
        (K,) array of per-block MMD^2 = ||p^R_b - p^S_b||^22.
    """
    slices = _block_slices(block_sizes)
    diffs = p_ref - p_sus
    return np.array([float(np.dot(diffs[s], diffs[s])) for s in slices])


def delta_kernel_mmd2(
    p_ref: np.ndarray,
    p_sus: np.ndarray,
    block_sizes,
    weights: np.ndarray | None = None,
) -> float:
    """Aggregate block-diagonal MMD^2. Uniform weights when `weights` is None.

    Args:
        weights: (K,) nonnegative weights. Uniform 1/K if None.
    """
    per_block = delta_kernel_mmd2_blockwise(p_ref, p_sus, block_sizes)
    if weights is None:
        weights = np.full(per_block.shape, 1.0 / len(per_block))
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != per_block.shape:
            raise ValueError(
                f"weights shape {weights.shape} != n_blocks {per_block.shape}"
            )
    return float(np.dot(weights, per_block))


def permutation_null(
    samples_ref: np.ndarray,
    samples_sus: np.ndarray,
    block_sizes,
    B: int = 1000,
    seed: int = 42,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Permutation null distribution of MMD^2_total under H0: same policy.

    Pool the reference and suspect one-hot rows, shuffle, resplit preserving
    the original sample sizes, and recompute MMD^2_total. Repeats B times.

    Args:
        samples_ref: (N_R, D) one-hot rows.
        samples_sus: (N_S, D) one-hot rows.
        block_sizes: per-block cardinalities summing to D.
        B: number of permutations.
        seed: RNG seed.
        weights: per-block weights (uniform if None).

    Returns:
        (B,) array of MMD^2_total values under permutation.
    """
    pooled = np.concatenate([samples_ref, samples_sus], axis=0)
    n_ref = samples_ref.shape[0]
    n_total = pooled.shape[0]
    rng = np.random.default_rng(seed)

    null = np.empty(B, dtype=float)
    for b in range(B):
        perm = rng.permutation(n_total)
        r_idx = perm[:n_ref]
        s_idx = perm[n_ref:]
        p_r = _samples_to_probs(pooled[r_idx], _block_slices(block_sizes))
        p_s = _samples_to_probs(pooled[s_idx], _block_slices(block_sizes))
        null[b] = delta_kernel_mmd2(p_r, p_s, block_sizes, weights)
    return null


def simulation_null(
    sampler_ref,
    n_sus: int,
    block_sizes,
    B: int = 1000,
    seed: int = 42,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Simulation null via fresh draws from a known-reference sampler.

    Use this when the reference model can be queried locally (open-weight or
    trusted vendor API), so we can draw two independent batches from it and
    measure MMD^2 between them. More powerful than the permutation null when
    reference access is cheap; falls back to permutation for black-box.

    Args:
        sampler_ref: callable `n -> (n, D) one-hot rows` drawing fresh samples
                     from the reference distribution.
        n_sus: sample size of the suspect batch (match to equalize power).
        block_sizes: per-block cardinalities.
        B: number of null draws.
        seed: RNG seed (passed through to sampler if it accepts it).
        weights: per-block weights (uniform if None).

    Returns:
        (B,) array of MMD^2_total.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(B, dtype=float)
    for b in range(B):
        # Independent halves: a pseudo-reference and a pseudo-suspect, both
        # from the same distribution. MMD^2 between them = null realization.
        a = sampler_ref(n_sus)
        s = sampler_ref(n_sus)
        p_a = _samples_to_probs(a, _block_slices(block_sizes))
        p_s = _samples_to_probs(s, _block_slices(block_sizes))
        null[b] = delta_kernel_mmd2(p_a, p_s, block_sizes, weights)
        # touch rng so the B index is reproducible even when sampler is seeded
        rng.integers(0, 2**31 - 1)
    return null


def decision(
    observed_mmd2: float,
    null_dist: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Reject H0 if observed exceeds the (1-alpha) quantile of the null.

    Returns:
        dict with keys `reject`, `pvalue`, `crit`. `pvalue` is the right-tail
        fraction of null values >= observed (exact permutation p-value).
    """
    null_dist = np.asarray(null_dist)
    crit = float(np.quantile(null_dist, 1.0 - alpha))
    pvalue = float((null_dist >= observed_mmd2).mean())
    return {
        "reject": bool(observed_mmd2 > crit),
        "pvalue": pvalue,
        "crit": crit,
    }


def samples_to_probs(samples: np.ndarray, block_sizes) -> np.ndarray:
    """Public helper: (N, D) one-hot rows -> (D,) probability vector.

    Useful for computing the observed MMD^2 outside of the null calibration
    loop. Block-wise probabilities sum to 1 within each block (or 0 for
    empty blocks).
    """
    return _samples_to_probs(samples, _block_slices(block_sizes))
