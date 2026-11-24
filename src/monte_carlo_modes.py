import numpy as np

N_MODES = 17
def select_modes(rng, nmodes, ntrials, p_detect=0.5, min_modes=3):
    """Random sparse detection; returns sorted array of detected mode numbers."""
    p = np.broadcast_to(p_detect, (ntrials,nmodes))      # scalar or per-mode probabilities
    while True:
        mask = rng.random((ntrials, nmodes)) < p
        if mask.sum() >= min_modes:
            return [np.flatnonzero(m) for m in mask]


def split_clusters(modes):
    """Split a sorted array of modes into runs of consecutive integers."""
    breaks = [np.flatnonzero(np.diff(m) > 1) + 1 for m in modes]
    return [np.split(m, b) for m,b in zip(modes, breaks)]


def _mislabel_clusters(clusters, rng, max_shift=None, min_gap=1,
                      force_mislabel=False, n_modes=N_MODES, max_tries=1000):
    """
    Independently shift each cluster (positive or negative) subject to:
      - labels within [0, n_modes-1]
      - label order preserved (known arrival order)
      - at least `min_gap` unlabeled modes between consecutive clusters
      - |shift| <= max_shift (if given)
    Returns list of label arrays, one per cluster.
    """
    n = len(clusters)
    start = np.array([c[0] for c in clusters])
    length = np.array([len(c) for c in clusters])
    true_gaps = start[1:] - (start[:-1] + length[:-1])      # missed modes between clusters
    if np.any(true_gaps < min_gap):
        raise ValueError("min_gap larger than a true inter-cluster gap; lower min_gap")
    ms = n_modes if max_shift is None else max_shift

    # Backward pass: latest feasible start for each cluster, accounting for
    # the clusters that still have to fit after it.
    U = np.empty(n, dtype=int)
    for i in range(n - 1, -1, -1):
        ub = min(start[i] + ms, n_modes - length[i])         # last label <= n_modes-1
        if i < n - 1:
            ub = min(ub, U[i + 1] - length[i] - min_gap)
        U[i] = ub

    for _ in range(max_tries):
        labels, prev_end = [], None
        for i in range(n):
            lo = max(start[i] - ms, 0)
            if prev_end is not None:
                lo = max(lo, prev_end + 1 + min_gap)
            s = rng.integers(lo, U[i] + 1)                   # U[i] >= lo by construction
            labels.append(np.arange(s, s + length[i]))
            prev_end = s + length[i] - 1
        identical = all(np.array_equal(l, c) for l, c in zip(labels, clusters))
        if not (force_mislabel and identical):
            return np.block(labels)
    raise RuntimeError("Could not draw a non-identity labeling; relax constraints")


def mislabel_clusters_mc(clusters, rng, max_shift=None, min_gap=1,
                      force_mislabel=False, n_modes=N_MODES, max_tries=1000):


    return [_mislabel_clusters(c, rng, max_shift=None, min_gap=1,
                              force_mislabel=False, n_modes=N_MODES, max_tries=1000)
                for c in clusters]


def make_simulation_labels(rng, ntrials, nmodes=N_MODES,)
