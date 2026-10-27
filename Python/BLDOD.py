"""Boundary-aware Local Density-based Outlier Detection (BLDOD): a NumPy port of BLDOD.m.

F. Aydin, "Boundary-aware local density-based outlier detection,"
Information Sciences, 647, 119520, 2023.
https://doi.org/10.1016/j.ins.2023.119520

For a scikit-learn compatible version, see
https://github.com/fatihaydin1/scikit-reduce
"""

import numpy as np


def BLDOD(X, lambda_p=1.0):
    """Return the outlier factor of each point, in [0, 1] (higher = more abnormal)."""
    X = np.asarray(X, dtype=float)

    # Compute the number of points
    m = X.shape[0]

    # Sample standard deviation of each column, ignoring NaN (MATLAB: std(X, 'omitnan'))
    stdDev = np.nanstd(X, axis=0, ddof=1)

    N = np.zeros(m)

    # Multiply the norm of the standard deviation of the points with lambda_p
    deviation = lambda_p * np.linalg.norm(stdDev)

    for i in range(m):
        # Move the considered point to the center (0, 0)
        D = X - X[i, :]
        # The number of points whose Euclidean distance to the considered point
        # is at most the deviation length
        N[i] = np.sum(np.sqrt(np.sum(D**2, axis=1)) <= deviation)

    # Normalize the values between 0-1 (MATLAB: 1 - normalize(N, 'range'))
    spread = N.max() - N.min()
    if spread == 0:
        return np.ones(m)
    return 1 - (N - N.min()) / spread
