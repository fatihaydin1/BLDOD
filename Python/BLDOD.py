import numpy as np
from numpy import linalg as LA

# Boundary-aware Local density-based Outlier Detection (BLDOD)
def BLDOD(X, lambda_p):
  # Compute the number of points
  m = X.shape[0];

  # Find the standard deviation of each column of the data set (X)
  stdDev = X.std(1);

  N = np.zeros(m);

  # Multiply the norm of the standard deviation of the points with lambda_p
  deviation = lambda_p * LA.norm(stdDev, 2);

  for i in range(m):
    # Move the considered point to the center (0, 0)
    D = X - X[i, :];

    # Calculate the number of points that are as far away as the deviation length value from the considered point
    Dsq = D*D;
    Dsq_len = np.sqrt(Dsq).sum(axis=1);
    N[i] = (Dsq_len <= deviation).sum();

  # Normalize the values between 0-1
  N = (N - np.min(N))/(np.max(N)-np.min(N));
  return 1 - N;