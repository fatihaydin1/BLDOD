# DEMO for BLDOD

import scipy.io
from sklearn.metrics import roc_auc_score

# Load MATLAB mat files
mat = scipy.io.loadmat('arrhythmia.mat');

# Obtain local factors
score = BLDOD(mat['X'], 1);

# Calculate the ROC curve (receiver operating characteristic curve)
rocarea = roc_auc_score(mat['y'], score);
print(rocarea);