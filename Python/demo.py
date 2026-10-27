# DEMO for BLDOD
# Run from the Python folder: python demo.py

import numpy as np
import scipy.io
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from BLDOD import BLDOD

# Load a data set from the "outlier datasets" folder (variables X and y)
mat = scipy.io.loadmat('../outlier datasets/arrhythmia.mat')
X, y = mat['X'], np.ravel(mat['y'])

# Obtain the outlier factors
score = BLDOD(X, 1)

# Area under the ROC curve and average precision (area under the precision-recall curve)
print('ROC AUC:', roc_auc_score(y, score))
print('Average precision:', average_precision_score(y, score))

# F-measure when points whose factor is above 0.9 are labelled outliers
print('F-measure:', f1_score(y, score > 0.9))
