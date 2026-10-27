# Boundary-aware Local density-based Outlier Detection (BLDOD)
https://doi.org/10.1016/j.ins.2023.119520

- `MATLAB/BLDOD.m`: the MATLAB implementation used in the paper.
- `Python/BLDOD.py`: a standalone NumPy port that returns the same scores as `BLDOD.m`.
- `outlier datasets/`: the data sets used in the experiments.

## scikit-learn compatible version

BLDOD is also available in [scikit-reduce](https://github.com/fatihaydin1/scikit-reduce), with the scikit-learn outlier detection interface (`fit`, `score_samples`, `decision_function`, `predict`):

```bash
pip install scikit-reduce
```

```python
from skreduce import BLDOD
detector = BLDOD(std_coef=1.0).fit(X)
detector.outlier_factor_   # higher = more abnormal
```

## Note on earlier versions of BLDOD.py

Until October 2026, `Python/BLDOD.py` computed the standard deviation of each row instead of each column and used the sum of absolute differences instead of the Euclidean distance, so its results differed from `BLDOD.m` and the paper. The current version gives the same results as `BLDOD.m`.
