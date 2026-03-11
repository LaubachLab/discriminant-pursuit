# discriminant-pursuit

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**Discriminant Pursuit via Wavelet Packets for Time-Series Feature Extraction**

A Python port of the discriminant pursuit algorithm originally developed by Jonathan Buckheit and David Donoho at Stanford University in 1995. Discriminant pursuit finds wavelet packet basis functions that maximize the separation between class means, producing an explicitly interpretable set of discriminative features with known time-frequency localization.

## Motivation

Modern time-series classifiers achieve high accuracy but often function as opaque models. Classic methods from applied harmonic analysis, such as discriminant pursuit, solve the interpretability problem directly by selecting basis functions from a structured dictionary that best separate the classes. However, the original Matlab implementations depend on the Wavelab toolbox and are effectively inaccessible to most current researchers. This package rescues the methodology by providing a fast, NumPy-based implementation with full scikit-learn compatibility.

## How It Works

Discriminant pursuit operates on an overcomplete wavelet packet dictionary. For a signal of length *n* = 2^*J*, the wavelet packet decomposition at depth *D* produces a table of *n* × (*D*+1) coefficients, organized by scale and frequency. The algorithm:

1. Computes the trimmed mean signal for each class
2. Forms all pairwise contrasts between class means
3. Decomposes each contrast into the wavelet packet dictionary
4. Greedily selects the coefficient with the largest absolute amplitude across all contrasts — this identifies the basis function that best separates one pair of classes at this step
5. Deflates the dictionary to remove the selected component
6. Repeats until the desired number of basis functions is extracted

The result is a sparse set of time-frequency atoms, each with known temporal support and frequency content, that collectively capture the discriminative structure between classes.

## Installation

```bash
pip install git+https://github.com/LaubachLab/discriminant-pursuit.git
```

Or for development:

```bash
git clone https://github.com/LaubachLab/discriminant-pursuit.git
cd discriminant-pursuit
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `scikit-learn`. Visualization requires `matplotlib`.

## Quick Start

### Scikit-learn pipeline

```python
from discr_pursuit import DiscriminantPursuit
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

# Extract 10 discriminant basis functions using Symmlet-8 wavelets
dp = DiscriminantPursuit(n_coef=10, filter_family='Symmlet', filter_par=8)

# Build a classification pipeline
clf = make_pipeline(dp, RidgeClassifierCV())
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Inspect the extracted basis functions
basis = dp.basis_functions_  # shape (n_coef, n_timepoints)
```

### Functional interface

```python
from discr_pursuit import discriminant_pursuit, make_on_filter

qmf = make_on_filter('Symmlet', 8)
results = discriminant_pursuit(
    n_coef=10,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    qmf=qmf,
)

# Results contain:
#   train_coefs, test_coefs — projections onto the selected basis
#   basis_functions — time-domain waveforms for each basis element
#   amplitudes — discriminative amplitude of each basis function
#   packet_indices — (depth, block, position) in the wavelet packet tree
```

### Wavelet packet tools

```python
from discr_pursuit import wp_analysis, make_wp, make_on_filter

# Decompose a signal into wavelet packets
qmf = make_on_filter('Daubechies', 8)
wp = wp_analysis(signal, D=5, qmf=qmf)

# Construct a specific basis function
basis_func = make_wp(d=3, b=2, k=5, qmf=qmf, n=64)
```

## Repository Structure

```text
discriminant-pursuit/
├── discr_pursuit.py              # Core package
├── test_discr_pursuit.py         # Validation tests (18 tests)
└── examples/
    ├── demo_waveform.ipynb       # Waveform-5000 analysis with Ridge classifier
    └── demo_rf_cv.ipynb          # Cross-validation with RandomForest and
                                  #   feature importance (Gini and permutation)
```

## Example Notebooks

**Waveform-5000 analysis** (`demo_waveform.ipynb`) demonstrates the complete discriminant pursuit workflow on the Breiman waveform benchmark: feature extraction, Ridge classification, basis function visualization, temporal coverage, feature distributions, and accuracy as a function of the number of basis functions.

**Cross-validation with feature importance** (`demo_rf_cv.ipynb`) demonstrates repeated stratified k-fold cross-validation using a `DiscriminantPursuit` + `RandomForestClassifier` pipeline. Both the feature extractor and classifier are fitted within each fold, preventing data leakage. Feature importance is estimated two ways:

- **Gini (MDI) importance** — the default sklearn measure, reported for reference. Gini importance is biased toward features with high cardinality or correlated predictors (Strobl et al., 2007) and should not be used for scientific interpretation. It is included in the demo for comparison purposes only.
- **Permutation importance** — shuffles each feature and measures the drop in accuracy on a held-out set. This is the recommended approach for interpretation (Strobl et al., 2007).

## Available Wavelet Filters

| Family | Parameters | Description |
|--------|-----------|-------------|
| Haar | — | Simplest wavelet, discontinuous |
| Daubechies | 4, 6, 8, 10, 12, 14, 16, 18, 20 | Maximally smooth scaling function |
| Symmlet | 4, 5, 6, 7, 8, 9, 10 | Least asymmetric, compactly supported |
| Coiflet | 1, 2, 3, 4, 5 | Vanishing moments for both mother and father |

## API Reference

### Class

**`DiscriminantPursuit(n_coef=10, filter_family='Symmlet', filter_par=8, trim_percent=10, verbose=False)`**

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Find the most discriminative wavelet packet basis functions. |
| `transform(X)` | Project time series onto the fitted basis functions. |
| `fit_transform(X, y)` | Fit and transform in one call. |

| Attribute | Description |
|-----------|-------------|
| `basis_functions_` | ndarray, shape (n_coef, n_timepoints). Time-domain basis functions. |
| `amplitudes_` | ndarray, shape (n_coef,). Discriminative amplitude of each basis function. |
| `packet_indices_` | list of (d, b, k) tuples. Wavelet packet tree coordinates. |
| `coef_indices_` | ndarray. Linear indices into the wavelet packet table. |
| `qmf_` | ndarray. The fitted quadrature mirror filter. |

### Module-Level Functions

| Function | Description |
|--------|-------------|
| `discriminant_pursuit(n_coef, X_train, y_train, X_test, y_test, qmf)` | Core algorithm (functional interface). |
| `make_on_filter(family, par)` | Generate orthonormal QMF filter. |
| `wp_analysis(x, D, qmf)` | Full wavelet packet decomposition. |
| `wp_impulse(wp, d, b, k, qmf)` | Impulse response of a single basis element. |
| `make_wp(d, b, k, qmf, n)` | Construct a time-domain basis function. |

## Testing

```bash
python test_discr_pursuit.py              # standalone
python -m pytest test_discr_pursuit.py -v # via pytest
```

The test suite (18 tests) validates:
- Filter generation and normalization for all 22 supported filter variants
- Perfect reconstruction through dyadic operators (Haar, Daubechies, Symmlet)
- Wavelet packet table shape and energy preservation (Parseval's theorem)
- Impulse response structure and unit energy
- Packet table indexing roundtrip
- Discriminant pursuit output structure, amplitude ordering, and classification accuracy
- Scikit-learn wrapper fit/transform consistency and pipeline compatibility
- get_params/set_params for clone() and GridSearchCV
- Basis function orthogonality
- Multiple filter families and binary classification
- RandomForest pipeline integration

No external data downloads are required for testing.

## References

- Buckheit, J. & Donoho, D.L. (1995). Improved linear discrimination using time-frequency dictionaries. *Proc. SPIE*, 2569, 540–551.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25.

## Acknowledgements

Original Matlab code by Jonathan Buckheit and David Donoho, Stanford University Department of Statistics (1995), as part of the Wavelab toolbox. The discriminant pursuit code was obtained from J. Buckheit in 1995 and modified and distributed with his permission by M. Laubach.

## Author

Mark Laubach (American University, Department of Neuroscience). Python port developed with Claude (Anthropic) as AI coding assistant.

## License

BSD-3-Clause
