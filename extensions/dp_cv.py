"""
dp_cv.py — Cross-Validation and Feature Selection for Discriminant Pursuit

Provides:
    - estimate_n_coef: Scree-plot estimation of the number of discriminant
      basis functions (two methods: linear tail fit and Kneedle algorithm)
    - single_split: Single train/test split with DP + classifier
    - cross_validate_dp: Full repeated stratified k-fold CV with DP + classifier
    - compute_importance: Gini and permutation feature importance

All functions are designed to work with multi-channel data by treating
each channel independently and concatenating features across channels.

License: BSD-3-Clause
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance


# ============================================================================
# SCREE PLOT: ESTIMATE NUMBER OF DISCRIMINANT FEATURES
# ============================================================================

def estimate_n_coef(X_train, y_train, qmf, max_coef=None,
                    method='linear_tail', trim_percent=10,
                    verbose=True):
    """
    Estimate the number of discriminant basis functions using a scree plot.

    Runs discriminant pursuit to full depth (or max_coef), then estimates
    the breakpoint where amplitudes transition from signal to noise.

    Parameters
    ----------
    X_train : ndarray, shape (n_samples, n_timepoints)
        Training data. n_timepoints must be a power of 2.
    y_train : array-like
        Class labels.
    qmf : ndarray
        Quadrature mirror filter.
    max_coef : int, optional
        Maximum number of coefficients to extract. Default: n_timepoints.
    method : str
        'linear_tail' — Fit a line to the second half of the amplitude
            curve and find where the first half exceeds the intercept.
        'kneedle' — Find the point of maximum curvature (Satopaa et al.,
            2011, "Finding a Kneedle in a Haystack").
        'both' — Return results from both methods.
    trim_percent : float
        Passed to discriminant_pursuit.
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'n_coef_estimated' : int
            Estimated number of features (from the primary method).
        'amplitudes' : ndarray
            Full amplitude sequence.
        'method' : str
            Method used.
        'linear_tail' : dict (if method is 'linear_tail' or 'both')
            'n_coef': int, 'intercept': float, 'slope': float
        'kneedle' : dict (if method is 'kneedle' or 'both')
            'n_coef': int, 'knee_index': int
    """
    from discr_pursuit import discriminant_pursuit

    n = X_train.shape[1]
    if max_coef is None:
        max_coef = n

    if verbose:
        print(f"Running full DP extraction ({max_coef} coefficients)...")

    dp = discriminant_pursuit(
        n_coef=max_coef,
        X_train=X_train, y_train=y_train,
        X_test=X_train[:1], y_test=y_train[:1],
        qmf=qmf,
        trim_percent=trim_percent,
        verbose=False,
    )

    amplitudes = dp['amplitudes']
    n_total = len(amplitudes)

    results = {
        'amplitudes': amplitudes,
        'method': method,
    }

    # Method 1: Linear tail fit
    if method in ('linear_tail', 'both'):
        half = n_total // 2
        tail_x = np.arange(half, n_total)
        tail_y = amplitudes[half:]

        # Fit line to tail
        coeffs = np.polyfit(tail_x, tail_y, 1)
        slope, intercept = coeffs

        # Count features above intercept
        threshold = intercept
        n_above = int(np.sum(amplitudes > threshold))

        results['linear_tail'] = {
            'n_coef': n_above,
            'intercept': intercept,
            'slope': slope,
            'threshold': threshold,
        }

        if verbose:
            print(f"  Linear tail: threshold={threshold:.4f}, "
                  f"n_coef={n_above}")

    # Method 2: Kneedle
    if method in ('kneedle', 'both'):
        knee_idx = _kneedle(amplitudes)

        results['kneedle'] = {
            'n_coef': knee_idx + 1,
            'knee_index': knee_idx,
        }

        if verbose:
            print(f"  Kneedle: knee at index {knee_idx}, "
                  f"n_coef={knee_idx + 1}")

    # Set primary estimate
    if method == 'linear_tail':
        results['n_coef_estimated'] = results['linear_tail']['n_coef']
    elif method == 'kneedle':
        results['n_coef_estimated'] = results['kneedle']['n_coef']
    else:
        results['n_coef_estimated'] = results['linear_tail']['n_coef']

    return results


def _kneedle(y):
    """
    Find the knee point in a decreasing curve using the Kneedle algorithm.

    Normalizes the curve to [0,1] x [0,1], computes the difference from
    the diagonal, and returns the index of maximum difference.

    Parameters
    ----------
    y : array-like
        Decreasing sequence (e.g., amplitudes from discriminant pursuit).

    Returns
    -------
    knee_idx : int
        Index of the knee point.

    References
    ----------
    Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011).
    Finding a "Kneedle" in a Haystack: Detecting Knee Points in
    System Behavior. ICDCS Workshops.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(n, dtype=float)

    # Normalize to [0, 1]
    x_norm = (x - x[0]) / (x[-1] - x[0] + 1e-15)
    y_norm = (y - y[-1]) / (y[0] - y[-1] + 1e-15)

    # Difference from the diagonal (line from first to last point)
    diff = y_norm - (1 - x_norm)

    knee_idx = int(np.argmax(np.abs(diff)))
    return knee_idx


def plot_scree(results, figsize=(10, 4), title=None):
    """
    Plot the scree plot with estimated breakpoints.

    Parameters
    ----------
    results : dict
        Output from estimate_n_coef().
    figsize : tuple
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    amplitudes = results['amplitudes']
    n = len(amplitudes)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, amplitudes, 'o-', color='steelblue', markersize=3,
            linewidth=1, label='Amplitudes')

    if 'linear_tail' in results:
        lt = results['linear_tail']
        threshold = lt['threshold']
        ax.axhline(threshold, color='tomato', linestyle='--', linewidth=1.2,
                    label=f'Linear tail threshold ({lt["n_coef"]} features)')
        ax.axvline(lt['n_coef'] + 0.5, color='tomato', linewidth=0.8,
                    alpha=0.5)

        # Plot the fitted line on the tail
        half = n // 2
        tail_x = np.arange(half, n)
        tail_line = lt['slope'] * tail_x + lt['intercept']
        ax.plot(tail_x + 1, tail_line, ':', color='tomato', linewidth=1)

    if 'kneedle' in results:
        kn = results['kneedle']
        ax.axvline(kn['knee_index'] + 1, color='forestgreen',
                    linestyle='-.', linewidth=1.2,
                    label=f'Kneedle ({kn["n_coef"]} features)')

    ax.set_xlabel('Basis function index')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    if title is None:
        title = 'Discriminant Pursuit Scree Plot'
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ============================================================================
# SINGLE SPLIT ANALYSIS
# ============================================================================

def single_split(X, y, n_coef, qmf=None, dp_kwargs=None,
                 classifier=None, test_size=0.3, random_state=42,
                 return_dp=False):
    """
    Single train/test split with DP feature extraction and classification.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
        Time series data. For multi-channel, features are concatenated.
    y : array-like
        Class labels.
    n_coef : int
        Number of DP basis functions per channel.
    qmf : ndarray, optional
        Quadrature mirror filter. If None, uses Symmlet-8.
    dp_kwargs : dict, optional
        Additional kwargs for discriminant_pursuit.
    classifier : sklearn classifier, optional
        Default: RidgeClassifierCV.
    test_size : float
    random_state : int
    return_dp : bool
        If True, return DP results for each channel.

    Returns
    -------
    results : dict with keys:
        'accuracy', 'balanced_accuracy', 'y_test', 'y_pred',
        'train_features', 'test_features', 'classifier'
        Optionally: 'dp_results' (list of DP results per channel)
    """
    from discr_pursuit import discriminant_pursuit, make_on_filter

    if qmf is None:
        qmf = make_on_filter('Symmlet', 8)
    if dp_kwargs is None:
        dp_kwargs = {}
    if classifier is None:
        classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # Handle multi-channel
    if X.ndim == 3:
        n_channels = X.shape[1]
        multi = True
    else:
        X = X[:, np.newaxis, :]
        n_channels = 1
        multi = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    all_train_coefs = []
    all_test_coefs = []
    dp_list = []

    for ch in range(n_channels):
        dp = discriminant_pursuit(
            n_coef=n_coef,
            X_train=X_train[:, ch, :],
            y_train=y_train,
            X_test=X_test[:, ch, :],
            y_test=y_test,
            qmf=qmf,
            verbose=False,
            **dp_kwargs,
        )
        all_train_coefs.append(dp['train_coefs'])
        all_test_coefs.append(dp['test_coefs'])
        dp_list.append(dp)

    train_features = np.hstack(all_train_coefs)
    test_features = np.hstack(all_test_coefs)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    clf = clone(classifier)
    clf.fit(train_scaled, y_train)
    y_pred = clf.predict(test_scaled)

    out = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'train_features': train_features,
        'test_features': test_features,
        'classifier': clf,
    }
    if return_dp:
        out['dp_results'] = dp_list

    return out


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validate_dp(X, y, n_coef, qmf=None, dp_kwargs=None,
                      classifier=None, n_splits=5, n_repeats=5,
                      random_state=42, verbose=True):
    """
    Repeated stratified k-fold CV with DP feature extraction and classification.

    Both DP and the classifier are fitted within each fold, preventing
    data leakage.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
    y : array-like
    n_coef : int
        Number of DP basis functions per channel.
    qmf : ndarray, optional
        Default: Symmlet-8.
    dp_kwargs : dict, optional
    classifier : sklearn classifier, optional
        Default: RandomForestClassifier(n_estimators=200).
    n_splits : int
    n_repeats : int
    random_state : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'fold_accuracies' : ndarray
        'fold_balanced_accuracies' : ndarray
        'mean_accuracy', 'std_accuracy'
        'mean_balanced_accuracy', 'std_balanced_accuracy'
    """
    from discr_pursuit import discriminant_pursuit, make_on_filter

    if qmf is None:
        qmf = make_on_filter('Symmlet', 8)
    if dp_kwargs is None:
        dp_kwargs = {}
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=200,
                                            random_state=random_state,
                                            n_jobs=-1)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim == 3:
        n_channels = X.shape[1]
    else:
        X = X[:, np.newaxis, :]
        n_channels = 1

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=random_state)

    fold_accs = []
    fold_bal_accs = []
    n_total = n_splits * n_repeats

    if verbose:
        print(f"Running {n_repeats}x{n_splits}-fold CV "
              f"({n_total} folds, {n_channels} channel(s), "
              f"{n_coef} features/channel)...")

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X[:, 0, :], y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        all_train_coefs = []
        all_test_coefs = []

        for ch in range(n_channels):
            dp = discriminant_pursuit(
                n_coef=n_coef,
                X_train=X_tr[:, ch, :],
                y_train=y_tr,
                X_test=X_te[:, ch, :],
                y_test=y_te,
                qmf=qmf,
                verbose=False,
                **dp_kwargs,
            )
            all_train_coefs.append(dp['train_coefs'])
            all_test_coefs.append(dp['test_coefs'])

        train_feat = np.hstack(all_train_coefs)
        test_feat = np.hstack(all_test_coefs)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_feat)
        test_scaled = scaler.transform(test_feat)

        clf = clone(classifier)
        clf.fit(train_scaled, y_tr)
        y_pred = clf.predict(test_scaled)

        acc = accuracy_score(y_te, y_pred)
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        fold_accs.append(acc)
        fold_bal_accs.append(bal_acc)

        if verbose and fold_idx % n_splits == 0:
            round_num = fold_idx // n_splits
            round_accs = fold_accs[-n_splits:]
            print(f"  Round {round_num}/{n_repeats} — "
                  f"mean acc: {np.mean(round_accs):.3f}  "
                  f"std: {np.std(round_accs):.3f}")

    fold_accs = np.array(fold_accs)
    fold_bal_accs = np.array(fold_bal_accs)

    out = {
        'fold_accuracies': fold_accs,
        'fold_balanced_accuracies': fold_bal_accs,
        'mean_accuracy': fold_accs.mean(),
        'std_accuracy': fold_accs.std(),
        'mean_balanced_accuracy': fold_bal_accs.mean(),
        'std_balanced_accuracy': fold_bal_accs.std(),
    }

    if verbose:
        print(f"\nOverall: acc = {out['mean_accuracy']:.3f} "
              f"± {out['std_accuracy']:.3f}")

    return out
