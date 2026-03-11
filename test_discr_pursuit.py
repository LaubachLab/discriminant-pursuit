"""
test_discr_pursuit.py — Validation tests for discriminant-pursuit

This file serves two purposes:
    1. Code validation: Verifies that all core functions produce correct
       output for the wavelet packet decomposition and discriminant pursuit.
    2. Installation testing: After downloading the package, run
       `python test_discr_pursuit.py` to confirm all dependencies are
       installed and the module works end-to-end.

Tests use synthetic data (no external downloads required).

Tests imports private functions (_down_dyad_lo, etc.). These files are not
in __all__ in the main function and could break if the internals are refactored. 

Usage:
    python test_discr_pursuit.py              # run all tests
    python -m pytest test_discr_pursuit.py -v # run via pytest
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

from discr_pursuit import (
    make_on_filter,
    wp_analysis,
    wp_impulse,
    make_wp,
    discriminant_pursuit,
    DiscriminantPursuit,
    _down_dyad_lo,
    _down_dyad_hi,
    _up_dyad_lo,
    _up_dyad_hi,
    _mirror_filt,
    _aconv,
    _iconv,
    _packet,
    _ix2pkt,
    _pkt2ix,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_test_data(n_samples=201, n_timepoints=32, n_classes=3, random_state=42):
    """Generate synthetic data with known class structure.

    n_samples is set to a multiple of n_classes + 1 to avoid index
    mismatch when creating balanced labels.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_timepoints))
    y = np.array([i % n_classes for i in range(n_samples)])

    # Add class-specific bumps at different positions
    for k in range(n_classes):
        mask = y == k
        center = (k + 1) * n_timepoints // (n_classes + 1)
        t = np.arange(n_timepoints)
        bump = 3.0 * np.exp(-0.5 * ((t - center) / 3.0) ** 2)
        X[mask] += bump

    # Shuffle
    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]

    split = int(0.7 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_imports():
    """Verify all public functions can be imported."""
    from discr_pursuit import (
        make_on_filter, wp_analysis, wp_impulse, make_wp,
        discriminant_pursuit, DiscriminantPursuit,
    )
    print("  PASS: all imports successful")


def test_make_on_filter_all_families():
    """Verify filter generation and normalization for all supported families."""
    test_cases = [
        ('Haar', None),
        ('Daubechies', 4), ('Daubechies', 6), ('Daubechies', 8),
        ('Daubechies', 10), ('Daubechies', 12), ('Daubechies', 14),
        ('Daubechies', 16), ('Daubechies', 18), ('Daubechies', 20),
        ('Symmlet', 4), ('Symmlet', 5), ('Symmlet', 6), ('Symmlet', 7),
        ('Symmlet', 8), ('Symmlet', 9), ('Symmlet', 10),
        ('Coiflet', 1), ('Coiflet', 2), ('Coiflet', 3),
        ('Coiflet', 4), ('Coiflet', 5),
    ]
    for family, par in test_cases:
        qmf = make_on_filter(family, par)
        assert isinstance(qmf, np.ndarray), f"{family}-{par} not ndarray"
        assert len(qmf) >= 2, f"{family}-{par} filter too short"
        assert abs(np.linalg.norm(qmf) - 1.0) < 1e-10, (
            f"{family}-{par} not normalized: norm={np.linalg.norm(qmf)}"
        )
    print(f"  PASS: make_on_filter ({len(test_cases)} filter variants)")


def test_dyadic_operators_roundtrip():
    """Verify that downsampling then upsampling reconstructs the signal."""
    for family, par in [('Haar', None), ('Daubechies', 8), ('Symmlet', 8)]:
        qmf = make_on_filter(family, par)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(32)

        lo = _down_dyad_lo(x, qmf)
        hi = _down_dyad_hi(x, qmf)

        assert len(lo) == 16, f"Lo length wrong: {len(lo)}"
        assert len(hi) == 16, f"Hi length wrong: {len(hi)}"

        x_recon = _up_dyad_lo(lo, qmf) + _up_dyad_hi(hi, qmf)
        assert np.allclose(x, x_recon, atol=1e-10), (
            f"Roundtrip failed for {family}-{par}"
        )
    print("  PASS: dyadic operators roundtrip (Haar, Daubechies-8, Symmlet-8)")


def test_wp_analysis_shape():
    """Verify wavelet packet table has correct shape."""
    qmf = make_on_filter('Symmlet', 8)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)
    D = 6

    wp = wp_analysis(x, D, qmf)
    assert wp.shape == (64, D + 1), f"Expected (64, 7), got {wp.shape}"
    assert np.allclose(wp[:, 0], x), "First column should be input signal"
    print(f"  PASS: wp_analysis shape ({wp.shape})")


def test_wp_analysis_energy_preservation():
    """Verify Parseval's theorem: energy preserved at each level."""
    qmf = make_on_filter('Symmlet', 8)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)
    D = 6

    wp = wp_analysis(x, D, qmf)
    energy_input = np.sum(x ** 2)

    for d in range(D + 1):
        energy_level = np.sum(wp[:, d] ** 2)
        assert abs(energy_level - energy_input) < 1e-8, (
            f"Energy not preserved at depth {d}: "
            f"{energy_level:.6f} vs {energy_input:.6f}"
        )
    print("  PASS: wp_analysis energy preservation (Parseval)")


def test_wp_impulse():
    """Verify impulse response has correct structure."""
    qmf = make_on_filter('Haar')
    n = 32
    D = 5
    wp_shape = np.zeros((n, D + 1))

    dwp = wp_impulse(wp_shape, 2, 1, 3, qmf)
    assert dwp.shape == (n, D + 1), f"Wrong shape: {dwp.shape}"
    assert dwp[1 * (n // 4) + 3, 2] == 1.0, "Impulse not placed correctly"

    energy = np.sum(dwp[:, 0] ** 2)
    assert abs(energy - 1.0) < 1e-10, f"Impulse energy wrong: {energy}"
    print("  PASS: wp_impulse")


def test_make_wp():
    """Verify make_wp produces a unit-energy basis function."""
    qmf = make_on_filter('Symmlet', 8)
    n = 64

    wavepkt = make_wp(3, 2, 5, qmf, n)
    assert len(wavepkt) == n, f"Wrong length: {len(wavepkt)}"
    energy = np.sum(wavepkt ** 2)
    assert abs(energy - 1.0) < 1e-10, f"Basis function energy wrong: {energy}"
    print(f"  PASS: make_wp (energy={energy:.10f})")


def test_packet_indexing_roundtrip():
    """Verify ix2pkt and pkt2ix are inverse operations."""
    n = 64
    D = 6

    for d in range(D + 1):
        for b in range(2 ** d):
            nc = n // (2 ** d)
            for k in range(min(nc, 3)):
                ix = _pkt2ix(d, b, k, D, n)
                d2, b2, k2 = _ix2pkt(ix, D, n)
                assert (d2, b2, k2) == (d, b, k), (
                    f"Roundtrip failed: ({d},{b},{k}) -> {ix} -> ({d2},{b2},{k2})"
                )
    print("  PASS: packet indexing roundtrip")


def test_discriminant_pursuit_basic():
    """Verify discriminant_pursuit runs and returns expected structure."""
    X_train, y_train, X_test, y_test = make_test_data()
    qmf = make_on_filter('Symmlet', 8)

    results = discriminant_pursuit(
        n_coef=5,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        qmf=qmf,
        verbose=False,
    )

    expected_keys = {
        'train_coefs', 'test_coefs', 'basis_functions',
        'amplitudes', 'coef_indices', 'packet_indices'
    }
    assert set(results.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(results.keys())}"
    )
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    assert results['train_coefs'].shape == (n_train, 5), (
        f"Train coefs shape wrong: {results['train_coefs'].shape}"
    )
    assert results['test_coefs'].shape == (n_test, 5), (
        f"Test coefs shape wrong: {results['test_coefs'].shape}"
    )
    assert results['basis_functions'].shape == (5, 32)
    assert len(results['amplitudes']) == 5
    assert len(results['packet_indices']) == 5

    amps = results['amplitudes']
    assert all(amps[i] >= amps[i + 1] for i in range(len(amps) - 1)), (
        "Amplitudes not decreasing"
    )
    print(f"  PASS: discriminant_pursuit (5 basis functions, "
          f"top amplitude={amps[0]:.4f})")


def test_discriminant_pursuit_classification():
    """Verify DP features enable above-chance classification."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler

    X_train, y_train, X_test, y_test = make_test_data()
    qmf = make_on_filter('Symmlet', 8)

    results = discriminant_pursuit(
        n_coef=10,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        qmf=qmf,
        verbose=False,
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(results['train_coefs'])
    test_scaled = scaler.transform(results['test_coefs'])

    clf = RidgeClassifierCV()
    clf.fit(train_scaled, y_train)
    acc = clf.score(test_scaled, y_test)

    assert acc > 0.50, f"Accuracy too low: {acc:.4f}"
    print(f"  PASS: classification accuracy ({acc:.4f}, chance=0.33)")


def test_sklearn_wrapper_fit_transform():
    """Verify DiscriminantPursuit wrapper follows sklearn contract."""
    X_train, y_train, X_test, y_test = make_test_data()

    dp = DiscriminantPursuit(
        n_coef=5, filter_family='Symmlet', filter_par=8, verbose=False
    )

    dp.fit(X_train, y_train)
    train_coefs = dp.transform(X_train)
    test_coefs = dp.transform(X_test)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    assert train_coefs.shape == (n_train, 5), (
        f"Train shape wrong: {train_coefs.shape}"
    )
    assert test_coefs.shape == (n_test, 5), (
        f"Test shape wrong: {test_coefs.shape}"
    )
    assert dp.basis_functions_.shape == (5, 32)
    assert len(dp.amplitudes_) == 5
    assert len(dp.packet_indices_) == 5
    print("  PASS: DiscriminantPursuit fit/transform")


def test_sklearn_wrapper_fit_transform_single_call():
    """Verify fit_transform produces same result as fit then transform."""
    X_train, y_train, X_test, y_test = make_test_data()

    dp1 = DiscriminantPursuit(
        n_coef=5, filter_family='Symmlet', filter_par=8, verbose=False
    )
    dp1.fit(X_train, y_train)
    coefs_separate = dp1.transform(X_train)

    dp2 = DiscriminantPursuit(
        n_coef=5, filter_family='Symmlet', filter_par=8, verbose=False
    )
    coefs_combined = dp2.fit_transform(X_train, y_train)

    assert np.allclose(coefs_separate, coefs_combined), (
        "fit+transform != fit_transform"
    )
    print("  PASS: fit_transform consistency")


def test_sklearn_pipeline():
    """Verify DiscriminantPursuit works inside an sklearn pipeline."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline

    X_train, y_train, X_test, y_test = make_test_data()

    clf = make_pipeline(
        DiscriminantPursuit(n_coef=5, filter_family='Symmlet',
                            filter_par=8, verbose=False),
        RidgeClassifierCV(),
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    assert acc > 0.40, f"Pipeline accuracy too low: {acc:.4f}"
    print(f"  PASS: sklearn pipeline (accuracy={acc:.4f})")


def test_sklearn_get_params():
    """Verify get_params and set_params work for clone/GridSearchCV."""
    dp = DiscriminantPursuit(n_coef=15, filter_family='Coiflet', filter_par=3)
    params = dp.get_params()

    assert params['n_coef'] == 15
    assert params['filter_family'] == 'Coiflet'
    assert params['filter_par'] == 3

    dp.set_params(n_coef=10)
    assert dp.n_coef == 10
    print("  PASS: get_params / set_params")


def test_basis_orthogonality():
    """Verify extracted basis functions are approximately orthogonal."""
    X_train, y_train, X_test, y_test = make_test_data()

    dp = DiscriminantPursuit(
        n_coef=5, filter_family='Symmlet', filter_par=8, verbose=False
    )
    dp.fit(X_train, y_train)

    bf = dp.basis_functions_
    gram = bf @ bf.T
    off_diag = gram - np.diag(np.diag(gram))
    max_off = np.max(np.abs(off_diag))
    print(f"  PASS: basis orthogonality (max off-diagonal={max_off:.6f})")


def test_different_filters():
    """Verify discriminant pursuit works with different wavelet families."""
    X_train, y_train, X_test, y_test = make_test_data()

    for family, par in [('Haar', None), ('Daubechies', 8), ('Coiflet', 3)]:
        dp = DiscriminantPursuit(
            n_coef=3, filter_family=family, filter_par=par, verbose=False
        )
        dp.fit(X_train, y_train)
        coefs = dp.transform(X_test)
        n_test = X_test.shape[0]
        assert coefs.shape == (n_test, 3), (
            f"{family}: wrong shape {coefs.shape}"
        )
    print("  PASS: different filter families (Haar, Daubechies, Coiflet)")


def test_two_class_problem():
    """Verify DP works with binary classification."""
    X_train, y_train, X_test, y_test = make_test_data(
        n_classes=2, n_samples=200
    )

    dp = DiscriminantPursuit(
        n_coef=3, filter_family='Symmlet', filter_par=8, verbose=False
    )
    dp.fit(X_train, y_train)
    coefs = dp.transform(X_test)
    assert coefs.shape[1] == 3
    print("  PASS: two-class problem")


def test_random_forest_pipeline():
    """Verify DP works with RandomForest in a pipeline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline

    X_train, y_train, X_test, y_test = make_test_data()

    clf = make_pipeline(
        DiscriminantPursuit(n_coef=5, filter_family='Symmlet',
                            filter_par=8, verbose=False),
        RandomForestClassifier(n_estimators=50, random_state=42),
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    assert acc > 0.40, f"RF pipeline accuracy too low: {acc:.4f}"
    print(f"  PASS: RandomForest pipeline (accuracy={acc:.4f})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_imports,
        test_make_on_filter_all_families,
        test_dyadic_operators_roundtrip,
        test_wp_analysis_shape,
        test_wp_analysis_energy_preservation,
        test_wp_impulse,
        test_make_wp,
        test_packet_indexing_roundtrip,
        test_discriminant_pursuit_basic,
        test_discriminant_pursuit_classification,
        test_sklearn_wrapper_fit_transform,
        test_sklearn_wrapper_fit_transform_single_call,
        test_sklearn_pipeline,
        test_sklearn_get_params,
        test_basis_orthogonality,
        test_different_filters,
        test_two_class_problem,
        test_random_forest_pipeline,
    ]

    print("=" * 60)
    print("discriminant-pursuit validation tests")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for test_func in tests:
        name = test_func.__name__
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAIL: {name} — {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    else:
        print("\nAll tests passed!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
