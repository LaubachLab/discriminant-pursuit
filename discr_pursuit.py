"""
discr_pursuit.py — Discriminant Pursuit via Wavelet Packets

A Python port of the discriminant pursuit algorithm originally implemented
in Matlab by Jonathan Buckheit (Stanford University, Department of Statistics)
as part of the Wavelab toolbox. Discriminant pursuit finds wavelet packet
basis functions that maximize the separation between class means in the
wavelet packet dictionary.

The Wavelab functions ported here are the minimal subset required for
wavelet packet analysis, synthesis, and discriminant pursuit:
    WPAnalysis, WPImpulse, MakeWP, MakeONFilter,
    DownDyadHi, DownDyadLo, UpDyadHi, UpDyadLo,
    aconv, iconv, MirrorFilt, packet, ix2pkt, pkt2ix

Scikit-learn Interface
----------------------
The DiscriminantPursuit class wraps the core algorithm as a scikit-learn
compatible transformer (BaseEstimator, TransformerMixin). It supports:

    fit(X, y)           Find discriminant basis functions from labeled data.
    transform(X)        Project time series onto the fitted basis functions.
    fit_transform(X, y) Fit and transform in one call.
    get_params()        Return estimator parameters (supports clone()).
    set_params(**p)     Set parameters (supports GridSearchCV).

The transformer can be used as a drop-in feature extraction step in any
sklearn Pipeline, cross-validation workflow, or hyperparameter search:

    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import RidgeClassifierCV
    clf = make_pipeline(DiscriminantPursuit(n_coef=10), RidgeClassifierCV())
    clf.fit(X_train, y_train)

Input time series must have length n = 2**J for integer J. After fitting,
the selected basis functions are available as dp.basis_functions_ (shape:
n_coef x n_timepoints) and their packet tree coordinates as
dp.packet_indices_ (list of (depth, block, translation) tuples).

Original Matlab code:
    Buckheit, J. & Donoho, D.L. (1995). Wavelab toolbox.
    Stanford University, Department of Statistics.
    Discriminant pursuit code obtained from J. Buckheit in 1995
    and modified by M. Laubach with permission.

References:
    Buckheit, J. & Donoho, D.L. (1995). Improved linear discrimination
    using time-frequency dictionaries. Proc. SPIE, 2569, 540-551.

Python port by Mark Laubach (American University) with Claude (Anthropic).
License: BSD-3-Clause
"""

__version__ = "0.1.0"

import numpy as np
from scipy.signal import lfilter
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "DiscriminantPursuit",
    "discriminant_pursuit",
    "make_on_filter",
    "wp_analysis",
    "wp_impulse",
    "make_wp",
]

# ============================================================================
# WAVELET FILTERS
# ============================================================================

def make_on_filter(family, par=None):
    """
    Generate orthonormal QMF filter for wavelet transform.

    Parameters
    ----------
    family : str
        Wavelet family. One of:
        'Haar'        — no par required; discontinuous, length-2 filter.
        'Daubechies'  — par in {4, 6, 8, 10, 12, 14, 16, 18, 20}; maximally
                        smooth scaling function indexed by filter length.
        'Symmlet'     — par in {4, 5, 6, 7, 8, 9, 10}; least-asymmetric
                        compactly-supported wavelets with maximum vanishing
                        moments.
        'Coiflet'     — par in {1, 2, 3, 4, 5}; both mother and father
                        wavelets have 2*par vanishing moments.
    par : int, optional
        Family order parameter (not used for Haar).

    Returns
    -------
    qmf : ndarray
        Normalized quadrature mirror filter.

    Notes
    -----
    Filter coefficients are ported from the Wavelab toolbox (Buckheit &
    Donoho, 1995). The returned filter is normalized so that norm(qmf) == 1,
    matching the Wavelab convention.
    """
    if family == 'Haar':
        f = np.array([1.0, 1.0]) / np.sqrt(2)

    elif family == 'Daubechies':
        filters = {
            4:  [.482962913145,  .836516303738,  .224143868042, -.129409522551],
            6:  [.332670552950,  .806891509311,  .459877502118, -.135011020010,
                 -.085441273882,  .035226291882],
            8:  [.230377813309,  .714846570553,  .630880767930, -.027983769417,
                 -.187034811719,  .030841381836,  .032883011667, -.010597401785],
            10: [.160102397974,  .603829269797,  .724308528438,  .138428145901,
                 -.242294887066, -.032244869585,  .077571493840, -.006241490213,
                 -.012580751999,  .003335725285],
            12: [.111540743350,  .494623890398,  .751133908021,  .315250351709,
                 -.226264693965, -.129766867567,  .097501605587,  .027522865530,
                 -.031582039317,  .000553842201,  .004777257511, -.001077301085],
            14: [.077852054085,  .396539319482,  .729132090846,  .469782287405,
                 -.143906003929, -.224036184994,  .071309219267,  .080612609151,
                 -.038029936935, -.016574541631,  .012550998556,  .000429577973,
                 -.001801640704,  .000353713800],
            16: [.054415842243,  .312871590914,  .675630736297,  .585354683654,
                 -.015829105256, -.284015542962,  .000472484574,  .128747426620,
                 -.017369301002, -.044088253931,  .013981027917,  .008746094047,
                 -.004870352993, -.000391740373,  .000675449406, -.000117476784],
            18: [.038077947364,  .243834674613,  .604823123690,  .657288078051,
                  .133197385825, -.293273783279, -.096840783223,  .148540749338,
                  .030725681479, -.067632829061,  .000250947115,  .022361662124,
                 -.004723204758, -.004281503682,  .001847646883,  .000230385764,
                 -.000251963189,  .000039347320],
            20: [.026670057901,  .188176800078,  .527201188932,  .688459039454,
                  .281172343661, -.249846424327, -.195946274377,  .127369340336,
                  .093057364604, -.071394147166, -.029457536822,  .033212674059,
                  .003606553567, -.010733175483,  .001395351747,  .001992405295,
                 -.000685856695, -.000116466855,  .000093588670, -.000013264203],
        }
        if par not in filters:
            raise ValueError(
                f"Daubechies par must be one of {sorted(filters)}; got {par}.")
        f = np.array(filters[par])

    elif family == 'Symmlet':
        filters = {
            4:  [-.107148901418, -.041910965125,  .703739068656,  1.136658243408,
                   .421234534204, -.140317624179, -.017824701442,   .045570345896],
            5:  [  .038654795955,  .041746864422, -.055344186117,   .281990696854,
                  1.023052966894,  .896581648380,  .023478923136,  -.247951362613,
                 -.029842499869,   .027632152958],
            6:  [  .021784700327,  .004936612372, -.166863215412,  -.068323121587,
                   .694457972958,  1.113892783926,  .477904371333,  -.102724969862,
                 -.029783751299,   .063250562660,  .002499922093,  -.011031867509],
            7:  [  .003792658534, -.001481225915, -.017870431651,   .043155452582,
                   .096014767936, -.070078291222,  .024665659489,   .758162601964,
                  1.085782709814,  .408183939725, -.198056706807,  -.152463871896,
                  .005671342686,   .014521394762],
            8:  [  .002672793393, -.000428394300, -.021145686528,   .005386388754,
                   .069490465911, -.038493521263, -.073462508761,   .515398670374,
                  1.099106630537,  .680745347190, -.086653615406,  -.202648655286,
                  .010758611751,   .044823623042, -.000766690896,  -.004783458512],
            9:  [  .001512487309, -.000669141509, -.014515578553,   .012528896242,
                   .087791251554, -.025786445930, -.270893783503,   .049882830959,
                   .873048407349,  1.015259790832,  .337658923602,  -.077172161097,
                  .000825140929,   .042744433602, -.016303351226,  -.018769396836,
                  .000876502539,   .001981193736],
            10: [  .001089170447,  .000135245020, -.012220642630,  -.002072363923,
                   .064950924579,  .016418869426, -.225558972234,  -.100240215031,
                   .667071338154,  1.088251530500,  .542813011213,  -.050256540092,
                 -.045240772218,   .070703567550,  .008152816799,  -.028786231926,
                 -.001137535314,   .006495728375,  .000080661204,  -.000649589896],
        }
        if par not in filters:
            raise ValueError(
                f"Symmlet par must be one of {sorted(filters)}; got {par}.")
        f = np.array(filters[par])

    elif family == 'Coiflet':
        filters = {
            1: [  .038580777748, -.126969125396, -.077161555496,
                   .607491641386,  .745687558934,  .226584265197],
            2: [  .016387336463, -.041464936782, -.067372554722,
                   .386110066823,  .812723635450,  .417005184424,
                 -.076488599078, -.059434418646,  .023680171947,
                  .005611434819, -.001823208871, -.000720549445],
            3: [ -.003793512864,  .007782596426,  .023452696142,
                 -.065771911281, -.061123390003,  .405176902410,
                  .793777222626,  .428483476378, -.071799821619,
                 -.082301927106,  .034555027573,  .015880544864,
                 -.009007976137, -.002574517688,  .001117518771,
                  .000466216960, -.000070983303, -.000034599773],
            4: [  .000892313668, -.001629492013, -.007346166328,
                   .016068943964,  .026682300156, -.081266699680,
                 -.056077313316,  .415308407030,  .782238930920,
                  .434386056491, -.066627474263, -.096220442034,
                  .039334427123,  .025082261845, -.015211731527,
                 -.005658286686,  .003751436157,  .001266561929,
                 -.000589020757, -.000259974552,  .000062339034,
                  .000031229876, -.000003259680, -.000001784985],
            5: [ -.000212080863,  .000358589677,  .002178236305,
                 -.004159358782, -.010131117538,  .023408156762,
                  .028168029062, -.091920010549, -.052043163216,
                  .421566206729,  .774289603740,  .437991626228,
                 -.062035963906, -.105574208706,  .041289208741,
                  .032683574283, -.019761779012, -.009164231153,
                  .006764185419,  .002433373209, -.001662863769,
                 -.000638131296,  .000302259520,  .000140541149,
                 -.000041340484, -.000021315014,  .000003734597,
                  .000002063806, -.000000167408, -.000000095158],
        }
        if par not in filters:
            raise ValueError(
                f"Coiflet par must be one of {sorted(filters)}; got {par}.")
        f = np.array(filters[par])

    else:
        raise ValueError(
            f"Unknown filter family '{family}'. "
            f"Choose from: 'Haar', 'Daubechies', 'Symmlet', 'Coiflet'.")

    return f / np.linalg.norm(f)


# ============================================================================
# CONVOLUTION UTILITIES
# ============================================================================

def _mirror_filt(x):
    """Apply (-1)^t modulation: h(t) = (-1)^(t-1) * x(t)."""
    return -((-1) ** np.arange(1, len(x) + 1)) * x


def _aconv(f, x):
    """Periodic convolution with time-reverse of f (low-pass downsampling)."""
    n = len(x)
    p = len(f)
    if p < n:
        xpadded = np.concatenate([x, x[:p]])
    else:
        z = np.zeros(p)
        for i in range(p):
            z[i] = x[i % n]
        xpadded = np.concatenate([x, z])
    fflip = f[::-1]
    ypadded = lfilter(fflip, 1.0, xpadded)
    return ypadded[p - 1:n + p - 1]


def _iconv(f, x):
    """Periodic convolution with f (high-pass downsampling)."""
    n = len(x)
    p = len(f)
    if p <= n:
        xpadded = np.concatenate([x[n - p:n], x])
    else:
        z = np.zeros(p)
        for i in range(p):
            z[i] = x[(p * n - p + i) % n]
        xpadded = np.concatenate([z, x])
    ypadded = lfilter(f, 1.0, xpadded)
    return ypadded[p:n + p]


def _lshift(x):
    """Circular left shift."""
    return np.concatenate([x[1:], x[:1]])


def _rshift(x):
    """Circular right shift."""
    return np.concatenate([x[-1:], x[:-1]])


def _upsample(x):
    """Upsample by factor 2 with zero-insertion."""
    n = len(x) * 2
    y = np.zeros(n)
    y[0::2] = x
    return y


# ============================================================================
# DYADIC OPERATORS
# ============================================================================

def _down_dyad_lo(x, qmf):
    """Lo-pass downsampling operator (periodized)."""
    d = _aconv(qmf, x)
    return d[0::2]


def _down_dyad_hi(x, qmf):
    """Hi-pass downsampling operator (periodized)."""
    d = _iconv(_mirror_filt(qmf), _lshift(x))
    return d[0::2]


def _up_dyad_lo(x, qmf):
    """Lo-pass upsampling operator (periodized)."""
    return _iconv(qmf, _upsample(x))


def _up_dyad_hi(x, qmf):
    """Hi-pass upsampling operator (periodized)."""
    return _aconv(_mirror_filt(qmf), _rshift(_upsample(x)))


# ============================================================================
# WAVELET PACKET TABLE INDEXING
# ============================================================================

def _packet(d, b, n):
    """Return indices for block (d, b) in packet table of signal length n."""
    npack = 2 ** d
    block_size = n // npack
    start = b * block_size
    return slice(start, start + block_size)


def _ix2pkt(ix, D, n):
    """Convert linear index to packet table index (d, b, k)."""
    d = ix // n
    row = ix % n
    nc = n // (2 ** d)
    b = row // nc
    k = row - b * nc
    return d, b, k


def _pkt2ix(d, b, k, D, n):
    """Convert packet table index (d, b, k) to linear index."""
    nc = n // (2 ** d)
    row = b * nc + k
    ix = row + d * n
    return ix


# ============================================================================
# WAVELET PACKET ANALYSIS AND SYNTHESIS
# ============================================================================

def wp_analysis(x, D, qmf):
    """
    Wavelet packet analysis: compute full packet table.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Signal of dyadic length n = 2^J.
    D : int
        Depth of decomposition (D <= J).
    qmf : ndarray
        Quadrature mirror filter.

    Returns
    -------
    wp : ndarray, shape (n, D+1)
        Wavelet packet table. Coefficients for frequency interval
        [b/2^d, (b+1)/2^d] are stored in wp[packet(d,b,n), d].
    """
    n = len(x)
    wp = np.zeros((n, D + 1))
    wp[:, 0] = x

    for d in range(D):
        lson = 0
        for b in range(2 ** d):
            s = wp[_packet(d, b, n), d]
            ls = _down_dyad_lo(s, qmf)
            hs = _down_dyad_hi(s, qmf)
            wp[_packet(d + 1, 2 * b + lson, n), d + 1] = ls
            wp[_packet(d + 1, 2 * b + 1 - lson, n), d + 1] = hs
            lson = 1 - lson

    return wp


def wp_impulse(wp, d, b, k, qmf):
    """
    Compute wavelet packet table of a single basis element at (d, b, k).

    Parameters
    ----------
    wp : ndarray, shape (n, D+1)
        Packet table (used only for shape).
    d, b, k : int
        Packet table indices.
    qmf : ndarray
        Quadrature mirror filter.

    Returns
    -------
    dwp : ndarray, shape (n, D+1)
        Packet table with impulse at (d, b, k) propagated.
    """
    n, L = wp.shape
    D = L - 1
    dwp = np.zeros((n, L))
    dwp[b * (n // 2 ** d) + k, d] = 1.0

    # Propagate downward
    if d < D:
        blo = b
        bhi = blo + 1
        for dl in range(d, D):
            for bi in range(blo, bhi):
                x = dwp[_packet(dl, bi, n), dl]
                xl = _down_dyad_lo(x, qmf)
                xh = _down_dyad_hi(x, qmf)
                lson = bi % 2
                dwp[_packet(dl + 1, 2 * bi + lson, n), dl + 1] = xl
                dwp[_packet(dl + 1, 2 * bi + 1 - lson, n), dl + 1] = xh
            blo = 2 * blo
            bhi = 2 * bhi

    # Propagate upward
    if d > 0:
        xl = dwp[_packet(d, b, n), d]
        bi = b
        for dl in range(d - 1, -1, -1):
            bparent = bi // 2
            upchan = (bparent % 2 + bi % 2) % 2
            if upchan:
                xl = _up_dyad_hi(xl, qmf)
            else:
                xl = _up_dyad_lo(xl, qmf)
            dwp[_packet(dl, bparent, n), dl] = xl
            bi = bparent

    return dwp


def make_wp(d, b, k, qmf, n):
    """
    Construct a single wavelet packet basis function.

    Parameters
    ----------
    d, b, k : int
        Packet table indices.
    qmf : ndarray
        Quadrature mirror filter.
    n : int
        Signal length.

    Returns
    -------
    wavepkt : ndarray, shape (n,)
        The wavelet packet basis function in the time domain.
    """
    L = d + 1
    wp = np.zeros((n, L))
    wp = wp_impulse(wp, d, b, k, qmf)
    return wp[:, 0]


# ============================================================================
# DISCRIMINANT PURSUIT
# ============================================================================

def discriminant_pursuit(
    n_coef,
    X_train,
    y_train,
    X_test,
    y_test,
    qmf,
    trim_percent=10,
    verbose=True,
):
    """
    Discriminant pursuit via wavelet packets.

    Finds wavelet packet basis functions that maximize separation between
    class means. At each step, selects the wavelet packet coefficient with
    the largest amplitude in the pairwise class-mean contrasts, then
    deflates the packet tables to remove that component.

    Parameters
    ----------
    n_coef : int
        Number of discriminant pursuit basis functions to extract.
    X_train : ndarray, shape (n_train, n_timepoints)
        Training data. n_timepoints must be a power of 2.
    y_train : array-like, shape (n_train,)
        Training labels.
    X_test : ndarray, shape (n_test, n_timepoints)
        Test data.
    y_test : array-like, shape (n_test,)
        Test labels.
    qmf : ndarray
        Quadrature mirror filter (from make_on_filter).
    trim_percent : float, default=10
        Percentage for trimmed mean of class averages.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict with keys:
        'train_coefs' : ndarray, shape (n_train, n_coef)
            Wavelet packet coefficients for training data.
        'test_coefs' : ndarray, shape (n_test, n_coef)
            Wavelet packet coefficients for test data.
        'basis_functions' : ndarray, shape (n_coef, n_timepoints)
            The discriminant pursuit basis functions in the time domain.
        'amplitudes' : ndarray, shape (n_coef,)
            Amplitude of each selected basis function in the class contrasts.
        'coef_indices' : ndarray, shape (n_coef,)
            Linear indices into the wavelet packet table.
        'packet_indices' : list of tuples
            (d, b, k) indices for each selected basis function.
    """
    from scipy.stats import trim_mean

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    n_train, n = X_train.shape
    n_test = X_test.shape[1]
    assert n == n_test, "Train and test must have same number of timepoints"

    D = int(np.log2(n))
    assert 2 ** D == n, f"Signal length must be power of 2, got {n}"

    classes = np.unique(y_train)
    g = len(classes)
    n_contr = g * (g - 1) // 2

    if verbose:
        print(f"Discriminant Pursuit: {n} timepoints, {g} classes, "
              f"{n_contr} contrasts, extracting {n_coef} basis functions")
        print(f"  Filter: length {len(qmf)}, depth D={D}")

    # Compute trimmed class means
    avgscores = np.zeros((n, g))
    for i, cls in enumerate(classes):
        mask = y_train == cls
        if trim_percent > 0:
            avgscores[:, i] = trim_mean(X_train[mask], trim_percent / 100, axis=0)
        else:
            avgscores[:, i] = X_train[mask].mean(axis=0)

    # Compute wavelet packet tables for all pairwise contrasts
    pkt_tables = np.zeros((n_contr, n * (D + 1)))
    k_contr = 0
    for i in range(g - 1):
        for j in range(i + 1, g):
            contr = avgscores[:, i] - avgscores[:, j]
            pkt = wp_analysis(contr, D, qmf)
            pkt_tables[k_contr] = pkt.flatten(order='F')
            k_contr += 1

    # Greedy selection of discriminant basis functions
    coef_indices = np.zeros(n_coef, dtype=int)
    amplitudes = np.zeros(n_coef)
    packet_indices = []

    for step in range(n_coef):
        # Find the coefficient with maximum amplitude across all contrasts
        amp = np.zeros(n_contr)
        ind = np.zeros(n_contr, dtype=int)
        for j in range(n_contr):
            table_j = np.abs(pkt_tables[j])
            ind[j] = np.argmax(table_j)
            amp[j] = table_j[ind[j]]

        best_contr = np.argmax(amp)
        amplitudes[step] = amp[best_contr]
        coef_indices[step] = ind[best_contr]

        d, b, k = _ix2pkt(ind[best_contr], D, n)
        packet_indices.append((d, b, k))

        if verbose:
            print(f"  Step {step+1}: d={d}, b={b}, k={k}, "
                  f"amplitude={amplitudes[step]:.4f}")

        # Deflate: remove this component from all contrast packet tables
        for j in range(n_contr):
            pkt = pkt_tables[j].reshape((n, D + 1), order='F')
            a = pkt[_pkt2ix(d, b, k, D, n) % n, d]

            # Create impulse response for this basis function
            wp_shape = np.zeros((n, D + 1))
            dwp = wp_impulse(wp_shape, d, b, k, qmf)

            pkt = pkt - a * dwp
            pkt_tables[j] = pkt.flatten(order='F')

    # Project train and test data onto selected basis functions
    train_coefs = np.zeros((n_train, n_coef))
    for i in range(n_train):
        pkt = wp_analysis(X_train[i], D, qmf)
        flat = pkt.flatten(order='F')
        train_coefs[i] = flat[coef_indices]

    test_coefs = np.zeros((X_test.shape[0], n_coef))
    for i in range(X_test.shape[0]):
        pkt = wp_analysis(X_test[i], D, qmf)
        flat = pkt.flatten(order='F')
        test_coefs[i] = flat[coef_indices]

    # Reconstruct basis functions in the time domain
    basis_functions = np.zeros((n_coef, n))
    for i in range(n_coef):
        d, b, k = packet_indices[i]
        basis_functions[i] = make_wp(d, b, k, qmf, n)

    return {
        'train_coefs': train_coefs,
        'test_coefs': test_coefs,
        'basis_functions': basis_functions,
        'amplitudes': amplitudes,
        'coef_indices': coef_indices,
        'packet_indices': packet_indices,
    }


# ============================================================================
# SCIKIT-LEARN COMPATIBLE WRAPPER
# ============================================================================

class DiscriminantPursuit(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for Discriminant Pursuit.

    Finds the most discriminative wavelet packet basis functions from
    training data and projects new data onto those basis functions.

    Compatible with sklearn pipelines, GridSearchCV, and cross_val_score.

    Parameters
    ----------
    n_coef : int, default=10
        Number of discriminant basis functions to extract.
    filter_family : str, default='Symmlet'
        Wavelet filter family. Options: 'Haar', 'Daubechies', 'Symmlet',
        'Coiflet'.
    filter_par : int, default=8
        Filter parameter (e.g., Symmlet-8, Daubechies-10, Coiflet-3).
    trim_percent : float, default=10
        Percentage for trimmed mean of class averages.
    verbose : bool, default=False
        Print progress during fitting.

    Attributes
    ----------
    qmf_ : ndarray
        Fitted quadrature mirror filter.
    D_ : int
        Depth of wavelet packet decomposition.
    coef_indices_ : ndarray, shape (n_coef,)
        Linear indices into the wavelet packet table.
    packet_indices_ : list of tuples
        (d, b, k) indices for each selected basis function.
    basis_functions_ : ndarray, shape (n_coef, n_timepoints)
        The discriminant pursuit basis functions in the time domain.
    amplitudes_ : ndarray, shape (n_coef,)
        Amplitude of each selected basis function in the class contrasts.

    Examples
    --------
    >>> from discr_pursuit import DiscriminantPursuit
    >>> from sklearn.linear_model import RidgeClassifierCV
    >>> from sklearn.pipeline import make_pipeline
    >>> dp = DiscriminantPursuit(n_coef=10, filter_family='Symmlet', filter_par=8)
    >>> clf = make_pipeline(dp, RidgeClassifierCV())
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self, n_coef=10, filter_family='Symmlet', filter_par=8,
                 trim_percent=10, verbose=False):
        self.n_coef = n_coef
        self.filter_family = filter_family
        self.filter_par = filter_par
        self.trim_percent = trim_percent
        self.verbose = verbose

    def fit(self, X, y):
        """
        Find the most discriminative wavelet packet basis functions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_timepoints)
            Training time series. n_timepoints must be a power of 2.
        y : array-like, shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[1]
        self.D_ = int(np.log2(n))
        assert 2 ** self.D_ == n, f"Signal length must be power of 2, got {n}"

        self.qmf_ = make_on_filter(self.filter_family, self.filter_par)
        self.n_timepoints_ = n

        # Run the core algorithm
        results = discriminant_pursuit(
            n_coef=self.n_coef,
            X_train=X, y_train=y,
            X_test=X[:1], y_test=y[:1],  # dummy, not used for fitting
            qmf=self.qmf_,
            trim_percent=self.trim_percent,
            verbose=self.verbose,
        )

        self.coef_indices_ = results['coef_indices']
        self.packet_indices_ = results['packet_indices']
        self.basis_functions_ = results['basis_functions']
        self.amplitudes_ = results['amplitudes']
        self.classes_ = np.unique(y)

        return self

    def transform(self, X):
        """
        Project time series onto the fitted discriminant basis functions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_timepoints)
            Time series data. Must have the same number of timepoints
            as the training data.

        Returns
        -------
        coefs : ndarray, shape (n_samples, n_coef)
            Wavelet packet coefficients for each discriminant basis function.
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, 'coef_indices_')

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        coefs = np.zeros((n_samples, self.n_coef))

        for i in range(n_samples):
            pkt = wp_analysis(X[i], self.D_, self.qmf_)
            flat = pkt.flatten(order='F')
            coefs[i] = flat[self.coef_indices_]

        return coefs
