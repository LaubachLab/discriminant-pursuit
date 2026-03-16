"""
cosine_packets.py — Cosine Packet Analysis (Local Cosine Transform)

Ported from the Wavelab toolbox (Donoho et al., 1995). Cosine packets
provide a time-frequency dictionary complementary to wavelet packets:
wavelet packets tile the frequency axis dyadically, while cosine packets
tile the time axis dyadically with smooth windowed cosine bases.

Cosine packets are well suited for signals with slowly varying frequency
content (e.g., LFPs, EEG, force/velocity curves) where the discriminative
structure is better captured by local frequency changes than by local
waveform shape.

References:
    Coifman, R.R. & Meyer, Y. (1991). Remarques sur l'analyse de Fourier
    à fenêtre. Comptes Rendus de l'Académie des Sciences, 312, 259-261.

    Malvar, H. (1990). Lapped transforms for efficient transform/subband
    coding. IEEE Trans. ASSP, 38(6), 969-978.

Python port by Mark Laubach (American University) with Claude (Anthropic).
License: BSD-3-Clause
"""

import numpy as np


# ============================================================================
# BELL FUNCTIONS (TAPER WINDOWS)
# ============================================================================

def make_on_bell(name, m):
    """
    Make bell (taper window) for orthonormal local cosine analysis.

    Parameters
    ----------
    name : str
        'Sine' or 'Trivial'.
    m : int
        Length of bell (half-overlap region).

    Returns
    -------
    bp : ndarray, shape (m,)
        Interior part of bell.
    bm : ndarray, shape (m,)
        Exterior part of bell.
    """
    xi = (1 + (np.arange(0.5, m) / m)) / 2
    if name == 'Trivial':
        bp = np.sqrt(xi)
    elif name == 'Sine':
        bp = np.sin(np.pi / 2 * xi)
    else:
        raise ValueError(f"Unknown bell: {name}. Use 'Sine' or 'Trivial'.")
    bm = np.sqrt(1 - bp ** 2)
    return bp, bm


# ============================================================================
# DCT-IV
# ============================================================================

def dct_iv(x):
    """
    Type-IV Discrete Cosine Transform.

    c_m = sqrt(2/N) * sum_n x(n) * cos(pi * (m-0.5) * (n-0.5) / N)

    The DCT-IV is its own inverse: x = dct_iv(dct_iv(x)).

    Parameters
    ----------
    x : ndarray, shape (N,)

    Returns
    -------
    c : ndarray, shape (N,)
    """
    N = len(x)
    n2 = 2 * N
    y = np.zeros(4 * n2)
    y[1::2][:N] = x
    z = np.fft.fft(y)
    c = np.sqrt(4 / n2) * np.real(z[1::2][:N])
    return c


# ============================================================================
# FOLDING / UNFOLDING
# ============================================================================

def _fold(xc, xl, xr, bp, bm):
    """Folding projection with (+,-) polarity."""
    m = len(bp)
    n = len(xc)
    y = xc.copy()
    y[:m] = bp * y[:m] + bm * xl[n - 1:n - 1 - m:-1]
    y[n - 1:n - 1 - m:-1] = bp * y[n - 1:n - 1 - m:-1] - bm * xr[:m]
    return y


def _unfold(y, bp, bm):
    """Undo folding projection with (+,-) polarity."""
    n = len(y)
    m = len(bp)
    xc = y.copy()
    xl = np.zeros(n)
    xr = np.zeros(n)
    xc[:m] = bp * y[:m]
    xc[n - 1:n - 1 - m:-1] = bp * y[n - 1:n - 1 - m:-1]
    xl[n - 1:n - 1 - m:-1] = bm * y[:m]
    xr[:m] = -bm * y[n - 1:n - 1 - m:-1]
    return xc, xl, xr


def _edgefold(which, xc, bp, bm):
    """Folding at edges to ensure exact reconstruction."""
    n = len(xc)
    m = len(bp)
    extra = np.zeros(n)
    if which == 'left':
        extra[n - 1:n - 1 - m:-1] = xc[:m] * (1 - bp) / bm
    elif which == 'right':
        extra[:m] = -xc[n - 1:n - 1 - m:-1] * (1 - bp) / bm
    return extra


def _edgeunfold(which, xc, bp, bm):
    """Undo edge folding."""
    n = len(xc)
    m = len(bp)
    extra = np.zeros(n)
    if which == 'left':
        extra[:m] = xc[:m] * (1 - bp) / bp
    elif which == 'right':
        extra[n - 1:n - 1 - m:-1] = xc[n - 1:n - 1 - m:-1] * (1 - bp) / bp
    return extra


# ============================================================================
# COSINE PACKET ANALYSIS AND SYNTHESIS
# ============================================================================

def cp_analysis(x, D, bellname='Sine'):
    """
    Cosine packet analysis: compute full local cosine transform table.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Signal of dyadic length n = 2^J.
    D : int
        Depth of finest time splitting.
    bellname : str, default='Sine'
        Bell (taper window) name.

    Returns
    -------
    cp : ndarray, shape (n, D+1)
        Cosine packet table. Coefficients for time interval
        [b/2^d, (b+1)/2^d] are stored in cp[packet(d,b,n), d].
    """
    from discr_pursuit import _packet

    n = len(x)
    m = n // (2 ** D) // 2
    bp, bm = make_on_bell(bellname, m)

    cp = np.zeros((n, D + 1))
    x = x.copy().ravel()

    for d in range(D, -1, -1):
        nbox = 2 ** d
        for b in range(nbox):
            pkt_slice = _packet(d, b, n)
            if b == 0:
                xc = x[pkt_slice]
                xl = _edgefold('left', xc, bp, bm)
            else:
                xl = xc
                xc = xr

            if b + 1 < nbox:
                xr = x[_packet(d, b + 1, n)]
            else:
                xr = _edgefold('right', xc, bp, bm)

            y = _fold(xc, xl, xr, bp, bm)
            c = dct_iv(y)
            cp[pkt_slice, d] = c

    return cp


def cp_impulse(cp, d, b, k, bellname='Sine'):
    """
    Cosine packet table of a single cosine packet basis element.

    Parameters
    ----------
    cp : ndarray, shape (n, D+1)
        Packet table (used only for shape).
    d, b, k : int
        Packet table indices.
    bellname : str
        Bell name.

    Returns
    -------
    dcp : ndarray, shape (n, D+1)
        Packet table with impulse at (d, b, k) propagated.
    """
    from discr_pursuit import _packet

    n, L = cp.shape
    D = L - 1
    m = n // (2 ** D) // 2
    bp, bm = make_on_bell(bellname, m)

    dcp = np.zeros((n, L))

    # Build time-domain version
    c = np.zeros(n // (2 ** d))
    c[k] = 1.0
    s = dct_iv(c)
    xc, xl, xr = _unfold(s, bp, bm)

    x = np.zeros(n)
    x[_packet(d, b, n)] = xc
    if b > 0:
        x[_packet(d, b - 1, n)] = xl
    else:
        x[_packet(d, 0, n)] += _edgeunfold('left', xc, bp, bm)
    if b < (2 ** d - 1):
        x[_packet(d, b + 1, n)] = xr
    else:
        x[_packet(d, b, n)] += _edgeunfold('right', xc, bp, bm)

    # Decompose in cosine packets (downward)
    if d <= D:
        blo = b
        bhi = blo + 1
        for dl in range(d, D + 1):
            nbox = 2 ** dl
            xr_local = x[_packet(dl, blo, n)]
            if blo > 0:
                xc_local = x[_packet(dl, blo - 1, n)]
            else:
                xc_local = _edgefold('left', xr_local, bp, bm)

            for bi in range(blo, bhi):
                xl_local = xc_local
                xc_local = xr_local
                if bi + 1 < nbox:
                    xr_local = x[_packet(dl, bi + 1, n)]
                else:
                    xr_local = _edgefold('right', xc_local, bp, bm)
                s_local = _fold(xc_local, xl_local, xr_local, bp, bm)
                c_local = dct_iv(s_local)
                dcp[_packet(dl, bi, n), dl] = c_local

            blo = 2 * blo
            bhi = 2 * bhi

    # Upward propagation
    if d > 0:
        bi = b
        for dl in range(d - 1, -1, -1):
            bi = bi // 2
            xc_local = x[_packet(dl, bi, n)]
            nbox = 2 ** dl
            if bi == 0:
                xl_local = _edgefold('left', xc_local, bp, bm)
            else:
                xl_local = x[_packet(dl, bi - 1, n)]
            if bi + 1 < nbox:
                xr_local = x[_packet(dl, bi + 1, n)]
            else:
                xr_local = _edgefold('right', xc_local, bp, bm)
            s_local = _fold(xc_local, xl_local, xr_local, bp, bm)
            c_local = dct_iv(s_local)
            dcp[_packet(dl, bi, n), dl] = c_local

    return dcp


def make_cosine_packet(d, b, k, bellname='Sine', D=None, n=None):
    """
    Construct a single cosine packet basis function.

    Parameters
    ----------
    d, b, k : int
        Packet table indices.
    bellname : str
        Bell name.
    D : int, optional
        Maximum depth. Default: d.
    n : int, optional
        Signal length. Default: 2^(d+5).

    Returns
    -------
    cospkt : ndarray, shape (n,)
        The cosine packet basis function in the time domain.
    """
    if n is None:
        n = 2 ** (d + 5)
    if D is None:
        D = d

    L = D + 1
    cp = np.zeros((n, L))
    cp = cp_impulse(cp, d, b, k, bellname)
    return dct_iv(cp[:, 0])
