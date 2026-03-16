"""
dp_visualization.py — Time-Frequency Visualization for Wavelet and Cosine Packets

Provides visualization of wavelet packet and cosine packet decompositions
in time-frequency space. Ported from the Wavelab toolbox visualization
functions (PlotPhaseTiling, PlotPhasePlane, PlotPacketTable).

Python port by Mark Laubach (American University) with Claude (Anthropic).
License: BSD-3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def plot_packet_table(pkt, scal=0, figsize=(12, 6), cmap='RdBu_r', title=None):
    """
    Display entries in a wavelet or cosine packet table as spike plots.

    Each row shows one depth level, with coefficients plotted as vertical
    bars. Dashed boxes delineate the frequency blocks at each level.

    Parameters
    ----------
    pkt : ndarray, shape (n, D+1)
        Wavelet or cosine packet table from wp_analysis or cp_analysis.
    scal : float
        Scale factor. 0 = autoscale per level.
    figsize : tuple
    cmap : str
        Not used for spike plot, reserved for future heatmap option.
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    n, L = pkt.shape
    D = L - 1

    fig, axes = plt.subplots(D + 1, 1, figsize=figsize, sharex=True)
    if D == 0:
        axes = [axes]

    t = (np.arange(0.5, n)) / n

    for d in range(D + 1):
        ax = axes[d]
        coeffs = pkt[:, d]

        if scal == 0:
            mult = 0.4 / (np.max(np.abs(coeffs)) + 1e-15)
        else:
            mult = scal

        # Plot coefficients as stems
        ax.vlines(t, 0, coeffs * mult, colors='steelblue', linewidth=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)

        # Draw block boundaries
        nbox = 2 ** d
        for b in range(nbox + 1):
            ax.axvline(b / nbox, color='gray', linewidth=0.5,
                       linestyle=':', alpha=0.5)

        ax.set_ylabel(f'd={d}', fontsize=8)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])

    axes[-1].set_xlabel('Position (normalized)')
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_phase_tiling(pkt_type, basis_indices, n, D, figsize=(8, 6),
                      title=None):
    """
    Partition phase space by rectangular blocks for selected basis elements.

    For wavelet packets, the x-axis is time and y-axis is frequency.
    For cosine packets, the axes are swapped.

    Parameters
    ----------
    pkt_type : str
        'WP' for wavelet packets, 'CP' for cosine packets.
    basis_indices : list of tuples
        List of (d, b, k) tuples for the selected basis elements.
    n : int
        Signal length.
    D : int
        Maximum decomposition depth.
    figsize : tuple
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for d, b, k in basis_indices:
        nbox = n // (2 ** d)

        if pkt_type == 'WP':
            # WP: x = time (k/nbox), y = frequency (b/2^d)
            x0 = k / nbox
            width = 1 / nbox
            y0 = b / (2 ** d)
            height = 1 / (2 ** d)
        else:
            # CP: x = time (b/2^d), y = frequency (k/nbox)
            x0 = b / (2 ** d)
            width = 1 / (2 ** d)
            y0 = k / nbox
            height = 1 / nbox

        rect = Rectangle((x0, y0), width, height,
                          linewidth=1.5, edgecolor='steelblue',
                          facecolor='steelblue', alpha=0.3)
        ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time' if pkt_type == 'WP' else 'Time')
    ax.set_ylabel('Frequency' if pkt_type == 'WP' else 'Frequency')
    ax.set_aspect('equal')

    if title is None:
        title = f'Phase Plane Tiling ({pkt_type})'
    ax.set_title(title)
    return fig


def plot_phase_plane(pkt_type, pkt, basis_indices, figsize=(8, 6),
                     cmap='hot_r', title=None):
    """
    Plot time-frequency energy distribution for selected basis elements.

    Each selected basis element is represented as a rectangle in the
    time-frequency plane, with color intensity proportional to the
    energy of the coefficient.

    Parameters
    ----------
    pkt_type : str
        'WP' for wavelet packets, 'CP' for cosine packets.
    pkt : ndarray, shape (n, D+1)
        Packet table.
    basis_indices : list of tuples
        List of (d, b, k) tuples for the selected basis elements.
    figsize : tuple
    cmap : str
        Colormap for energy intensity.
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    n, L = pkt.shape
    total_energy = np.sum(pkt[:, 0] ** 2) + 1e-15

    fig, ax = plt.subplots(figsize=figsize)
    patches = []
    colors = []

    for d, b, k in basis_indices:
        nbox = n // (2 ** d)
        coef_energy = n * (pkt[b * nbox + k, d] / np.sqrt(total_energy)) ** 2

        if pkt_type == 'WP':
            x0 = k / nbox
            width = 1 / nbox
            y0 = b / (2 ** d)
            height = 1 / (2 ** d)
        else:
            x0 = b / (2 ** d)
            width = 1 / (2 ** d)
            y0 = k / nbox
            height = 1 / nbox

        rect = Rectangle((x0, y0), width, height)
        patches.append(rect)
        colors.append(coef_energy)

    if patches:
        pc = PatchCollection(patches, cmap=cmap, alpha=0.8,
                             edgecolors='gray', linewidths=0.5)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        plt.colorbar(pc, ax=ax, label='Relative energy')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')

    if title is None:
        title = f'Phase Plane ({pkt_type})'
    ax.set_title(title)
    return fig


def plot_dp_phase_plane(dp_results, pkt_type='WP', n=None, figsize=(8, 6),
                        title=None):
    """
    Plot the discriminant pursuit basis functions in time-frequency space.

    Each selected basis function is shown as a rectangle in the
    time-frequency plane, with color proportional to its discriminative
    amplitude.

    Parameters
    ----------
    dp_results : dict
        Output from discriminant_pursuit().
    pkt_type : str
        'WP' for wavelet packets, 'CP' for cosine packets.
    n : int, optional
        Signal length. Inferred from basis_functions if not provided.
    figsize : tuple
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    basis_indices = dp_results['packet_indices']
    amplitudes = dp_results['amplitudes']

    if n is None:
        n = dp_results['basis_functions'].shape[1]

    fig, ax = plt.subplots(figsize=figsize)
    patches = []
    colors = []

    amp_max = np.max(np.abs(amplitudes)) + 1e-15

    for i, (d, b, k) in enumerate(basis_indices):
        nbox = n // (2 ** d)

        if pkt_type == 'WP':
            x0 = k / nbox
            width = 1 / nbox
            y0 = b / (2 ** d)
            height = 1 / (2 ** d)
        else:
            x0 = b / (2 ** d)
            width = 1 / (2 ** d)
            y0 = k / nbox
            height = 1 / nbox

        rect = Rectangle((x0, y0), width, height)
        patches.append(rect)
        colors.append(amplitudes[i] / amp_max)

        # Add label
        cx = x0 + width / 2
        cy = y0 + height / 2
        ax.text(cx, cy, f'{i + 1}', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    if patches:
        pc = PatchCollection(patches, cmap='YlOrRd', alpha=0.8,
                             edgecolors='black', linewidths=1)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        plt.colorbar(pc, ax=ax, label='Normalized amplitude')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')

    if title is None:
        title = 'Discriminant Pursuit Basis Functions in Phase Space'
    ax.set_title(title)
    return fig


def plot_full_vs_dp_decomposition(pkt, dp_results, pkt_type='WP',
                                   figsize=(14, 6), title=None):
    """
    Side-by-side comparison of full packet table and DP-selected features.

    Left panel: full time-frequency decomposition (all coefficients).
    Right panel: only the coefficients selected by discriminant pursuit.

    Parameters
    ----------
    pkt : ndarray, shape (n, D+1)
        Full packet table (from wp_analysis or cp_analysis of a class mean).
    dp_results : dict
        Output from discriminant_pursuit().
    pkt_type : str
        'WP' or 'CP'.
    figsize : tuple
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    n, L = pkt.shape
    D = L - 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Full decomposition heatmap
    im1 = ax1.imshow(np.abs(pkt).T, aspect='auto', origin='lower',
                     cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Depth')
    ax1.set_title('Full Decomposition')
    plt.colorbar(im1, ax=ax1, label='|coefficient|')

    # DP-selected only
    dp_pkt = np.zeros_like(pkt)
    for d, b, k in dp_results['packet_indices']:
        nbox = n // (2 ** d)
        idx = b * nbox + k
        if 0 <= idx < n and 0 <= d < L:
            dp_pkt[idx, d] = pkt[idx, d]

    im2 = ax2.imshow(np.abs(dp_pkt).T, aspect='auto', origin='lower',
                     cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Depth')
    ax2.set_title(f'DP-Selected ({len(dp_results["packet_indices"])} features)')
    plt.colorbar(im2, ax=ax2, label='|coefficient|')

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig
