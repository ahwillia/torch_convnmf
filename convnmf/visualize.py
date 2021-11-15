"""
Functions to visualize model results.
"""
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_convnmf(
        data, W, H, tmin=0, tmax=-1, outer_pad=.05, inner_pad=.05,
        data_ax_height=.7, data_ax_width=.7, figsize=(10, 6)
    ):
    """Plots model factors and data.

    Parameters
    ----------
    data : ndarray
        Matrix holding data or model prediction (num_features x num_timesteps).
    W : ndarray
        Temporal motifs (num_features x num_components x num_lags)
    H : ndarray
        Times that motifs occur (num_componets x num_timesteps-n_lags+1)
    tmin, tmax : int
        Subset of timesteps to plot for data and H.
    outer_pad : float
        Fraction of figure space to leave for all margins.
    inner_pad : float
        Fraction of figure space to leave for space between subplots
    data_ax_height : float
        Fractional height of main data/prediction plot.
    data_ax_width : float
        Fractional width of main data/prediction plot.
    figsize : tuple
        Specifies (width, height) of figure.

    Returns
    -------
    fig : Figure
        Figure instance.
    w_ax : list of matplotlib Axes
        Axes holding the model motifs / sequences.
    h_ax : ndarray
        Axes holding the temporal factors.
    """

    # Truncate data and H to desired window.
    data = data[:, tmin:tmax]
    H = H[:, tmin:tmax]
    num_components = H.shape[0]

    # Layout parameters for figure.
    h_ax_height = 1 - data_ax_height
    w_ax_width = 1 - data_ax_width
    pad = inner_pad + outer_pad

    # Create figure and axes for plotting data.
    fig = plt.figure(figsize=figsize)
    data_ax_pos = {
        "left": w_ax_width + inner_pad * w_ax_width,
        "bottom": outer_pad,
        "right": 1.0 - outer_pad,
        "top": data_ax_height - inner_pad * h_ax_height,
    }
    data_ax = plt.subplot(GridSpec(1, 1, **data_ax_pos)[0])

    data_ax.set_xticks([])
    data_ax.set_yticks([])

    # Set up axes for visualizing model motifs.
    w_ax = []
    w_ax_pos = {
        "left": outer_pad,
        "bottom": outer_pad,
        "right": w_ax_width,
        "top": data_ax_height - inner_pad * h_ax_height,
        "wspace": inner_pad,
    }
    for gs in GridSpec(1, num_components, **w_ax_pos):
        w_ax.append(plt.subplot(gs))

    # for ax in w_ax[1:]:
    #     ax.get_shared_x_axes().join(w_ax[0], ax)
    #     ax.get_shared_y_axes().join(w_ax[0], ax)
    for ax in w_ax:
        ax.set_yticks([])
        ax.set_xticks([])

    # Set up axes for visualizing motif times.
    h_ax = []
    h_ax_pos = {
        "left": w_ax_width + inner_pad * w_ax_width,
        "bottom": data_ax_height,
        "right": 1 - outer_pad,
        "top": 1 - outer_pad,
        "hspace": inner_pad,
    }
    for gs in GridSpec(num_components, 1, **h_ax_pos):
        h_ax.append(plt.subplot(gs))

    # for ax in h_ax[1:]:
    #     ax.get_shared_x_axes().join(w_ax[0], ax)
    #     ax.get_shared_y_axes().join(w_ax[0], ax)
    for ax in h_ax:
        ax.set_yticks([])
        ax.set_xticks([])

    # Plot data
    data_ax.imshow(data, aspect='auto')

    # Plot timing factors.
    for ax, h in zip(h_ax, H):
        ax.plot(h)
        ax.set_xlim([0, len(h)])
        ax.axis('off')

    # Plot motifs.
    for ax, w in zip(w_ax, torch.swapaxes(W, 0, 1)):
        ax.imshow(w, aspect='auto')

    return fig, w_ax, h_ax, data_ax
