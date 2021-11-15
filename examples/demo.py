import torch
from convnmf.datasets import synthetic_sequences
from convnmf import plot_convnmf, fit_convnmf

import matplotlib.pyplot as plt


params = {
    "n_components": 3,
    "n_features": 100,
    "n_lags": 50,
    "n_timebins": 300
}
data, W_true, H_true = synthetic_sequences(**params)
fig, w_ax, h_ax, data_ax = plot_convnmf(data, W_true, H_true)
plt.show()

model = fit_convnmf(
    data,
    n_components=params["n_components"],
    n_lags=params["n_lags"],
    loss="quadratic",
    tol=1e-3,
    max_iters=1000
)

fig, w_ax, h_ax, data_ax = plot_convnmf(
    data,
    model.W.detach(),
    model.H.detach()
)
plt.show()
