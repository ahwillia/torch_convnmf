from convnmf.datasets import synthetic_sequences
from convnmf import fit_convnmf

import matplotlib.pyplot as plt


params = {
	"n_components": 3,
	"n_features": 100,
	"n_lags": 50,
	"n_timebins": 300
}
data = synthetic_sequences(**params)

model, optimizer = fit_convnmf(
	data,
	n_components=params["n_components"],
	n_lags=params["n_lags"],
	loss="poisson",
	tol=1e-3
)

# result.plot()

