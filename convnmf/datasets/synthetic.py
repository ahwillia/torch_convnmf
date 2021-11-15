import numpy as np
import torch
from convnmf.model import conv_predict


def synthetic_sequences(
        n_components=3, n_features=100, n_lags=50, n_timebins=10000,
        H_sparsity=0.9, noise_scale=1.0, seed=None, dtype=torch.float32
    ):

        # Generate random convolutional parameters
        rs = np.random.RandomState(seed)
        W = np.zeros((n_features, n_components, n_lags))
        H = rs.rand(n_components, n_timebins + n_lags - 1)

        # Add sparsity to factors
        H *= rs.binomial(1, 1 - H_sparsity, size=H.shape)

        # Add structure to motifs
        for feature, component in enumerate(
                rs.choice(n_components, size=n_features)
            ):
            W[feature, component] += _gauss_plus_delay(n_lags)

        # Determine noise
        noise = noise_scale * rs.rand(n_features, n_timebins)

        # Add noise to model prediction
        data = conv_predict(
            torch.from_numpy(W),
            torch.from_numpy(H)
        ) + torch.from_numpy(noise)

        return data.type(dtype)


def _gauss_plus_delay(n_steps):
    tau = np.random.uniform(-1.5, 1.5)
    x = np.linspace(-3-tau, 3-tau, n_steps)
    y = np.exp(-x**2)
    return y / y.max()
