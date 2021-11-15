"""
Main model specification.
"""

import torch
import torch.nn.functional as F


def conv_predict(W, H):
    """
    Computes the model's prediction of the multivariate time series.

    Parameters
    ----------
    W : torch.Tensor
        Has shape (n_features, n_components, n_lags).
        Holds motifs / sequences.

    H : torch.Tensor
        Has shape (n_components, n_features, n_timebins + n_lags - 1)
        Holds temporal factors.

    Returns
    -------
    prediction : torch.Tensor
        Has shape (n_features, n_timebins).
    """
    assert W.shape[1] == H.shape[0]
    n_features, n_components, n_lags = W.shape

    return torch.sum(
        F.conv1d(
            W, H[:, None, :],
            padding=(H.shape[-1] - n_lags),
            groups=n_components
        ),
        axis=1
    )


class ConvNMF(torch.nn.Module):
    """
    Convolutional Nonnegative Matrix Factorization (convNMF) model.
    """
    def __init__(self, data, n_components, n_lags, loss):
        super(ConvNMF, self).__init__()
        n_features, n_timebins = data.shape

        if loss not in ("quadratic", "poisson"):
            raise ValueError(
                "String parameter `loss` must be one "
                "of 'poisson' or 'quadratic'."
            )

        # Initialize parameters.
        _W_pre = torch.rand(
            n_features, n_components, n_lags
        )
        _H_pre = torch.rand(
            n_components, n_timebins + n_lags - 1
        )

        # Rescale parameters to roughly match data norm.
        data_norm = torch.norm(data)
        _W_pre *= torch.sqrt(data_norm) / torch.norm(_W_pre)
        _H_pre *= torch.sqrt(data_norm) / torch.norm(_H_pre)

        # Wrap parameters in nn.Parameter to track gradients.
        self._W_pre = torch.nn.Parameter(_W_pre)
        self._H_pre = torch.nn.Parameter(_H_pre)

        # Save metadata.
        self.data = data
        self.n_features = n_features
        self.n_timebins = n_timebins
        self.n_components = n_components
        self.n_lags = n_lags
        self.loss = loss

    @property
    def W(self):
        return F.softplus(self._W_pre)
    
    @property
    def H(self):
        return F.softplus(self._H_pre)

    def forward(self):
        
        # Compute model prediction.
        pred = conv_predict(self.W, self.H)

        # Evaluate loss.
        if self.loss == "quadratic":
            resids = pred - self.data
            sse = torch.inner(resids.ravel(), resids.ravel())
            return sse / (self.n_features * self.n_timebins)
        
        elif self.loss == "poisson":
            return torch.mean(pred - torch.log(pred) * self.data)

        else:
            raise AssertionError("Did not recognize loss.")

