import torch
from torch.optim import SGD

from convnmf.model import ConvNMF


def fit_convnmf(
        data, *, n_components, n_lags, loss, tol=1e-3,
        patience=10, init_lr=1e-2, momentum=0.0,
        backtrack_factor=0.5, max_iters=100,
        verbosity=1
    ):

    # Construct model.
    model = ConvNMF(data, n_components, n_lags, loss)

    # Keep track of best parameters and best loss.
    best_params = (
        model._W_pre.detach().clone(),
        model._H_pre.detach().clone()
    )
    best_loss = model()
    
    # Do a backwards pass to determine norms of gradients.
    best_loss.backward()

    optimizer = SGD([
        {
            "params": [model._W_pre],
            "lr": init_lr / torch.norm(model._W_pre.grad.data)
        },
        {
            "params": [model._H_pre],
            "lr": init_lr / torch.norm(model._W_pre.grad.data)
        }
    ], momentum=momentum)

    best_loss = best_loss.item()
    losses = [best_loss]
    loss_decrease_counter = patience
    convergence_counter = 0
    itercount = 0
    upcount = 0
    loss_has_not_changed = True
    effective_lr = init_lr

    while convergence_counter < patience:

        # Compute loss and gradients.
        optimizer.zero_grad()
        _loss = model()
        _loss.backward()
        itercount += 1

        # Extract float value of loss.
        loss = _loss.item()

        if verbosity > 0:
            print(
                f"ITER: {itercount:4g}" +
                (" " * 10) +
                f"LOSS: {loss:8.4f}" +
                (" " * 10) +
                # f"BEST: {best_loss:8.4f}" +
                # (" " * 10) +
                f"LR: {effective_lr:10.4f}" +
                (" " * 10) +
                f"CVRG : {convergence_counter}"
            )

        # Count number of times the loss changes
        if (upcount > 0) and (abs(loss - best_loss) < (tol * abs(best_loss))):
            convergence_counter += 1
        else:
            convergence_counter = 0

        # If loss has gone down.
        if loss <= best_loss:

            # Save parameters.
            losses.append(loss)
            best_loss = loss
            best_params = (
                model._W_pre.detach().clone(),
                model._H_pre.detach().clone()
            )

            # Increase learning rate
            effective_lr /= (backtrack_factor) ** (0.5 ** upcount)
            for group in optimizer.param_groups:
                group["lr"] /= (backtrack_factor) ** (0.5 ** upcount)

            # Take another parameter step.
            optimizer.step()

        # If loss has gone up.
        else:

            # Reset parameters.
            model._W_pre.data = best_params[0]
            model._H_pre.data = best_params[1]

            # Decrease learning rate.
            effective_lr *= backtrack_factor
            for group in optimizer.param_groups:
                group["lr"] *= backtrack_factor

            # Count number of upticks.
            upcount += 1
            convergence_counter = 0

    return model
