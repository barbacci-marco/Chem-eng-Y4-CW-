import numpy as np
import torch
from torch.optim import Adam
from torch.quasirandom import SobolEngine

from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


def fit_gpytorch_model(mll, lr=0.001, max_iter=50):
    """
    Manual training loop for a GPyTorch Marginal Log Likelihood (mll).
    """
    model = mll.model
    model.train()
    mll.train()

    optimizer = Adam(model.parameters(), lr=lr)
    for _ in range(max_iter):
        optimizer.zero_grad()
        output = model(model.train_inputs[0])
        loss = -mll(output, model.train_targets)  # negative log likelihood
        loss.backward()
        optimizer.step()

    return mll


def bayesian_optimization_wrapper2(
    f,
    xdim,
    bounds,
    n_init=5,
    num_iterations=10,
    beta=2.0,
    device="cpu"
):
    """"
    Parameters
    ----------
    f : object
        Has method f.fun_test(x: np.ndarray) -> float
    xdim : int
        Dimension of the input space.
    bounds : np.ndarray or list
        Shape (xdim, 2). Each row is [lower_bound, upper_bound] for that dimension.
    n_init : int
        Number of Sobol points to sample initially.
    num_iterations : int
        Number of Bayesian optimization iterations (each iteration adds 1 new point).
    beta : float
        UCB exploration parameter (higher => more exploration).
    device : str
        "cpu" or "cuda".

    Returns
    -------
    best_point : torch.Tensor (1, xdim)
    best_value : float
    team_names : list of str
    cids : list of str
    """

    # 1) Convert bounds (xdim,2)->(2,xdim) for BoTorch, and move to Torch
    bounds_np = np.array(bounds, dtype=float)
    assert bounds_np.shape == (xdim, 2), f"Bounds must be (xdim,2), got {bounds_np.shape}"

    # [lb, ub] so the shape is (2, xdim)
    bounds_torch = torch.tensor(bounds_np.T, dtype=torch.float32, device=device)

    # 2) Sobol initialization in [0,1]^xdim, then scale to [lb_i, ub_i]
    sobol_engine = SobolEngine(dimension=xdim, scramble=True)
    sobol_samples = sobol_engine.draw(n_init).to(device)  # shape (n_init, xdim) in [0,1]

    lower = bounds_torch[0]  # shape (xdim,)
    upper = bounds_torch[1]
    widths = upper - lower   # shape (xdim,)

    
    X_list = []
    y_list = []

    def evaluate_fun(x_np: np.ndarray) -> float:
        
        return f.fun_test(x_np)  # returns float

    # Evaluating the initial Sobol points
    for i in range(n_init):
       
        x_scaled = lower + widths * sobol_samples[i]
        # Convert to NumPy for objective function
        x_np = x_scaled.detach().cpu().numpy()
        y_val = evaluate_fun(x_np)
        X_list.append(x_scaled.view(1, -1))  # shape (1, xdim)
        y_list.append(torch.tensor([[y_val]], dtype=torch.float32, device=device))

    X_train = torch.cat(X_list, dim=0)  # shape (n_init, xdim)
    y_train = torch.cat(y_list, dim=0)  # shape (n_init, 1)

    # 3) BO Loop
    for iteration in range(num_iterations):
        # Build & train GP
        model = SingleTaskGP(X_train, y_train)

        
        if hasattr(model.covar_module, "initialize"):
            # If it's an RBFKernel, you can set lengthscale:
            model.covar_module.initialize(lengthscale=1.0)
        if hasattr(model.likelihood.noise_covar, "initialize"):
            model.likelihood.noise_covar.initialize(noise=1e-2)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, lr=0.001)

        # UCB acquisition function
        UCB = UpperConfidenceBound(model=model, beta=beta)

        # Optimize acquisition in [lower, upper]
        candidate, acq_value = optimize_acqf(
            acq_function=UCB,
            bounds=bounds_torch,  # shape (2, xdim)
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        # Evaluate new candidate
        candidate_np = candidate.squeeze(0).detach().cpu().numpy()
        y_new_val = evaluate_fun(candidate_np)

        # Add to training
        X_train = torch.cat([X_train, candidate.view(1, -1)], dim=0)
        y_train = torch.cat(
            [y_train, torch.tensor([[y_new_val]], dtype=torch.float32, device=device)],
            dim=0
        )
    #Best point
    best_val, best_idx = y_train.min(dim=0)
    best_point = X_train[best_idx]

    
    team_names = ["7", "8"]
    cids = ["01234567"]

    return best_point, best_val.item(), team_names, cids
