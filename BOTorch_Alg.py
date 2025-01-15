import torch
import numpy as np 
from torch.optim import Adam
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

def Random_search(f, n_p, bounds_rs, iter_rs):
    localx   = np.zeros((iter_rs, n_p))  
    localval = np.zeros(iter_rs)

    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.rand(n_p)*bounds_range + bounds_bias
        localx[sample_i, :] = x_trial
        localval[sample_i]  = f.fun_test(x_trial)

    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[minindex, :]

    return f_b, x_b, localx, localval

def fit_gpytorch_model(mll, lr=0.001, max_iter=50):
    model = mll.model
    model.train()
    mll.train()

    optimizer = Adam(model.parameters(), lr=lr)
    for _ in range(max_iter):
        optimizer.zero_grad()
        output = model(model.train_inputs[0])
        loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()

    return mll

def bayesian_optimization_wrapper(
    f,
    xdim,
    bounds,
    n_init=5,
    num_iterations=100,
    beta=2.0,
    device="cpu",
    n_pre_samples=2,
    X_init=None,
    y_init=None
):
    # Ensure bounds is shape (xdim, 2)
    bounds = np.array(bounds)
    assert bounds.shape == (xdim, 2), f"Bounds must be ({xdim}, 2), got {bounds.shape}"

    def sample_loss(x):
        return f.fun_test(x)

    # Initialize training data
    if X_init is not None and y_init is not None:
        if isinstance(X_init, np.ndarray):
            X_train = torch.tensor(X_init, dtype=torch.float32)
        else:
            X_train = X_init.float()
        if isinstance(y_init, np.ndarray):
            y_train = torch.tensor(y_init, dtype=torch.float32)
        else:
            y_train = y_init.float()

        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)
    else:
        X_list, y_list = [], []
        for i in range(n_init):
            x_rand_np = np.random.rand(xdim)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            y_val = sample_loss(x_rand_np)
            X_list.append(x_rand_np)
            y_list.append(y_val)

        X_train = torch.tensor(X_list, dtype=torch.float32)
        y_train = torch.tensor(y_list, dtype=torch.float32).unsqueeze(-1)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Convert (xdim,2) -> (2,xdim) for BoTorch
    bounds_torch = torch.tensor(bounds.T, dtype=torch.float32, device=device)
    lower = bounds_torch[0]
    upper = bounds_torch[1]

    def clamp_candidate(cand: torch.Tensor) -> torch.Tensor:
        return torch.min(torch.max(cand, lower), upper)

    for iteration in range(num_iterations):
        # Fit GP
        model = SingleTaskGP(X_train, y_train)
        
       
        model.covar_module.initialize(lengthscale=1.0)
        
        # Noise
        model.likelihood.noise_covar.initialize(noise=1e-6)
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, lr=0.001)

        # UCB acquisition
        UCB = UpperConfidenceBound(model=model, beta=beta)

        candidate_torch, acq_value = optimize_acqf(
            acq_function=UCB,
            bounds=bounds_torch,  
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        candidate_torch = candidate_torch.squeeze(0)
        candidate_torch = clamp_candidate(candidate_torch)

        candidate_np = candidate_torch.detach().cpu().numpy()
        y_new_val = sample_loss(candidate_np)

        X_train = torch.cat([X_train, candidate_torch.view(1, -1)], dim=0)
        y_train = torch.cat(
            [y_train, torch.tensor([[y_new_val]], dtype=torch.float32, device=device)],
            dim=0
        )

    best_val, best_idx = y_train.min(dim=0)
    best_point = X_train[best_idx]

    team_names = ["7", "8"]
    cids = ["01234567"]
    return best_point, best_val.item(), team_names, cids


