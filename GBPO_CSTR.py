import numpy as np


# -------------------------------
# 1. Kernel Functions
# -------------------------------
def rbf_kernel(X1, X2, length_scale):
    """
    Radial Basis Function (RBF) kernel.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
    return np.exp(-0.5 / (length_scale**2) * sqdist)

def matern_kernel(X1, X2, length_scale, nu=2.5):
    """
    Matérn kernel. Supports nu=1.5 or nu=2.5.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dist = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2))
    
    if nu == 1.5:
        scale = 1.0 + (np.sqrt(3) * dist)/length_scale
        return scale * np.exp(-np.sqrt(3)*dist / length_scale)
    elif nu == 2.5:
        scale = 1.0 + (np.sqrt(5) * dist)/length_scale + (5*dist**2)/(3*(length_scale**2))
        return scale * np.exp(-np.sqrt(5)*dist / length_scale)
    else:
        raise ValueError("Unsupported nu value. Use nu=1.5 or nu=2.5.")

# -------------------------------
# 2. Standard Normal PDF/CDF
# -------------------------------
def standard_normal_pdf(z):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)

def erf(z):
    """
    Approximation of the error function.
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911
    
    sign = np.sign(z)
    z = np.abs(z)
    
    t = 1.0 / (1.0 + p*z)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * np.exp(-z*z)
    return sign * y

def standard_normal_cdf(z):
    return 0.5 * (1 + erf(z / np.sqrt(2.0)))

# -------------------------------
# 3. GP Posterior
# -------------------------------
def gp_posterior(X_train, y_train, X_test,
                 alpha, length_scale, kernel_type='Matern', nu=2.5):
    """
    Gaussian Process posterior using a chosen kernel (RBF or Matern).
    """
    if kernel_type == 'RBF':
        kernel = rbf_kernel
    elif kernel_type == 'Matern':
        kernel = lambda A, B, l: matern_kernel(A, B, l, nu=nu)
    else:
        raise ValueError("kernel_type must be 'RBF' or 'Matern'.")

    # Build the covariance matrices
    K    = kernel(X_train, X_train, length_scale) + alpha**2 * np.eye(len(X_train))
    K_s  = kernel(X_train, X_test,  length_scale)
    K_ss = kernel(X_test,  X_test,  length_scale) + alpha**2 * np.eye(len(X_test))

    # Add small jitter for numerical stability
    K += 1e-9 * np.eye(len(K))

    # Solve for alpha_vec
    L         = np.linalg.cholesky(K)
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Mean
    mu_s = K_s.T.dot(alpha_vec)  # shape (len(X_test),)
    
    # Covariance
    v     = np.linalg.solve(L, K_s)        # shape (n_train, n_test)
    cov_s = K_ss - v.T.dot(v)              # shape (n_test, n_test)

    return mu_s.flatten(), cov_s

# -------------------------------
# 4. Expected Improvement
# -------------------------------
def expected_improvement(X, X_train, y_train, mu, sigma, f_best, xi):
    """
    EI for minimization problem with best value 'f_best'.
    """
    sigma = np.maximum(sigma, 1e-12)
    Z = (mu - f_best - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    ei = (mu - f_best - xi)*Phi + sigma*phi
    return ei

# -------------------------------
# 5. Global Acquisition Maximization
#    (No Trust Region)
# -------------------------------
def random_acquisition_global(
    acquisition,
    X_train, y_train,
    bounds,
    num_samples,
    alpha,
    length_scale,
    xi,
    kernel_type='RBF',
    nu=2.5
):
    """
    1) Generate 'num_samples' random points across the entire domain.
    2) Compute GP posterior -> mean, variance at those points.
    3) Evaluate the acquisition (EI) at each point.
    4) Return the point with the highest acquisition.
    """
    d = bounds.shape[0]
    # Sample uniformly in [lb, ub] for each dim
    random_points = np.zeros((num_samples, d))
    for i in range(d):
        lb, ub = bounds[i]
        random_points[:, i] = np.random.rand(num_samples)*(ub - lb) + lb

    # Posterior
    mu, cov = gp_posterior(X_train, y_train, random_points,
                           alpha=alpha, length_scale=length_scale,
                           kernel_type=kernel_type, nu=nu)
    sigma = np.sqrt(np.diag(cov))
    f_best = np.min(y_train)  # Minimization

    # EI at each random point
    acq_values = np.array([
        acquisition(x.reshape(1, -1),
                    X_train, y_train,
                    mu_i, sigma_i,
                    f_best, xi)
        for x, mu_i, sigma_i in zip(random_points, mu, sigma)
    ])
    
    # Pick best
    idx_best = np.argmax(acq_values)
    return random_points[idx_best]

# -------------------------------
# 6. Simple Random Search
# -------------------------------

def Random_search(f, n_p, bounds_rs, iter_rs):
    """
    This function is a naive optimization routine that randomly samples the
    allowed space and returns the best value & best point.

    Parameters
    ----------
    f         : an object or function that has 'fun_test(x)' returning f(x).
    n_p       : int, dimension of the input space.
    bounds_rs : array shape (n_p, 2), [lower_bound, upper_bound] for each dimension.
    iter_rs   : int, number of random points to create.

    Returns
    -------
    f_b    : float
        The minimum function value found among the samples.
    x_b    : array of shape (n_p,)
        The point that yielded the best value.
    localx : array of shape (iter_rs, n_p)
        All sampled points.
    localval : array of shape (iter_rs,)
        Function values at the sampled points.
    """
    # arrays to store sampled points & function values
    localx   = np.zeros((iter_rs, n_p))  
    localval = np.zeros(iter_rs)

    # bounds
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        # uniform sample in [lower_bound, upper_bound] for each dimension
        x_trial = np.random.uniform(0, 1, n_p) * bounds_range + bounds_bias
        localx[sample_i, :] = x_trial
        localval[sample_i]  = f.fun_test(x_trial)

    # choosing the best among the sampled points
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[minindex, :]

    return f_b, x_b, localx, localval

# -------------------------------
# 7. GP + Global Random Acquisition Optimizer
# -------------------------------
def gp_random_optimizer(
    sample_loss,  
    bounds,       
    n_iters=50,    
    n_pre_samples=2,
    alpha=1e-8,
    length_scale=1.0,
    xi=0.01,
    num_samples=100,  # how many random points to sample each iteration
    kernel_type='RBF',
    nu=2.5,
    X_init=None,  # optional pre-loaded data
    y_init=None,
):
    """
    Minimizes sample_loss over 'bounds' using:
    - Gaussian Process
    - Expected Improvement
    - 'random_acquisition_global' across entire domain
    """
    bounds = np.array(bounds)
    d = bounds.shape[0]

    # 1) Initialize data
    if X_init is not None and y_init is not None:
        X_train = X_init
        y_train = y_init
    else:
        # random initial points
        X_train = np.zeros((n_pre_samples, d))
        y_train = np.zeros(n_pre_samples)
        for i in range(n_pre_samples):
            x_rand = np.random.rand(d)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            X_train[i] = x_rand
            y_train[i] = sample_loss(x_rand)

    # 2) Main loop
    for i in range(n_iters):
        # Acquire next sample by sampling the entire domain
        X_next = random_acquisition_global(
            acquisition = expected_improvement,
            X_train = X_train,
            y_train = y_train,
            bounds  = bounds,
            num_samples = num_samples,
            alpha   = alpha,
            length_scale = length_scale,
            xi      = xi,
            kernel_type = kernel_type,
            nu      = nu
        )
        y_next = sample_loss(X_next)

        # Update dataset
        X_train = np.vstack([X_train, X_next])
        y_train = np.append(y_train, y_next)

    # 3) Return best
    best_idx = np.argmin(y_train)
    best_x = X_train[best_idx]
    best_f = y_train[best_idx]
    return best_x, best_f

# -------------------------------
# 8. Wrapper Function
# -------------------------------
def opt_GP(f, x_dim, bounds, iter_tot,
                 kernel_type='RBF', nu=2.5):
    """
    1) Uses random search for part of the budget.
    2) Then uses a GP-based random acquisition approach (EI),
       sampling across the entire domain (no trust region).
    """
    bounds = np.array(bounds)
    # Decide how many evaluations for random search
    n_rs = int(min(100, max(iter_tot*0.05, 5)))  # e.g., 5% of total, at least 5, up to 100
    

    # 1) Random search
    x_best_rs, f_best_rs, X_rs, Y_rs = Random_search(
        f, x_dim, bounds, n_rs
    )

    # 3) Remaining budget for GP
    gp_iters = iter_tot - n_rs
    if gp_iters < 0:
        gp_iters = 0

    # We'll define sample_loss as:
    sample_loss = lambda x: f.fun_test(x)

    # 4) Run GP random optimizer
    x_opt, f_opt = gp_random_optimizer(
        sample_loss = sample_loss,
        bounds      = bounds,
        n_iters     = gp_iters,
        n_pre_samples = 0,       
        alpha       = 1e-10,
        length_scale= 0.5,
        xi          = 0.1,
        num_samples = 64,
        kernel_type = kernel_type,
        nu          = nu,
        X_init      = X_rs,
        y_init      = Y_rs
    )

    team_names = ["7", "8"]
    cids = ["01234567"]
    return x_opt, f_opt, team_names, cids

