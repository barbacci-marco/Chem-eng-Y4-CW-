import numpy as np
import math

# --------------------------------
# Kernel Functions
# --------------------------------
def rbf_kernel(X1, X2, length_scale):
    """
    Radial Basis Function (RBF) kernel.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
    return np.exp(-0.5 / (length_scale**2) * sqdist)

def matern_kernel(X1, X2, length_scale, nu=2.5):
    """
    Matérn kernel. Supports nu = 1.5 (less smooth) or nu = 2.5 (smoother).
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dist = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))
    
    if nu == 1.5:
        scale = 1.0 + (np.sqrt(3) * dist) / length_scale
        return scale * np.exp(-np.sqrt(3) * dist / length_scale)
    elif nu == 2.5:
        scale = 1.0 + (np.sqrt(5) * dist) / length_scale + (5 * dist**2) / (3 * length_scale**2)
        return scale * np.exp(-np.sqrt(5) * dist / length_scale)
    else:
        raise ValueError("Unsupported nu value. Use nu=1.5 or nu=2.5.")


# Standard Normal PDF
def standard_normal_pdf(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

# Error Function Approximation
def erf(z):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911
    
    sign = np.sign(z)
    z = np.abs(z)
    
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    return sign * y

# Standard Normal CDF
def standard_normal_cdf(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))

# --------------------------------
# GP Posterior with Kernel Selection
# --------------------------------
def gp_posterior(X_train, y_train, X_test, alpha, length_scale, kernel_type='Matern', nu=2.5):
    """
    Gaussian Process posterior using a selected kernel.
    """
    if kernel_type == 'RBF':
        kernel = rbf_kernel
    elif kernel_type == 'Matern':
        kernel = lambda X1, X2, l: matern_kernel(X1, X2, l, nu=nu)
    else:
        raise ValueError("Invalid kernel_type. Use 'RBF' or 'Matern'.")

    # Kernel computations
    K = kernel(X_train, X_train, length_scale) + alpha**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_test, length_scale)
    K_ss = kernel(X_test, X_test, length_scale) + alpha**2 * np.eye(len(X_test))
    
    # Jitter for numerical stability
    K += 1e-8 * np.eye(len(K))
    
    # Solve for alpha_vec in K * alpha_vec = y
    L = np.linalg.cholesky(K)
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    # Posterior mean & covariance
    mu_s = K_s.T.dot(alpha_vec)              # shape (n_test,)
    v    = np.linalg.solve(L, K_s)          # shape (n_train, n_test)
    cov_s = K_ss - v.T.dot(v)               # shape (n_test, n_test)
    
    return mu_s.flatten(), cov_s

# Expected Improvement Function
def expected_improvement(X, X_train, y_train, mu, sigma, f_best, xi):
    sigma = np.maximum(sigma, 1e-8)
    Z = (mu - f_best - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    ei = (mu - f_best - xi) * Phi + sigma * phi
    return ei

def initialize_trust_region(bounds, initial_radius):
    trust_region_center = None
    trust_region_radius = initial_radius
    return trust_region_center, trust_region_radius

def update_trust_region(trust_region_center, trust_region_radius, X_new, y_new, y_best, bounds, shrink_factor, expand_factor):
    # If new sample is better than best so far, expand; otherwise, shrink
    if y_new < y_best:
        trust_region_radius *= expand_factor
    else:
        trust_region_radius *= shrink_factor

    # Ensure the radius stays within bounds
    min_radius = 0.01 * np.ptp(bounds, axis=1)
    max_radius = 0.5 * np.ptp(bounds, axis=1)
    trust_region_radius = np.maximum(np.minimum(trust_region_radius, max_radius), min_radius)

    # Update the center to the new point
    trust_region_center = X_new
    return trust_region_center, trust_region_radius

def random_acquisition_maximization(acquisition, X_train, y_train, trust_region_center, trust_region_radius, 
                                    bounds, num_samples, alpha, length_scale, xi):
    # Generate random samples within the trust region
    lower_bounds = np.maximum(trust_region_center - trust_region_radius, bounds[:, 0])
    upper_bounds = np.minimum(trust_region_center + trust_region_radius, bounds[:, 1])
    samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_samples, bounds.shape[0]))
    
    # Compute GP posterior for samples
    mu, cov = gp_posterior(X_train, y_train, samples, alpha=alpha, length_scale=length_scale)
    sigma = np.sqrt(np.diag(cov))
    f_best = np.min(y_train)  # Minimization problem
    
    # Compute acquisition values
    acquisition_values = np.array([
        acquisition(x.reshape(1, -1), X_train, y_train, mu_i, sigma_i, f_best, xi)
        for x, mu_i, sigma_i in zip(samples, mu, sigma)
    ])
    
    # Select the point with the highest acquisition value
    idx_max = np.argmax(acquisition_values)
    X_next = samples[idx_max]
    return X_next
    
# --------------------------------
# GP Trust Region Optimizer
# --------------------------------
def gp_trust_optimizer(
    sample_loss,        # callable: given x -> f(x)
    bounds,             # array shape (d, 2)conda activate
    n_iters=99,         # how many iterations
    n_pre_samples=2,    # random initial points
    alpha=1e-8,          # noise term
    initial_trust_radius=1,
    length_scale=2,
    xi=0.1,            # EI exploration param
    shrink_factor=0.8,
    expand_factor=1.3,
    num_samples=1000,
    kernel_type='Matern',  # Kernel selection ('RBF' or 'Matern')
    nu=2.5              # For Matérn kernel, specifies smoothness
):
    """
    Minimizes sample_loss over 'bounds' using GP + trust region + random acquisition maximization.
    Returns: (best_x, best_f)
    """
    bounds = np.array(bounds)
    d = bounds.shape[0]

    # 1) Initial random points
    X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_pre_samples, d))
    y_train = np.array([sample_loss(x) for x in X_train])
    
    # 2) Find the best among these initial samples
    best_idx = np.argmin(y_train)
    y_best = y_train[best_idx]
    trust_region_center = X_train[best_idx]
    trust_region_radius = initial_trust_radius * np.ptp(bounds, axis=1)  # vector, same dimension
    
    for i in range(n_iters):
        # Acquire next sample
        X_next = random_acquisition_maximization(
            acquisition=expected_improvement,
            X_train=X_train, 
            y_train=y_train,
            trust_region_center=trust_region_center,
            trust_region_radius=trust_region_radius,
            bounds=bounds,
            num_samples=num_samples,
            alpha=alpha,
            length_scale=length_scale,
            xi=xi
        )
        
        # Evaluate at X_next
        y_next = sample_loss(X_next)
        
        # Update training set
        X_train = np.vstack((X_train, X_next))
        y_train = np.append(y_train, y_next)
        
        # Update best
        y_best=np.min(y_train)
        
        # Update trust region
        trust_region_center, trust_region_radius = update_trust_region(
            trust_region_center, 
            trust_region_radius, 
            X_next, 
            y_next, 
            y_best, 
            bounds, 
            shrink_factor, 
            expand_factor
        )
        
    # Return the best found
    best_idx_final = np.argmin(y_train)
    best_x = X_train[best_idx_final]
    best_f = y_train[best_idx_final]
    
    return best_x, best_f

# --------------------------------
# My Algorithm
# --------------------------------
def my_algorithm(f, x_dim, bounds, iter_tot, kernel_type='Matern', nu=2.5):
    """
    Optimizer using GP + trust region with selectable kernel type.
    """
    sample_loss = lambda x: f.fun_test(x)

    x_opt, f_opt = gp_trust_optimizer(
        sample_loss=sample_loss,
        bounds=bounds,
        n_iters=iter_tot,
        n_pre_samples=2,
        alpha=1e-8,
        initial_trust_radius=5,
        length_scale=0.5,
        xi=0.01,
        shrink_factor=0.8,
        expand_factor=1.3,
        num_samples=100000,
        kernel_type=kernel_type,  # Select kernel type
        nu=nu                    # Matérn smoothness parameter
    )

    team_names = ["7", "8"]
    cids = ["01234567"]
    return x_opt, f_opt, team_names, cids
