import numpy as np

# -------------------------------
# 1. Kernel Functions
# -------------------------------
def rbf_kernel(X1, X2, length_scale):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
    return np.exp(-0.5 / (length_scale**2) * sqdist)

def matern_kernel(X1, X2, length_scale, nu=2.5):
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
        raise ValueError("Unsupported nu value (use nu=1.5 or nu=2.5).")

# -------------------------------
# 2. Standard Normal PDF/CDF
# -------------------------------
def erf(z):
    # Approximation of the error function
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

def standard_normal_pdf(z):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)

def standard_normal_cdf(z):
    return 0.5 * (1 + erf(z / np.sqrt(2.0)))

# -------------------------------
# 3. GP Posterior
# -------------------------------
def gp_posterior(X_train, y_train, X_test,
                 alpha, length_scale, kernel_type='RBF', nu=2.5):
    """
    Gaussian Process posterior using RBF or Matern kernel.
    """
    if kernel_type == 'RBF':
        kernel = rbf_kernel
    elif kernel_type == 'Matern':
        kernel = lambda A, B, l: matern_kernel(A, B, l, nu=nu)
    else:
        raise ValueError("kernel_type must be 'RBF' or 'Matern'.")

    # Build covariances
    K    = kernel(X_train, X_train, length_scale) + alpha**2 * np.eye(len(X_train))
    K_s  = kernel(X_train, X_test,  length_scale)
    K_ss = kernel(X_test,  X_test,  length_scale) + alpha**2 * np.eye(len(X_test))

    # Jitter
    K += 1e-9 * np.eye(len(K))

    # Solve K alpha = y
    L         = np.linalg.cholesky(K)
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Posterior mean
    mu_s = K_s.T.dot(alpha_vec)
    
    # Posterior cov
    v     = np.linalg.solve(L, K_s)
    cov_s = K_ss - v.T.dot(v)

    return mu_s, cov_s

# -------------------------------
# 4. Vectorized EI
# -------------------------------
def expected_improvement_vectorized(mu, sigma, f_best, xi):
    """
    Vectorized EI for minimization problem.
    mu, sigma: arrays of shape (N,)
    """
    sigma = np.maximum(sigma, 1e-12)
    Z = (mu - f_best - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    ei = (mu - f_best - xi) * Phi + sigma * phi
    return ei

# -------------------------------
# 5. GA to Maximize Acquisition
# -------------------------------
def ga_acquisition_maximization(
    X_train, y_train,
    bounds,
    alpha,
    length_scale,
    xi,
    kernel_type='RBF',
    nu=2.5,
    pop_size=20,       # population size
    n_gens=10,         # number of generations
    crossover_prob=0.7,
    mutation_prob=0.1
):
    """
    Uses a Genetic Algorithm to maximize EI across 'bounds'.
    Returns the single best point found by the GA.
    
    Workflow:
      1) Initialize population of random points in 'bounds'.
      2) For each generation:
         - Evaluate acquisition (EI) for each individual in the population.
         - Select parents, do crossover & mutation, form new population.
      3) Return the best point in the final population.
    """

    d = bounds.shape[0]

    # Initialize random population
    pop = np.zeros((pop_size, d))
    for i in range(d):
        lb, ub = bounds[i]
        pop[:, i] = np.random.rand(pop_size)*(ub - lb) + lb

    # Evaluate function that GA is maximizing (EI)  
    def fitness(pop_points):
        # pop_points: shape (pop_size, d)
        # 1) GP posterior at pop_points
        mu, cov = gp_posterior(
            X_train, y_train,
            pop_points,
            alpha         = alpha,
            length_scale  = length_scale,
            kernel_type   = kernel_type,
            nu            = nu
        )
        sigma = np.sqrt(np.diag(cov))
        f_best = np.min(y_train)
        # 2) Vectorized EI
        ei_vals = expected_improvement_vectorized(mu, sigma, f_best, xi)
        return ei_vals  # shape (pop_size,)

    for gen in range(n_gens):
        # Evaluate population
        fitness_vals = fitness(pop)  # shape (pop_size,)

        # ---- Selection (tournament or rank-based, simplified here) ----
        # We'll do a simple "tournament of size 2" for demonstration
        new_pop = []
        for _ in range(pop_size // 2):
            # pick 2 random individuals, pick the best as parent1
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent1 = pop[idx1] if fitness_vals[idx1] > fitness_vals[idx2] else pop[idx2]

            # pick 2 more
            idx3, idx4 = np.random.choice(pop_size, 2, replace=False)
            parent2 = pop[idx3] if fitness_vals[idx3] > fitness_vals[idx4] else pop[idx4]

            # ---- Crossover ----
            child1 = parent1.copy()
            child2 = parent2.copy()
            if np.random.rand() < crossover_prob:
                # single-point crossover (for demonstration)
                cp = np.random.randint(1, d)  # crossover point
                child1[cp:], child2[cp:] = child2[cp:].copy(), child1[cp:].copy()

            # ---- Mutation ----
            for child in (child1, child2):
                if np.random.rand() < mutation_prob:
                    # pick dimension to mutate
                    dim = np.random.randint(d)
                    lb, ub = bounds[dim]
                    child[dim] = np.random.rand()*(ub - lb) + lb

            new_pop += [child1, child2]

        pop = np.array(new_pop)
        # If pop_size is odd, we might have one leftover—handle if needed.

    # Final evaluation
    final_fitness = fitness(pop)
    idx_best = np.argmax(final_fitness)
    return pop[idx_best]

# -------------------------------
# 6. GP + GA-based Acquisition Optimizer
# -------------------------------
def gp_ga_optimizer(
    sample_loss,
    bounds,
    n_iters=10,
    n_pre_samples=2,
    alpha=1e-8,
    length_scale=1.0,
    xi=0.01,
    kernel_type='RBF',
    nu=2.5,
    pop_size=20,
    n_gens=10,
    X_init=None,
    y_init=None
):
    """
    Minimizes sample_loss using:
      - Gaussian Process
      - Expected Improvement
      - Genetic Algorithm to maximize EI each iteration.

    Each iteration:
      1) Build GP on (X_train, y_train).
      2) GA finds X_next that maximizes EI.
      3) Evaluate sample_loss(X_next), update data.
    """
    bounds = np.array(bounds)
    d = bounds.shape[0]

    # ---------------- Initialize Data ----------------
    if X_init is not None and y_init is not None:
        X_train = X_init
        y_train = y_init
    else:
        # Random initial points
        X_train = np.zeros((n_pre_samples, d))
        y_train = np.zeros(n_pre_samples)
        for i in range(n_pre_samples):
            x_rand = np.random.rand(d)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            X_train[i] = x_rand
            y_train[i] = sample_loss(x_rand)

    # ---------------- Main Loop ----------------
    for _ in range(n_iters):
        # Use GA to find next point that maximizes EI
        X_next = ga_acquisition_maximization(
            X_train, y_train,
            bounds        = bounds,
            alpha         = alpha,
            length_scale  = length_scale,
            xi            = xi,
            kernel_type   = kernel_type,
            nu            = nu,
            pop_size      = pop_size,
            n_gens        = n_gens
        )
        y_next = sample_loss(X_next)

        # Append to dataset
        X_train = np.vstack((X_train, X_next))
        y_train = np.append(y_train, y_next)

    # ------------- Return Best Solution -------------
    best_idx = np.argmin(y_train)
    best_x   = X_train[best_idx]
    best_f   = y_train[best_idx]
    return best_x, best_f

# -------------------------------
# 7. (Optional) Random Search Warm Start
# -------------------------------
def Random_search(f, n_p, bounds_rs, iter_rs):
    """
    Simple random search that samples 'iter_rs' points and returns best.
    """
    localx   = np.zeros((iter_rs, n_p))
    localval = np.zeros(iter_rs)
    b_range  = bounds_rs[:,1] - bounds_rs[:,0]
    b_bias   = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.rand(n_p)*b_range + b_bias
        localx[sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial)

    idx_min = np.argmin(localval)
    f_b = localval[idx_min]
    x_b = localx[idx_min]
    return f_b, x_b, localx, localval

# -------------------------------
# 8. Top-level "my_algorithm"
# -------------------------------
def opt_GP(f, x_dim, bounds, iter_tot,
                 kernel_type='RBF', nu=2.5):
    """
    1) Uses random search for part of the budget (if desired).
    2) Then runs GP + GA-based acquisition for the remainder.
    """
    bounds = np.array(bounds)

    # Decide how many evals for random search
    n_rs = int(min(100, max(iter_tot * 0.2, 5)))  # e.g. 20% for random search
    if n_rs > iter_tot:
        n_rs = iter_tot

    # ---- Random Search phase ----
    f_b, x_b, X_rs, Y_rs = Random_search(f, x_dim, bounds, n_rs)

    # Remaining for GP+GA
    ga_iters = iter_tot - n_rs
    if ga_iters < 0:
        ga_iters = 0

    sample_loss = lambda x: f.fun_test(x)

    # ---- GP + GA-based BO ----
    x_opt, f_opt = gp_ga_optimizer(
        sample_loss   = sample_loss,
        bounds        = bounds,
        n_iters       = ga_iters,
        n_pre_samples = 0,      # we have data
        alpha         = 1e-6,
        length_scale  = 0.5,
        xi            = 0.08,
        kernel_type   = kernel_type,
        nu            = nu,
        pop_size      = 10,
        n_gens        = 3,
        X_init        = X_rs,
        y_init        = Y_rs
    )

    team_names = ["Genetic", "Bayes"]
    cids       = ["12345678"]
    return x_opt, f_opt, team_names, cids
