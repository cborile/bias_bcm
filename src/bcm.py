import numpy as np
from scipy.special import expit as sigmoid

def generate_extraction_graph(N, T, seed=None):
    if seed is not None:
        np.random.seed(seed)

    G = []
    for t in range(T):
        i = np.random.randint(N)
        while True:
            j = np.random.randint(N)
            if i != j: break
        G.append([i, j, t])
    return np.array(G)


def generate_dynamics(N, T, G, mu=0.5, eps=.25, beta=50, x0=None, seed=None, return_dist = False):
    if seed is not None:
        np.random.seed(seed)

    if x0 is None:
        x0 = np.random.uniform(size=N) * 2 - 1

    u_v_t_w = []
    dist_list = []
    s = []

    t = 0
    X = [x0]
    for i, j, t in G:
        if t >= T:
            break
        xt = X[-1]
        xtp1 = xt.copy()
        dist = np.abs(xt[i] - xt[j])
        dist_list.append(dist)
        p = sigmoid(beta * (eps-dist))
        extraction = np.random.uniform()
        if extraction<=p:
            xtp1[i] += mu * (xt[j] - xt[i])
            xtp1[j] += mu * (xt[i] - xt[j])
            s.append(1)
            u_v_t_w.append( (i, j, t, 1) )
        else:
            s.append(0)
            u_v_t_w.append( (i, j, t, 0) )
        xtp1 = np.clip(xtp1, -1, 1)

        X.append(xtp1)
    X = np.vstack(X)
    if return_dist:
        return u_v_t_w, X, np.array(dist_list), np.array(s)
    else:
        return u_v_t_w, X


def neg_log_likelihood(params):
    eps_hat, mu_hat = params
    xt = x0.copy()
    # X = [xt]
    log_likelihood = 0
    for i, j, t, has_interacted_at_t in u_v_t_w:
        dist = np.abs(xt[i] - xt[j])
        if has_interacted_at_t:
            log_sigma = log_sigmoid(eps_hat - dist)  # to avoid overflow
            log_likelihood += log_sigma
            xt[i] += mu_hat * (xt[j] - xt[i])
            xt[j] += mu_hat * (xt[i] - xt[j])
        else:
            log_likelihood -= np.logaddexp(0, beta*(eps_hat - dist))
        # X.append(xt.copy())
    return -log_likelihood
