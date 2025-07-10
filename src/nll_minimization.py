import os
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from tqdm import tqdm

from scipy.optimize import minimize, root_scalar



sigmoid = lambda x: 1. / (1 + np.exp(-beta * x))
log_sigmoid = lambda x: -np.logaddexp(0, -beta * x)


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


def generate_dynamics(N, T, G, mu=0.5, eps=.25, beta=50, x0=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if x0 is None:
        x0 = np.random.uniform(size=N) * 2 - 1
    
    u_v_t_w = []
    
    t = 0
    X = [x0]
    for i, j, t in G:
        if t >= T:
            break
        xt = X[-1]
        xtp1 = xt.copy()
        dist = np.abs(xt[i] - xt[j])
        p = sigmoid(eps-dist)
        extraction = np.random.uniform()
        if extraction<=p:
            xtp1[i] += mu * (xt[j] - xt[i])
            xtp1[j] += mu * (xt[i] - xt[j])
            u_v_t_w.append( (i, j, t, 1) )
        else:
            u_v_t_w.append( (i, j, t, 0) )
        xtp1 = np.clip(xtp1, -1, 1)
    
        X.append(xtp1)
    X = np.vstack(X)
    
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


beta = 60
N = 100

for mu in [0.01, 0.05, 0.1, 0.2, 0.4]:
    for eps in [0.1, 0.25, 0.5, 0.75]:
        initial_seed = 0
        
        results = []
        ll_profile = []
        for seed in range(10):
            for T in [100., 600, 1200., 2300.,  3400.,  4500.,  5600., 6700.,  7800., 8900., 10000.]: 
                np.random.seed(seed)
                G = generate_extraction_graph(N, int(T), seed=seed)
                x0 = np.random.uniform(size=int(N)) * 2 - 1

                trial = 0
                trial_seed = np.random.randint(low=0, high=32000)
                u_v_t_w, X = generate_dynamics(N, T, G, mu=mu, eps=eps, beta=beta, x0=x0, seed=trial_seed)
                u_v_t_w = np.array(u_v_t_w)

                eps_hat = eps
                for mu_hat in np.linspace(0., 0.5, 501):
                    nll = neg_log_likelihood((eps_hat, mu_hat))
                    ll_profile.append([N, T, seed, eps_hat, mu_hat, nll])

                initial_guess = np.random.uniform(low=0.1, high=.5, size=2)
                nll_minimum = minimize(neg_log_likelihood, 
                                   x0=initial_guess,
                                   bounds=[(0., 2.), (0., 0.5)], 
                                   method='Nelder-Mead'
                                  )

                print(N, T, seed, neg_log_likelihood((eps, mu)), neg_log_likelihood(nll_minimum.x), nll_minimum.x)

                results.append([N, T, seed, trial, trial_seed, nll_minimum.x[0], nll_minimum.x[1]])
                

            np.savetxt(f"../outputs/nll_minimization/{eps}_{mu}_{beta}_{N}_{X.shape[0]-1}_{seed}_trajectory.csv", 
                       X, 
                       delimiter=','
                      )

        results = np.array(results)
        df = pd.DataFrame(results, columns=['N', 'T', 'seed', 'trial', 'trial_seed', 'eps_hat', 'mu_hat'])
        df.to_csv(f"../outputs/nll_minimization/{eps}_{mu}_{beta}_{N}_{X.shape[0]}_results.csv")

        ll_profile = np.array(ll_profile)
        ll_df = pd.DataFrame(ll_profile, columns=['N', 'T', 'seed', 'eps', 'mu', 'nll'])
        ll_df.to_csv(
            f"../outputs/nll_minimization/{eps}_{mu}_{beta}_{N}_{X.shape[0]-1}_ll_profile.csv")