# this script runs a set of experiments with varying epsilon, T and N
# 1. generate the opinion dynamics trajectories
# 2. compute the MLE 
# 3. compute the theoretical bias
# 4. save the experiments results


import numpy as np
import sys
sys.path += ["../src"]
import experiments_rasch as R
import bcm
import pandas as pd

if __name__ == "__main__":
    rho = 15
    N = 10000
    mu = 0.01

    experiment_errors = []
    
    for seed in range(10000, 20000):
        print("seed", seed)
        
        for i,epsilon_real in enumerate([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]):
            for j,T in enumerate([50,100,500]):
                for k,N in enumerate([100, 500, 1000]): 
                    # 1. generate the opinion dynamics trajectories
                    G = bcm.generate_extraction_graph(N = N, T = T, seed = seed * 1000 + k * 100 + i * 10 + j)
                    e, X, delta_X, s = bcm.generate_dynamics(N = N, T = T, G = G, mu = mu, eps = epsilon_real, 
                                                             beta = rho, seed = seed * 1000 + k * 100 + i * 10 + j,
                                                             return_dist = True)
                    
                    # 2. compute the MLE 
                    compute_loss_eps = lambda eps: R.compute_loss(eps = eps, rho = rho, delta_x = delta_X, s = s)
                    mle = R.minimize(compute_loss_eps, x0 = 0.)["x"][0]
                    # 3. compute the theoretical bias
                    bias = R.compute_bias_(eps = epsilon_real, delta_x = delta_X, rho = rho)
                    var_bias = R.var_bias_(eps = epsilon_real, delta_x = delta_X, rho = rho)

                    exp_error = [epsilon_real, rho, T, N, seed, mle, mle - epsilon_real, bias, var_bias, s.sum()]

                    experiment_errors.append(exp_error)
                    # 4. save the experiments results
                    exp_df = pd.DataFrame(experiment_errors, 
                                          columns = ["epsilon_real", "rho", "T", "N", "seed", "mle", "empirical_error", "theoretical_bias", "var_bias", "pos_interactions"])
                    
                    exp_df.to_csv("/data/opdyn_identification/error_bias_rho_seed/exp_250527_sbcm_000.csv")