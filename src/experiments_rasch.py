
# this script computes the bias and the MLE of the Rasch model and sBCM
# by running this script, we do a set of experiments to estimate the parameter of a Rasch model
# with varying rho, epsilon and number of interactions
# the experiment returns the theoretical bias and the empirical error


import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.optimize import minimize

# probability of positive response
compute_p = lambda eps, delta_x, rho: sigmoid(rho * (eps - delta_x))
# loss Rasch
compute_loss = lambda eps, rho, delta_x, s: np.sum((compute_p(eps, delta_x, rho) - s) ** 2)
# explicit negative log likelihood Rasch
compute_neg_log_likelihood = lambda eps, rho, delta_x, s: -np.sum(s * np.log(compute_p(eps, delta_x, rho)) + (1 - s) * np.log(1 - compute_p(eps, delta_x, rho)))

# bias and variance Rasch model with the probabilities as input 
compute_bias = lambda pi, rho: (1 / (rho * np.sum(pi * (1 - pi)) ** 2)) * np.sum(pi * (1 - pi) * (pi - 0.5))
var_bias = lambda pi, rho: 1 / (rho ** 2 * np.sum(pi * (1 - pi)))

# bias and variance Rasch model with delta x and epsilon as input
compute_bias_ = lambda eps, delta_x, rho: (1 / (rho * (np.sum(compute_p(eps, delta_x, rho) * (1 - compute_p(eps, delta_x, rho))))**2)) * np.sum(compute_p(eps, delta_x, rho) * (1 - compute_p(eps, delta_x, rho)) * (compute_p(eps, delta_x, rho) - 0.5))
var_bias_ = lambda eps, delta_x, rho: 1 / (rho ** 2 * np.sum(compute_p(eps, delta_x, rho) * (1 - compute_p(eps, delta_x, rho))))


def compute_save_mle(rho, epsilon_real, step, seed, save_loss = False, return_loss = False, return_errors = True):
    np.random.seed(seed)
    x0 = 0.0000001
    # in this experiment the interactions are equally spaced
    x = np.arange(x0, 1., step = step)
    s = (sigmoid(rho * (epsilon_real - x)) > np.random.random(size = len(x))) + 0.
    
    theoretical_bias = compute_bias_(epsilon_real, x, rho = rho)
    try:
        np.seterr(all = "raise")    
        compute_loss_eps = lambda eps: compute_loss(eps, rho, x, s)
        mle = minimize(compute_loss_eps, x0 = 0.5)["x"][0]
        empirical_error = mle - epsilon_real
    except:
        print("overflow")
        mle = "overflow"
        empirical_error = "overflow"
    return (epsilon_real, rho, len(x), seed, mle, empirical_error, theoretical_bias)

if __name__ == "__main__":
    experiment_errors = []
    for seed in range(10000):
        print("seed", seed)
        for rho in [1,5, 10, 20, 40]:
            print("rho", rho)
            for epsilon_real in [0.02, 0.05, 0.1, 0.2, 0.8, 0.9, 0.95, 0.98]:
                for step in [0.05, 0.005, 0.0005, 0.00005, 0.000005]:
                    exp_error = compute_save_mle(rho, epsilon_real, step, 100000 + seed, 
                                                save_loss = False, return_loss = False, return_errors = True)
                    experiment_errors.append(exp_error)
                    exp_df = pd.DataFrame(experiment_errors, 
                                          columns = ["epsilon_real", "rho", "len_x", "seed", "mle", "empirical_error", "theoretical_bias"])
                    exp_df.to_csv("/data/opdyn_identification/error_bias_rho_seed/exp_250411_minimize.csv")
                    
                    



