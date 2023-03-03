import numbers
import numpy as np

def mbic(log_likelihood, n_samples, k_mixtures):
    """Compute the BIC for this BLN mixture model.
    
    Compute the Bayesian Information Criterion (BIC) for our BLN mixture
    model.  We compute the number of parameters, p, of our k components 
    BLN mixture model by summing k coefficients, k means, and
    1 variance parameters.  However, when including the constraint
    that the coefficients sum to 1, the effective number of 
    parameters are

    p = k + k + 1 - 1 = 2k.

    Args:
        log_likelihood : (float)
        n_samples : (int)
        k_mixtures : (int)

    Returns:
        (float) Bayesian information criterion
    """
    if k_mixtures <= 0 or n_samples <= 0:
        raise ValueError
    elif (not isinstance(log_likelihood, numbers.Number) or
            not isinstance(n_samples, numbers.Number) or
            not isinstance(k_mixtures, numbers.Number)):
        raise ValueError

    return 2*k_mixtures * np.log(n_samples) - 2*log_likelihood


def logistic(s):
    """Logistic function

    Args:
        s : (number or np.ndarray)

    Returns:
        logistic transform of value(s) given
    """
    return 1 / (1 + np.exp(-s))

# def logit(p):
#     if p > 1 or p < 0:
#         raise ValueError
# 
#     if p == 1:
#         raise ValueError
# 
#     return np.log(p / (1-p))
