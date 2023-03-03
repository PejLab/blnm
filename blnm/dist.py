import numbers
import numpy as np

import scipy.special as scisp
from scipy.stats import binom

from . import utils

DEFAULT_PMF_SAMPLING_SIZE = 1000


def get_mixture_component_from_prob(coefs, val):

    sum_coefs = 0
    for coef in coefs:
        sum_coefs += coef
        if coef < 0 or coef > 1:
            raise ValueError("Coefs need to be on interval [0,1]")

    if sum_coefs != 1:
        raise ValueError("Coefficients do not sum to 1.")
    elif val > 1 or val < 0:
        raise ValueError("Input val outside bounds [0,1]")

    cumulative_prob = 0

    for k, coef in enumerate(coefs):

        cumulative_prob += coef

        if val < cumulative_prob:
            return k

    raise ValueError("Something went wrong")


#TODO verify output
def sample_bln(mean, variance, n_counts, n_samples=1, rng=None):
    """Sample BLN

    Args:
        mean: (float)
        variance: (float)
        n_counts: (int)
        n_samples: (int) default 1
        rng : (numpy Generator object) default None
    """
    rng = np.random.default_rng(seed=rng)

    s = rng.normal(mean, np.sqrt(variance), size=n_samples)

    return rng.binomial(n_counts, utils.logistic(s)).astype(np.int64)


def sample_blnm(coefs, means, variance, n_counts, n_samples, rng=None):
    """Sample BLN mixture model.

    Args:
        coefs : ((K, ) np.ndarray) mixture probabilities
        means : ((K, ) np.ndarray) mean of each mixture
        variance : (float)
        n_counts : (int) number of reference + alternative allele counts
        n_samples : (int) number of samples to draw from mixture dist

    Returns:
        alt_counts : ((n_samples, ) np.ndarray) counts of alternative allele
    """

    if len(coefs) != len(means):
        raise ValueError("Coefs and means need identical length")
    elif variance <= 0 or n_counts <= 0 or n_samples <= 0:
        raise ValueError("variance, n_counts, n_samples "
                         "must be greater than zero.")
    
    rng = np.random.default_rng(seed=rng)

    k_mixtures = len(coefs)

    alt_counts = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):

        # sample mixture component
        k = get_mixture_component_from_prob(coefs, rng.uniform())

        alt_counts[i] = sample_bln(means[k], 
                            variance, n_counts, 1, 
                            rng=rng)

    return alt_counts


def _optimized_pmf(x_counts, 
                   n_counts, 
                   mean, variance, 
                   probability, 
                   size=DEFAULT_PMF_SAMPLING_SIZE,
                   seed=None):
    """In place computation of pmf.

    Inputs are not validated, and probability array must
    be supplied.

    Args:
        
    """

    rng = np.random.default_rng(seed=seed)

    p = utils.logistic(rng.normal(loc=mean, 
                                scale=np.sqrt(variance), 
                                size=size))

    i = 0
    for x, n in zip(x_counts, n_counts):

        # probability[i] = np.mean(scisp.binom(n, x) *
        #                             p**x * (1-p)**(n - x))
        probability[i] = np.mean(binom.pmf(x, n, p))

        i += 1

def pmf(x_counts, n_counts, mean, variance, 
        size=DEFAULT_PMF_SAMPLING_SIZE, seed=None):
    """Probability mass by sampling

    Args:
        x_counts : (int or (N,) np.ndarray) counts of one of the two outcomes
        n_counts : (int or (N,) np.ndarray) total counts
        mean : (float)
        variance : (float)
        size : (int) number of samples to draw for computing expectation
        seed : (int or numpy random Generator object)

    Returns
        probabilities : (float or (N,) np.ndarray)
    """

    # make scalar inputs iterables for simplicity
    if (isinstance(x_counts, numbers.Number) and 
            isinstance(n_counts, numbers.Number)):

        x_counts = [x_counts]
        n_counts = [n_counts]

    elif isinstance(x_counts, numbers.Number):

        x_counts = [x_counts for _ in n_counts]
        
    elif isinstance(n_counts, numbers.Number):

        n_counts = [n_counts for _ in x_counts]

    elif len(n_counts) != len(x_counts):

        raise ValueError("Incompatible x_counts and n_counts.")

    # Validate input values
    for x, n in zip(x_counts, n_counts):
        if x > n or x < 0:
            raise ValueError
        elif n <= 0:
            raise ValueError

    probability = np.zeros(len(x_counts))

    _optimized_pmf(x_counts, n_counts,
                   mean, variance,
                   probability,
                   size,
                   seed=seed)

    return probability

def mixture_pmf(x_counts, n_counts, coefs, means, variance,
                size=DEFAULT_PMF_SAMPLING_SIZE, seed=None):

    probability = 0

    for coef, mean in zip(coefs, means):
        probability += coef * pmf(x_counts, n_counts, 
                                  mean, variance, 
                                  size=size, seed=seed)
    
    return probability

