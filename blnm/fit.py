"""Fitting routines.
"""
import numpy as np

from scipy.stats import binom
from scipy import special as scisp

from . import utils
from . import dist

INTEGRAL_N_SAMPLES = 1000


def _data_log_likelihood(zeroth_order):
    """Compute the data log likelihood.

    Args:
        zeroth_order : ((k mixture, N sample) np.ndarray) the values are
            those updated by the _e_step function.
    Returns:
        float
    """
    return np.sum(np.log(np.sum(zeroth_order, axis=0)))

def _e_step(x_counts, n_counts, 
            coefs, means, variance, k_mixtures,
            zeroth_order,
            first_order,
            second_order,
            integral_n_samples,
            seed):
    """E step of EM algorithm.

    Updates the conditional expectation for the zeroth, 
    first and second order equation in place.

    Args:
        x_counts : ((N, ) np.ndarray)
        n_counts : ((N, ) np.ndarray)
        coefs : ((k_mixtures, ) np.ndarray)
        means : ((k_mixtures, ) np.ndarray)
        variance : (float)
        k_mixtures : (int)
        zeroth_order : ((k_mixtures, N) np.ndarray)
        first_order : ((k_mixtures, N) np.ndarray)
        second_order : ((k_mixtures, N) np.ndarray)
        integral_n_samples : (int) number of samples to draw for
            computing expectations
        seed : any seed compatible with numpy random number
            generator object (e.g. numpy.random.default_rng(seed=seed))

    Returns
        None
    """

    rng = np.random.default_rng(seed=seed)

    for k, coef_k in enumerate(coefs):

        for i, (x_i, n_i) in enumerate(zip(x_counts, n_counts)):

            # generate s for computing binomial probabilities
            s = rng.normal(means[k], 
                           np.sqrt(variance),
                           size = integral_n_samples)

            # compute the binomial probabilites
            p = utils.logistic(s)
            # f_X = scisp.binom(n_i, x_i) * p**x_i * (1-p)**(n_i-x_i)
            f_X = binom.pmf(x_i, n_i, p)

            zeroth_order[k, i] = np.mean(f_X) * coef_k
            first_order[k, i] = np.mean(s * f_X) * coef_k
            second_order[k, i] = np.mean(s**2 * f_X) * coef_k


def _m_step(zeroth_order,
           first_order,
           second_order):
    """M step of EM algorithm

    Args:
        zeroth_order : ((k_mixtures, N) np.ndarray)
        first_order : ((k_mixtures, N) np.ndarray)
        second_order : ((k_mixtures, N) np.ndarray)

    Returns:
        coefs : ((k mixtures, ) np.ndarray) 
        means : ((k mixture, ) np.ndarray)
        variance : (float)
    """

    # compute helper values
    beta = np.sum(zeroth_order, axis=0)

    k_mixtures, N = zeroth_order.shape

    coefs = np.zeros(k_mixtures)
    means = np.zeros(k_mixtures)
    variance = 0

    for k in range(k_mixtures):
        # helper value
        N_k = np.sum(zeroth_order[k, :] / beta)

        coefs[k] = N_k / N
        means[k] = np.sum(first_order[k, :] / beta) / N_k
        variance += np.mean(second_order[k, :] / beta) - coefs[k]*means[k]**2

    return coefs, means, variance

# return (coefs, means, variance)

def blnm(x_counts, 
         n_counts,
         k_mixtures, 
         seed=None,
         tolerance=1e-6, 
         max_iter=1000,
         disp = True,
         integral_n_samples = INTEGRAL_N_SAMPLES):
    """Fit mixture of BLN models.

    Args:
        x_counts : ((N,) np.ndaray) alternative allele specific expression 
            counts
        n_counts : ((N,) np.ndarray) alternative + reference allele 
            specific expression counts
        k_mixtures : (int) > 0 number of mixtures to fit
        seed : (any input to numpy random Generator object)
        tolerance : (float) criterion for convergence
        max_iter : (int) maximum number of interations
        disp : (bool) print iteration information to standard out
        integral_n_samples : (int) number of samples for computing
            the BLN integral

    Returns:
        dict : 
            coefs: ((k_mixtures,) np.ndarray) mixture fractions
            means: ((k_mixtures,) np.ndarray) BLN mixture means
            variance: (float) variance of all BLN mixture
            log_likelihood: (float)
            converged: (int) 0 if converged 1 otherwise
            converge_message: (string)
            iterations: (int) number of iterations for convergence
    """
    # Validate inputs

    if k_mixtures < 1 or not isinstance(k_mixtures, int):
        raise ValueError("k_mixtures must be an integer greater than 1")
    elif x_counts.shape != n_counts.shape:
        raise ValueError("alt_allele_counts and ref_allele_counts must have"
                        "shape")
    elif x_counts.ndim != 1:
        raise ValueError

    # validate data
    for x_i, n_i in zip(x_counts, n_counts):
        if x_i < 0 or n_i < 0:
            raise ValueError("Counts must be 0 or positive integers.")
        elif x_i > n_i:
            raise ValueError("Alternative allele counts must be less than total counts.")
        #TODO come back to this point
#         elif not isinstance(x_i, int):
#             raise ValueError("Counts must be integers.")
#         elif not isinstance(n_i, int):
#             raise ValueError("Counts must be integers.")


    rng = np.random.default_rng(seed=seed)

    # initialize parameters
    coefs = rng.uniform(low=0.1, high=0.9, size=k_mixtures)
    coefs = coefs / np.sum(coefs)

    p = rng.uniform(low=0.01, high=0.99, size=k_mixtures)
    means = np.log(p / (1-p))

    variance = rng.uniform(low=0.1, high=3)

    # preallocate memory for arrays constructed in the E step
    # each array represents
    # E_s[ s^order f_X(x;s,n) | j^th iteration parameters ]
    zeroth_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    first_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    second_order = np.zeros(shape=(k_mixtures, len(x_counts)))


    _e_step(x_counts, n_counts,
            coefs, means, variance, k_mixtures,
            zeroth_order,
            first_order,
            second_order,
            integral_n_samples,
            rng)

    # store log likelihoods
    log_likelihood = [None, _data_log_likelihood(zeroth_order)]

    # Perform E and M steps until the difference in log-likelihood
    # from iter_n to iter_n +1 iteration is less than tolerance,
    # or maximum number of iterations are reached

    delta = 100
    iter_n = 0

    if disp:
        print("Iter\tlog likelihood")
        print(f"{iter_n:04d}", 
              f"{log_likelihood[1]:0.4f}",
              sep="\t", end="\n")

    while delta >= tolerance and max_iter > iter_n:

        log_likelihood[0] = log_likelihood[1]


        # inplace assignment of order arrays
        _e_step(x_counts, n_counts,
                coefs, means, variance, k_mixtures,
                zeroth_order,
                first_order,
                second_order,
                integral_n_samples,
                rng)

        # sanity check
        # beta = np.sum(zeroth_order, axis=0)
        # tmp = 0
        # for k in range(k_mixtures):
        #     tmp += zeroth_order[k, :] / beta

        # assert np.sum(tmp) == x_counts.size

        coefs, means, variance = _m_step(zeroth_order,
                                        first_order,
                                        second_order)


        log_likelihood[1] = _data_log_likelihood(zeroth_order)

        delta = log_likelihood[1] - log_likelihood[0]

        iter_n += 1

        if disp:
            print(f"{iter_n:04d}", 
                  f"{log_likelihood[1]:0.4f}",
                  sep="\t", end="\n")

    out = {
            "coefs": coefs,
            "means": means,
            "variance": variance,
            "log_likelihood": log_likelihood[1],
            "converged": 0,
            "converge_message": "success",
            "iterations": iter_n
            }

    if iter_n == max_iter:
        out["converged"] = 1
        out["converge_message"] = "max iteration reached without convergence"

    return out






def blnm_user(x_counts,
         n_counts,
         k_mixtures,
         guessed_coefs,
         guessed_p,
         guessed_variance,
         seed=None,
         tolerance=1e-6,
         max_iter=1000,
         disp = True,
         integral_n_samples = INTEGRAL_N_SAMPLES):
    """Fit mixture of BLN models.

    Args:
        x_counts : ((N,) np.ndaray) alternative allele specific expression
            counts
        n_counts : ((N,) np.ndarray) alternative + reference allele
            specific expression counts
        k_mixtures : (int) > 0 number of mixtures to fit
        guessed_coefs : ((k_mixtures,) np.ndarray) initial mixture fractions
        guessed_p : ((k_mixtures,) np.ndarray) initial BLN probability parameters
        guessed_variance : (float) initial variance of all BLN mixture
        seed : (any input to numpy random Generator object)
        tolerance : (float) criterion for convergence
        max_iter : (int) maximum number of interations
        disp : (bool) print iteration information to standard out
        integral_n_samples : (int) number of samples for computing
            the BLN integral

    Returns:
        dict :
            coefs: ((k_mixtures,) np.ndarray) mixture fractions
            means: ((k_mixtures,) np.ndarray) BLN mixture means
            variance: (float) variance of all BLN mixture
            log_likelihood: (float)
            converged: (int) 0 if converged 1 otherwise
            converge_message: (string)
            iterations: (int) number of iterations for convergence
    """

    if k_mixtures < 1 or not isinstance(k_mixtures, int):
        raise ValueError("k_mixtures must be an integer greater than 1")
    elif x_counts.shape != n_counts.shape:
        raise ValueError("alt_allele_counts and ref_allele_counts must have"
                        "shape")
    elif x_counts.ndim != 1:
        raise ValueError

    if len(guessed_coefs) != k_mixtures or len(guessed_p)!= k_mixtures:
        raise ValueError("guessed_coefs must have length")

    for x_i, n_i in zip(x_counts, n_counts):
        if x_i < 0 or n_i < 0:
            raise ValueError("Counts must be 0 or positive integers.")
        elif x_i > n_i:
            raise ValueError("Alternative allele counts must be less than total counts.")



    rng = np.random.default_rng(seed=seed)
    #guessing parameters
    coefs = np.array(guessed_coefs)/np.sum(guessed_coefs)
    means = np.log(np.array(guessed_p) / (1-np.array(guessed_p)))
    variance = guessed_variance

    # preallocate memory for arrays constructed in the E step
    # each array represents
    # E_s[ s^order f_X(x;s,n) | j^th iteration parameters ]
    zeroth_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    first_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    second_order = np.zeros(shape=(k_mixtures, len(x_counts)))


    _e_step(x_counts, n_counts,
            coefs, means, variance, k_mixtures,
            zeroth_order,
            first_order,
            second_order,
            integral_n_samples,
            rng)


    log_likelihood = [None, _data_log_likelihood(zeroth_order)]
    delta = 100
    iter_n = 0

    if disp:
        print("Iter\tlog likelihood")
        print(f"{iter_n:04d}",
              f"{log_likelihood[1]:0.4f}",
              sep="\t", end="\n")

    while delta >= tolerance and max_iter > iter_n:

        log_likelihood[0] = log_likelihood[1]



        _e_step(x_counts, n_counts,
                coefs, means, variance, k_mixtures,
                zeroth_order,
                first_order,
                second_order,
                integral_n_samples,
                rng)

        # sanity check
        # beta = np.sum(zeroth_order, axis=0)
        # tmp = 0
        # for k in range(k_mixtures):
        #     tmp += zeroth_order[k, :] / beta

        # assert np.sum(tmp) == x_counts.size

        coefs, means, variance = _m_step(zeroth_order,first_order,second_order)


        log_likelihood[1] = _data_log_likelihood(zeroth_order)

        delta = log_likelihood[1] - log_likelihood[0]

        iter_n += 1

        if disp:
            print(f"{iter_n:04d}",
                  f"{log_likelihood[1]:0.4f}",
                  sep="\t", end="\n")

    out = {
            "coefs": coefs,
            "means": means,
            "variance": variance,
            "log_likelihood": log_likelihood[1],
            "converged": 0,
            "converge_message": "success",
            "iterations": iter_n
            }

    if iter_n == max_iter:
        out["converged"] = 1
        out["converge_message"] = "max iteration reached without convergence"

    return out


def blnm_start(x_counts,
               n_counts,
               k_mixtures,
               guess_time,
               seed=None,
               tolerance=1e-6,
               max_iter=1000,
               disp=True,
               integral_n_samples=INTEGRAL_N_SAMPLES):
    """Fit mixture of BLN models with multiple initial guesses and select the best one.

    Args:
        x_counts : ((N,) np.ndarray) alternative allele specific expression counts
        n_counts : ((N,) np.ndarray) alternative + reference allele specific expression counts
        k_mixtures : (int) > 0 number of mixtures to fit
        guess_time : (int) number of random initializations to try
        seed : (any input to numpy random Generator object)
        tolerance : (float) criterion for convergence
        max_iter : (int) maximum number of iterations
        disp : (bool) print iteration information to standard out
        integral_n_samples : (int) number of samples for computing the BLN integral

    Returns:
        dict :
            coefs: ((k_mixtures,) np.ndarray) mixture fractions
            means: ((k_mixtures,) np.ndarray) BLN mixture means
            variance: (float) variance of all BLN mixture
            log_likelihood: (float)
            converged: (int) 0 if converged 1 otherwise
            converge_message: (string)
            iterations: (int) number of iterations for convergence
    """

    if k_mixtures < 1 or not isinstance(k_mixtures, int):
        raise ValueError("k_mixtures must be an integer greater than 1")
    elif guess_time < 1 or not isinstance(guess_time, int):
        raise ValueError("guess_time must be an integer greater than 1")
    elif x_counts.shape != n_counts.shape:
        raise ValueError("alt_allele_counts and ref_allele_counts must have the same shape")
    elif x_counts.ndim != 1:
        raise ValueError

    rng = np.random.default_rng(seed=seed)

    best_log_likelihood = -np.inf
    best_params = None


    for _ in range(guess_time):
        # Generate random initial guesses
        coefs = rng.uniform(low=0.1, high=0.9, size=k_mixtures)
        coefs = coefs / np.sum(coefs)

        p = rng.uniform(low=0.01, high=0.99, size=k_mixtures)
        means = np.log(p / (1 - p))

        variance = rng.uniform(low=0.1, high=3)

       
        zeroth_order = np.zeros(shape=(k_mixtures, len(x_counts)))
        first_order = np.zeros(shape=(k_mixtures, len(x_counts)))
        second_order = np.zeros(shape=(k_mixtures, len(x_counts)))

        
        _e_step(x_counts, n_counts,
                coefs, means, variance, k_mixtures,
                zeroth_order,
                first_order,
                second_order,
                integral_n_samples,
                rng)

        
        log_likelihood = _data_log_likelihood(zeroth_order)

        # Keep track of the best initial guess
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_params = (coefs, means, variance)

    # Use the best initial parameters for EM iterations
    coefs, means, variance = best_params


    zeroth_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    first_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    second_order = np.zeros(shape=(k_mixtures, len(x_counts)))

    _e_step(x_counts, n_counts,
            coefs, means, variance, k_mixtures,
            zeroth_order,
            first_order,
            second_order,
            integral_n_samples,
            rng)

    log_likelihood = [None, _data_log_likelihood(zeroth_order)]
    delta = 100
    iter_n = 0

    if disp:
        print("Iter\tlog likelihood")
        print(f"{iter_n:04d}", f"{log_likelihood[1]:0.4f}", sep="\t", end="\n")

    while delta >= tolerance and max_iter > iter_n:
        log_likelihood[0] = log_likelihood[1]


        _e_step(x_counts, n_counts,
                coefs, means, variance, k_mixtures,
                zeroth_order,
                first_order,
                second_order,
                integral_n_samples,
                rng)


        coefs, means, variance = _m_step(zeroth_order, first_order, second_order)

        log_likelihood[1] = _data_log_likelihood(zeroth_order)
        delta = log_likelihood[1] - log_likelihood[0]

        iter_n += 1

        if disp:
            print(f"{iter_n:04d}", f"{log_likelihood[1]:0.4f}", sep="\t", end="\n")

    out = {
        "coefs": coefs,
        "means": means,
        "variance": variance,
        "log_likelihood": log_likelihood[1],
        "converged": 0,
        "converge_message": "success",
        "iterations": iter_n
    }

    if iter_n == max_iter:
        out["converged"] = 1
        out["converge_message"] = "max iteration reached without convergence"

    return out
