"""A simple script to visually evaluate bln fitting.

By: Robert Vogel

This script generates samples from a binomial logit 
normal (bln) distribution and then attempts to infer
the parameters of the sampling distribution from
these simulated data.  The resulting plot of simulation data
and fit are printed to pdf and the simulation parameters
are printed to json.  These files will printed to the current
directory.
"""

import sys
import json
import argparse

import numpy as np

import matplotlib.pyplot as plt

from blnm import dist
from blnm import fit


# Sampling parameters
COEFS = [0.2, 0.1, 0.3, 0.4]
MEANS = [-3, -1, 1, 3]
VARIANCE = 0.25
N_COUNTS = 100
N_SAMPLES = 2500
N_REPS = 10

# figure parameters
FIGSIZE = (4, 3.5)
FONTSIZE = 15
AX_POSITION = (0.2, 0.2, 0.75, 0.75)


def hist(data, normalize=True):
    """Histogram count data and returned centered x values."""
    dmin, dmax = data.min(), data.max()

    bins = np.linspace(dmin-0.5, dmax+0.5, num=(dmax-dmin + 2))

    y, x = np.histogram(data, bins = bins)

    if normalize:
        y = y / np.sum(y)

    x = x[1:] - 0.5

    return x.astype(np.int64), y


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_counts",
            type=int,
            default=N_COUNTS,
            help=f"Total number of read counts (default {N_COUNTS})")
    parser.add_argument("--n_samples",
            type=int,
            default=N_SAMPLES,
            help=f"Number of samples to draw (default {N_SAMPLES})")
    return parser.parse_args(args)


def main():
    args = _parse_args(sys.argv[1:])

    # each sample needs an alt count and total count.  I've
    # set the total, n_counts, for each sample to be a single value.
    alt_allele_counts = dist.sample_blnm(COEFS, MEANS, VARIANCE,
                                    args.n_counts, 
                                    args.n_samples)
    n_counts = np.array([args.n_counts for _ in range(args.n_samples)])

    # ===========================================
    # Infer distribution parameters from sampled data
    
    tmp_pars = fit.blnm(alt_allele_counts, n_counts, len(MEANS), disp=False)
    best_pars = tmp_pars
    
    # The bln mixture model uses random initial parameters.  This
    # loop fits data to a different random intialization and
    # stores the best fit
    
    print("Iter\tLog Like\tConverge Iters")
    for i in range(1, N_REPS):
    
        print(i, tmp_pars["log_likelihood"], tmp_pars["iterations"], sep="\t")

        tmp_pars = fit.blnm(alt_allele_counts, 
                n_counts, len(MEANS), disp=False)
    
        if tmp_pars["log_likelihood"] > best_pars["log_likelihood"]:
            best_pars = tmp_pars
    
    
    print("Best log-likelihood", best_pars["log_likelihood"])
    
    # ===========================================
    # Plot results
    
    x, empirical_prob = hist(alt_allele_counts)

    # true probability mass at each point sampled
    true_prob = dist.mixture_pmf(x,
                            args.n_counts,
                            COEFS,
                            MEANS,
                            VARIANCE)

    fig, ax = plt.subplots(1,1,figsize=FIGSIZE)
    ax.plot(x, empirical_prob, "o", 
            mfc="none", 
            color="k", 
            label="Histogram samples")
    ax.plot(x, true_prob, "-", color="k", label="True Density")
    
    for coef_i, mean_i in zip(best_pars["coefs"], best_pars["means"]):
    
        prob = coef_i * dist.pmf(x, args.n_counts,
                               mean_i,
                               best_pars["variance"])
        ax.plot(x, prob, "-", label="Inferred mixture")
    
    ax.legend(loc=0)
    ax.set_position(AX_POSITION)
    
    ax.set_xlabel("Alternative Counts", fontsize=FONTSIZE)
    ax.set_ylabel("Probability", fontsize=FONTSIZE)
    
    fig.savefig("pmf.png")
    
    with open("sim_pars.json", "w") as fid:
        json.dump({
                "coef": COEFS,
                "means":MEANS,
                "variance":VARIANCE,
                "n_counts":args.n_counts,
                "n_samples":args.n_samples
                },
            fid,
            indent=4)


if __name__ == "__main__":
    main()
