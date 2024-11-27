## Slice wasserstein estimation
# Student : Tristan Waddington

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp


def slice_wasserstein_gaussian(
    mu_1: np.ndarray[float],
    mu_2: np.ndarray[float],
    sigma: np.ndarray[float],
    n_samples: int = 100,
):
    """
    MC estimation of the slice Wasserstein distance between two Gaussian
    distributions with means mu_1 and mu_2 and standard deviation sigma

    Parameters:
    mu_1 : 1-D array_like, of length N
        Mean of the 1st N-dimensional distribution.
    mu_2 : 1-D array_like, of length N
        Mean of the 2nd N-dimensional distribution.
    sigma:    2-D array_like, of shape (N, N)
        Covariance matrix of the 2 distributions. It must be symmetric and
        positive-semidefinite for proper sampling.
    Returns:
    wasserstein_distance:   float
        MC estimation
    """
    # Draw samples from the unit sphere
    d = mu_1.shape[0]
    rng = np.random.default_rng()
    directions = rng.normal(loc=0, scale=1, size=(n_samples, d))
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    # Compute the Wasserstein distance for each direction
    wasserstein_distance = []
    for direction in directions:
        # Project the mean on the direction
        mu_1_proj = np.dot(mu_1, direction)  # u#\mu_1
        mu_2_proj = np.dot(mu_2, direction)  # u#\mu_2
        m_y = mu_1_proj - mu_2_proj

        # Compute the wasserstein distance
        left_term = np.abs(m_y) * (
            1 - 2 * sp.stats.norm.cdf(-np.abs(m_y) / np.linalg.norm(sigma))
        )
        right_term = (
            np.linalg.norm(sigma)
            * np.sqrt(2 / np.pi)
            * np.exp(-(m_y**2) / (2 * np.linalg.norm(sigma) ** 2))
        )
        wasserstein_distance.append(left_term + right_term)

    # The sliced Wasserstein distance between the true and the fake distributions
    # is computed as the average Wasserstein distance along all the projections.
    # [Deshpande et al. 2018]
    slice_wasserstein_distance = np.mean(wasserstein_distance)

    return slice_wasserstein_distance


def slice_wasserstein_generic(X, Y, n_samples=100, n_slices=100):
    """
    Compute the sliced Wasserstein distance between the empirical distribution
    of two datasets X and Y

    X and Y are numpy arrays of dimension n_X x d and n_Y x d where n is the
        number of samples and d is the dimension of the samples
    n_samples is the number of uniform samples used to estimate the
        Wasserstein distance
    n_slices is the number of slices (random direction) used to estimate the
        sliced Wasserstein distance
    """
    # dimension control
    d = X.shape[1]
    assert X.shape[0] >= n_samples, "Not enough samples for X"
    assert Y.shape[0] >= n_samples, "Not enough samples for Y"

    # Draw samples from the unit sphere
    rng = np.random.default_rng()
    directions = rng.normal(loc=0, scale=1, size=(n_slices, d))
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    wasserstein_distance = []
    for omega in directions:
        # Sample from the distributions
        X_samples = X[np.random.choice(d, n_samples)]
        Y_samples = Y[np.random.choice(d, n_samples)]

        # Project the samples onto the random direction omega
        X_omega = np.dot(X_samples, omega)
        Y_omega = np.dot(Y_samples, omega)

        # Sort the projections to get the quantile functions
        X_omega_sorted = np.sort(X_omega)
        Y_omega_sorted = np.sort(Y_omega)

        # Compute the Wasserstein distance for the projections
        # with the quantile function formula.
        wd1 = np.mean(np.abs(X_omega_sorted - Y_omega_sorted))
        wasserstein_distance.append(wd1)

    # The sliced Wasserstein distance between the true and the fake distributions
    # is computed as the average Wasserstein distance along all the projections.
    sliced_wasserstein_distance = np.mean(wasserstein_distance)

    return sliced_wasserstein_distance
