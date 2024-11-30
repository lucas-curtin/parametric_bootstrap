"""Holds our PDFs for modelling statistical distributions and their computations."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, uniform

from .integrator import integrate_pdf

# Define hardcoded bounds
X_LOWER = 0
X_UPPER = 5
Y_LOWER = 0
Y_UPPER = 10


def crystal_ball_inverse_normalisation(sigma: float, beta: float, m: float) -> float:
    """Computes the inverse normalisation constant (N^-1) for the Crystal Ball PDF.

    Parameters:
        sigma (float): Standard deviation (scale) of the Gaussian core.
        beta (float): Transition point for the power-law tail.
        m (float): Slope of the power-law tail.

    Returns:
        float: The inverse normalisation constant N^-1.
    """
    # Calculate the components of the formula
    term1 = (m / (beta * (m - 1))) * np.exp(-(beta**2) / 2)
    term2 = np.sqrt(2 * np.pi) * norm.cdf(beta)

    # Combine terms to compute N^-1
    return sigma * (term1 + term2)


def crystal_ball(
    x: float | np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
) -> np.ndarray:
    """Computes the Crystal Ball probability density function (PDF)."""
    z = (x - mu) / sigma
    a = ((m / beta) ** m) * np.exp(-(beta**2) / 2)
    b = (m / beta) - beta - z

    if z > -beta:
        return np.exp(-(z**2) / 2) * (z > -beta)

    return a * (b ** (-m)) * (z <= -beta)


def truncated_crystal_ball(
    x: float | np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
) -> np.ndarray:
    """Computes the truncated Crystal Ball PDF normalised over the specified range."""
    pdf = crystal_ball(x=x, mu=mu, sigma=sigma, beta=beta, m=m)
    inverse_norm = crystal_ball_inverse_normalisation(sigma=sigma, beta=beta, m=m)
    return pdf * inverse_norm


def exponential_decay(y: float | np.ndarray, lam: float) -> np.ndarray:
    """Computes the exponential decay PDF."""
    return lam * np.exp(-lam * y)


def truncated_exponential_decay(
    y: float | np.ndarray,
    lam: float,
) -> np.ndarray:
    """Computes the truncated exponential decay PDF normalised over the specified range."""
    norm_factor = integrate_pdf(lambda t: exponential_decay(t, lam), Y_LOWER, Y_UPPER)
    pdf = exponential_decay(y, lam)
    return pdf / norm_factor


def uniform_pdf(x: float | np.ndarray) -> np.ndarray:
    """Computes the uniform distribution PDF over a specified range."""
    return uniform.pdf(x, loc=X_LOWER, scale=X_UPPER - X_LOWER)


def truncated_gaussian_pdf(
    y: float | np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Computes the truncated Gaussian PDF normalised over the specified range."""
    norm_factor = norm.cdf(Y_UPPER, mu, sigma) - norm.cdf(Y_LOWER, mu, sigma)
    pdf = norm.pdf(y, mu, sigma)
    return pdf / norm_factor


def signal_pdf(
    x: float | np.ndarray,
    y: float | np.ndarray,
    params: dict,
) -> np.ndarray:
    """Computes the signal PDF as a product of a truncated Crystal Ball and exponential decay."""
    gs_x = truncated_crystal_ball(
        x,
        params["mu"],
        params["sigma"],
        params["beta"],
        params["m"],
    )
    hs_y = truncated_exponential_decay(y, params["lambda"])
    return gs_x * hs_y


def background_pdf(
    x: float | np.ndarray,
    y: float | np.ndarray,
    params: dict,
) -> np.ndarray:
    """Computes the background PDF as a product of a uniform distribution and a truncated Gaussian."""
    gb_x = uniform_pdf(x)
    hb_y = truncated_gaussian_pdf(y, params["mu_b"], params["sigma_b"])
    return gb_x * hb_y


def total_pdf(
    x: float | np.ndarray,
    y: float | np.ndarray,
    params: dict,
) -> np.ndarray:
    """Computes the total PDF as a weighted sum of signal and background PDFs."""
    f = params["f"]
    signal = signal_pdf(x, y, params)
    background = background_pdf(x, y, params)
    return f * signal + (1 - f) * background
