"""Holds our PDFs for modelling statistical distributions and their computations."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, uniform

from .integrator import integrate_single_pdf

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
    x_lower: float,
    x_upper: float,
) -> np.ndarray:
    """Computes the truncated Crystal Ball PDF normalised over the specified range."""
    pdf = crystal_ball(x=x, mu=mu, sigma=sigma, beta=beta, m=m)
    norm_factor = integrate_single_pdf(
        pdf=lambda t: crystal_ball(t, mu, sigma, beta, m),
        lower=x_lower,
        upper=x_upper,
    )
    return pdf / norm_factor


def exponential_decay(y: float | np.ndarray, lam: float) -> np.ndarray:
    """Computes the exponential decay PDF.

    Parameters:
        y (float | np.ndarray): Input values.
        lam (float): Rate parameter of the exponential distribution.

    Returns:
        np.ndarray: The PDF values.
    """
    return lam * np.exp(-lam * y)


def exponential_decay_inverse_normalisation(
    lam: float,
    lower: float,
    upper: float,
) -> float:
    """Computes the inverse normalisation constant (N^-1) for the truncated exponential decay.

    Parameters:
        lam (float): Rate parameter of the exponential distribution.
        lower (float): Lower bound of the truncation range.
        upper (float): Upper bound of the truncation range.

    Returns:
        float: The inverse normalisation constant N^-1.
    """
    # Compute the integral of the exponential decay over the range analytically
    integral = np.exp(-lam * lower) - np.exp(-lam * upper)
    # Return the inverse of the integral as the normalisation constant
    return 1 / integral


def truncated_exponential_decay(
    y: float | np.ndarray,
    lam: float,
    y_lower: float,
    y_upper: float,
) -> np.ndarray:
    """Computes the truncated exponential decay PDF normalized over the specified range.

    Parameters:
        y (float | np.ndarray): Input values.
        lam (float): Rate parameter of the exponential distribution.
        lower (float): Lower bound of the truncation range.
        upper (float): Upper bound of the truncation range.

    Returns:
        np.ndarray: The normalized PDF values.
    """
    # Compute the inverse normalization constant
    inverse_norm = exponential_decay_inverse_normalisation(lam=lam, lower=y_lower, upper=y_upper)
    # Compute the exponential decay values
    pdf = exponential_decay(y, lam)
    # Normalize the PDF
    return pdf * inverse_norm


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
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
) -> np.ndarray:
    """Computes the signal PDF as a product of a truncated Crystal Ball and exponential decay."""
    gs_x = truncated_crystal_ball(
        x,
        params["mu"],
        params["sigma"],
        params["beta"],
        params["m"],
        x_lower=x_lower,
        x_upper=x_upper,
    )
    hs_y = truncated_exponential_decay(y, params["lambda"], y_lower=y_lower, y_upper=y_upper)
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
    y_lower: float,
    y_upper: float,
    x_lower: float,
    x_upper: float,
) -> np.ndarray:
    """Computes the total PDF as a weighted sum of signal and background PDFs."""
    f = params["f"]
    signal = signal_pdf(
        x=x,
        y=y,
        params=params,
        y_lower=y_lower,
        y_upper=y_upper,
        x_lower=x_lower,
        x_upper=x_upper,
    )
    background = background_pdf(x, y, params)
    return f * signal + (1 - f) * background
