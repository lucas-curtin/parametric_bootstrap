"""Construct our core pdfs."""

from __future__ import annotations

from math import erf, pi, sqrt

import numpy as np
from numba import njit


@njit
def normal_cdf(x: float) -> float:
    """CDF of a standard normal distribution."""
    return 0.5 * (1 + erf(x / sqrt(2)))


@njit
def g_s_base(  # noqa: PLR0913
    x: np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Normalized Truncated Crystal Ball PDF."""
    pdf = np.zeros_like(x)

    z = (x - mu) / sigma
    out_of_bounds = (x < x_min) | (x > x_max)
    pdf[out_of_bounds] = 0.0

    beta_plus = (z > -beta) & ~out_of_bounds
    pdf[beta_plus] = np.exp(-(z[beta_plus] ** 2) / 2)

    beta_minus = (z <= -beta) & ~out_of_bounds
    a = ((m / beta) ** m) * np.exp(-(beta**2) / 2)
    b = (m / beta) - beta
    pdf[beta_minus] = a * (b - z[beta_minus]) ** (-m)

    z_min = (x_min - mu) / sigma
    z_max = (x_max - mu) / sigma

    gauss_contrib = 0
    if z_max > beta:
        gauss_min = max(-beta, z_min)
        gauss_max = z_max
        gauss_contrib = sqrt(2 * pi) * (normal_cdf(gauss_max) - normal_cdf(gauss_min))
    tail_contrib = 0.0

    if z_min <= -beta:
        tail_min = z_min
        tail_max = min(z_max, -beta)

        u_min = m / beta - beta - tail_min
        u_max = m / beta - beta - tail_max

        c = ((m / beta) ** m) * np.exp(-(beta**2) / 2)

        tail_contrib = (c / (m - 1)) * (u_max ** (1 - m) - u_min ** (1 - m))

    normalisation = sigma * (gauss_contrib + tail_contrib)
    return pdf / normalisation


@njit
def h_s_base(y: np.ndarray, lam: float, y_min: float, y_max: float) -> np.ndarray:
    """Normalised Truncated Exponential PDF."""
    pdf = np.zeros_like(y)
    within_bounds = (y >= y_min) & (y <= y_max)
    y = y[within_bounds]
    normalisation = np.exp(-lam * y_min) - np.exp(-lam * y_max)
    trunc_pdf = lam * np.exp(-lam * y) / normalisation
    pdf[within_bounds] = trunc_pdf
    return pdf


@njit
def g_b_base(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Normalised Truncated Uniform PDF."""
    pdf = np.zeros_like(x)
    within_bounds = (x >= x_min) & (x <= x_max)
    normalisation = x_max - x_min
    pdf[within_bounds] = 1.0 / normalisation
    return pdf


@njit
def h_b_base(y: np.ndarray, mu: float, sigma: float, y_min: float, y_max: float) -> np.ndarray:
    """Normalised Truncated Normal PDF."""
    pdf = np.zeros_like(y)
    within_bounds = (y >= y_min) & (y <= y_max)
    y = y[within_bounds]
    z_min = (y_min - mu) / sigma
    z_max = (y_max - mu) / sigma
    normalisation = normal_cdf(z_max) - normal_cdf(z_min)
    z = (y - mu) / sigma
    trunc_pdf = np.exp(-(z**2) / 2) / (sigma * np.sqrt(2 * np.pi) * normalisation)
    pdf[within_bounds] = trunc_pdf
    return pdf
