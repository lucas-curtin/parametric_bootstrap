"""Holds our PDFs for modelling statistical distributions and their computations."""

from __future__ import annotations

import numpy as np
from integrator import integrate_pdf
from scipy.stats import norm, uniform


def crystal_ball(
    x: float | np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
) -> np.ndarray:
    """Computes the Crystal Ball probability density function (PDF).

    Parameters:
        x (float or np.ndarray): The input variable.
        mu (float): Mean (location) of the Gaussian core.
        sigma (float): Standard deviation (scale) of the Gaussian core.
        beta (float): Transition point for the power-law tail.
        m (float): Slope of the power-law tail.

    Returns:
        np.ndarray: Values of the PDF at the given points.
    """
    z = (x - mu) / sigma
    a = (m / beta) ** m * np.exp(-(beta**2) / 2)
    b = (m / beta) - beta - z

    gaussian = np.exp(-(z**2) / 2) * (z > -beta)
    power_law = a * b**-m * (z <= -beta)

    return gaussian + power_law


def truncated_crystal_ball(
    x: float | np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Computes the truncated Crystal Ball PDF normalised over the specified range.

    Parameters:
        x (float or np.ndarray): The input variable.
        mu (float): Mean (location) of the Gaussian core.
        sigma (float): Standard deviation (scale) of the Gaussian core.
        beta (float): Transition point for the power-law tail.
        m (float): Slope of the power-law tail.
        lower (float): Lower bound of the truncation.
        upper (float): Upper bound of the truncation.

    Returns:
        np.ndarray: Normalised values of the PDF at the given points.
    """
    norm_factor = integrate_pdf(lambda t: crystal_ball(t, mu, sigma, beta, m), lower, upper)
    pdf = crystal_ball(x, mu, sigma, beta, m)
    return pdf / norm_factor


def exponential_decay(y: float | np.ndarray, lam: float) -> np.ndarray:
    """Computes the exponential decay PDF.

    Parameters:
        y (float or np.ndarray): The input variable.
        lam (float): Decay constant.

    Returns:
        np.ndarray: Values of the PDF at the given points.
    """
    return lam * np.exp(-lam * y)


def truncated_exponential_decay(
    y: float | np.ndarray,
    lam: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Computes the truncated exponential decay PDF normalised over the specified range.

    Parameters:
        y (float or np.ndarray): The input variable.
        lam (float): Decay constant.
        lower (float): Lower bound of the truncation.
        upper (float): Upper bound of the truncation.

    Returns:
        np.ndarray: Normalised values of the PDF at the given points.
    """
    norm_factor = integrate_pdf(lambda t: exponential_decay(t, lam), lower, upper)
    pdf = exponential_decay(y, lam)
    return pdf / norm_factor


def uniform_pdf(x: float | np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Computes the uniform distribution PDF over a specified range.

    Parameters:
        x (float or np.ndarray): The input variable.
        lower (float): Lower bound of the range.
        upper (float): Upper bound of the range.

    Returns:
        np.ndarray: Values of the PDF at the given points.
    """
    return uniform.pdf(x, loc=lower, scale=upper - lower)


def truncated_gaussian_pdf(
    y: float | np.ndarray,
    mu: float,
    sigma: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Computes the truncated Gaussian PDF normalised over the specified range.

    Parameters:
        y (float or np.ndarray): The input variable.
        mu (float): Mean (location) of the Gaussian.
        sigma (float): Standard deviation (scale) of the Gaussian.
        lower (float): Lower bound of the truncation.
        upper (float): Upper bound of the truncation.

    Returns:
        np.ndarray: Normalised values of the PDF at the given points.
    """
    norm_factor = norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)
    pdf = norm.pdf(y, mu, sigma)
    return pdf / norm_factor


def signal_pdf(
    x: float | np.ndarray,
    y: float | np.ndarray,
    params: dict,
) -> np.ndarray:
    """Computes the signal PDF as a product of a truncated Crystal Ball and exponential decay.

    Parameters:
        x (float or np.ndarray): Input variable for the x-dimension.
        y (float or np.ndarray): Input variable for the y-dimension.
        params (dict): Dictionary containing the signal parameters:
                       mu, sigma, beta, m, lambda.

    Returns:
        np.ndarray: Values of the signal PDF at the given points.
    """
    gs_x = truncated_crystal_ball(
        x,
        params["mu"],
        params["sigma"],
        params["beta"],
        params["m"],
        0,
        5,
    )
    hs_y = truncated_exponential_decay(y, params["lambda"], 0, 10)
    return gs_x * hs_y


def background_pdf(
    x: float | np.ndarray,
    y: float | np.ndarray,
    params: dict,
) -> np.ndarray:
    """Computes the background PDF as a product of a uniform distribution and a truncated Gaussian.

    Parameters:
        x (float or np.ndarray): Input variable for the x-dimension.
        y (float or np.ndarray): Input variable for the y-dimension.
        params (dict): Dictionary containing the background parameters:
                       mu_b, sigma_b.

    Returns:
        np.ndarray: Values of the background PDF at the given points.
    """
    gb_x = uniform_pdf(x, 0, 5)
    hb_y = truncated_gaussian_pdf(y, params["mu_b"], params["sigma_b"], 0, 10)
    return gb_x * hb_y


def total_pdf(x: float | np.ndarray, y: float | np.ndarray, params: dict) -> np.ndarray:
    """Computes the total PDF as a weighted sum of signal and background PDFs.

    Parameters:
        x (float or np.ndarray): Input variable for the x-dimension.
        y (float or np.ndarray): Input variable for the y-dimension.
        params (dict): Dictionary containing the parameters:
                       f (signal fraction), and other signal and background parameters.

    Returns:
        np.ndarray: Values of the total PDF at the given points.
    """
    f = params["f"]
    signal = signal_pdf(x, y, params)
    background = background_pdf(x, y, params)
    return f * signal + (1 - f) * background
