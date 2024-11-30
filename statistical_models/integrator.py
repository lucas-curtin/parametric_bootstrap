"""Handles integration of PDFs for normalisation checks and other computations."""

from __future__ import annotations

from typing import Callable

from scipy.integrate import dblquad

from .pdfs import background_pdf, signal_pdf, total_pdf


def integrate_pdf(
    pdf: Callable[[float, float], float],
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
) -> float:
    """Numerically integrates a given two-dimensional PDF over specified bounds.

    Parameters:
        pdf (Callable[[float, float], float]): A two-dimensional PDF to integrate.
        x_lower (float): Lower bound of integration for the x-dimension.
        x_upper (float): Upper bound of integration for the x-dimension.
        y_lower (float): Lower bound of integration for the y-dimension.
        y_upper (float): Upper bound of integration for the y-dimension.

    Returns:
        float: Result of the numerical integration.
    """
    result, _ = dblquad(
        lambda y, x: pdf(x, y),
        x_lower,
        x_upper,
        lambda x: y_lower,
        lambda x: y_upper,
    )
    return result


def check_normalisation(params: dict) -> tuple[float, float, float]:
    """Checks the normalisation of the signal, background, and total PDFs.

    Parameters:
        params (dict): Dictionary containing the parameters for the signal and background PDFs:
                       - Signal parameters: mu, sigma, beta, m, lambda.
                       - Background parameters: mu_b, sigma_b.
                       - Total parameters: f (signal fraction).

    Returns:
        tuple[float, float, float]: A tuple containing:
            - Signal PDF normalisation (float).
            - Background PDF normalisation (float).
            - Total PDF normalisation (float).
    """
    signal_norm = integrate_pdf(lambda x, y: signal_pdf(x, y, params), 0, 5, 0, 10)
    background_norm = integrate_pdf(lambda x, y: background_pdf(x, y, params), 0, 5, 0, 10)
    total_norm = integrate_pdf(lambda x, y: total_pdf(x, y, params), 0, 5, 0, 10)
    return signal_norm, background_norm, total_norm
