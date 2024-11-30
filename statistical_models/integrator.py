"""Handles integration of PDFs for normalisation checks and other computations."""

from __future__ import annotations

from typing import Callable

from loguru import logger
from scipy.integrate import dblquad, quad


def integrate_pdf(
    pdf: Callable,
    x_lower: float | None = None,
    x_upper: float | None = None,
    y_lower: float | None = None,
    y_upper: float | None = None,
) -> float:
    """Determines whether to perform single or double integration based on the input.

    Parameters:
        pdf (Callable): A single-variable or two-variable PDF to integrate.
        x_lower (Optional[float]): Lower bound of integration for the x-dimension.
        x_upper (Optional[float]): Upper bound of integration for the x-dimension.
        y_lower (Optional[float]): Lower bound of integration for the y-dimension (if 2D).
        y_upper (Optional[float]): Upper bound of integration for the y-dimension (if 2D).

    Returns:
        float: Result of the numerical integration.
    """
    if y_lower is not None and x_lower is not None:
        # Perform double integration if y bounds are provided
        logger.info("Double Integration")
        return integrate_double_pdf(pdf, x_lower, x_upper, y_lower, y_upper)
    # Perform single integration if only x bounds are provided
    if x_lower is not None:
        logger.info("X Integration")
        return integrate_single_pdf(pdf, x_lower, x_upper)
    logger.info("Y Integration")
    return integrate_single_pdf(pdf, y_lower, y_upper)


def integrate_single_pdf(
    pdf: Callable[[float], float],
    lower: float,
    upper: float,
) -> float:
    """Numerically integrates a single-variable PDF over specified bounds.

    Parameters:
        pdf (Callable[[float], float]): A single-variable PDF to integrate.
        lower (float): Lower bound of integration.
        upper (float): Upper bound of integration.

    Returns:
        float: Result of the numerical integration.
    """
    result, _ = quad(pdf, lower, upper)
    return result


def integrate_double_pdf(
    pdf: Callable[[float, float], float],
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
) -> float:
    """Numerically integrates a two-dimensional PDF over specified bounds.

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
