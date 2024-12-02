"""Handles our pdf class."""

from __future__ import annotations

import numpy as np
from scipy.integrate import nquad
from scipy.stats import norm
from scipy.stats import uniform as scipy_uniform


class PDF:
    """A base class for probability density functions."""

    def __init__(
        self,
        func: callable,
        bounds: list[tuple],
    ) -> None:
        """Create our pdf object with bounds."""
        self.func = func  # The PDF function
        self.bounds = bounds  # List of bounds, one tuple per dimension

    def evaluate(self, *args: float) -> float:
        """Evaluate the PDF at given values."""
        return self.func(*args)

    def integrate(self) -> float:
        """Integrate the PDF over its specified bounds."""
        if not self.bounds:
            msg = "Integration bounds not specified."
            raise ValueError(msg)

        # Use n-dimensional integration based on the number of bounds provided
        return nquad(self.func, self.bounds)[0]

    def normalise(self) -> PDF:
        """Return a normalised PDF over its bounds.

        This ensures the total integral of the PDF over the specified bounds is equal to 1.
        """
        integral = self.integrate()
        if integral == 0:
            msg = "Cannot normalise a PDF with zero integral."
            raise ValueError(msg)

        # Define a new function that represents the normalised PDF
        def normalised_func(*args: float) -> float:
            return self.func(*args) / integral

        # Return a new PDF object with the normalised function
        return PDF(normalised_func, bounds=self.bounds)

    def __mul__(self, other: PDF | float) -> PDF:
        """Multiply the PDF with another PDF or a scalar."""
        if isinstance(other, PDF):

            def multiplied_func(*args: float) -> float:
                """Multiply our function."""
                return self.evaluate(*args) * other.evaluate(*args)

            return PDF(multiplied_func, bounds=self.bounds)

        if isinstance(other, (float, int)):

            def scaled_func(*args: float) -> float:
                """Scale our function."""
                return self.evaluate(*args) * other

            return PDF(scaled_func, bounds=self.bounds)

        msg = "Can only multiply with another PDF or a scalar."
        raise ValueError(msg)

    def __rmul__(self, other: float) -> PDF:
        """Handle scalar multiplication from the left."""
        return self.__mul__(other)

    def __add__(self, other: PDF | float) -> PDF:
        """Add the PDF with another PDF or a scalar."""
        if isinstance(other, PDF):

            def added_func(*args: float) -> float:
                """Add our PDFs together."""
                return self.evaluate(*args) + other.evaluate(*args)

            return PDF(added_func, bounds=self.bounds)

        if isinstance(other, (float, int)):

            def shifted_func(*args: float) -> float:
                """Add a constant to our PDF."""
                return self.evaluate(*args) + other

            return PDF(shifted_func, bounds=self.bounds)

        msg = "Can only add with another PDF or a scalar."
        raise ValueError(msg)

    def __radd__(self, other: float) -> PDF:
        """Handle scalar addition from the left."""
        return self.__add__(other)

    def __sub__(self, other: PDF | float) -> PDF:
        """Subtract another PDF or a scalar from this PDF."""
        if isinstance(other, PDF):

            def subtracted_func(*args: float) -> float:
                """Subtract our PDFs."""
                return self.evaluate(*args) - other.evaluate(*args)

            return PDF(subtracted_func, bounds=self.bounds)

        if isinstance(other, (float, int)):

            def shifted_func(*args: float) -> float:
                """Subtract a constant from our PDF."""
                return self.evaluate(*args) - other

            return PDF(shifted_func, bounds=self.bounds)

        msg = "Can only subtract another PDF or a scalar."
        raise ValueError(msg)

    def __rsub__(self, other: float) -> PDF:
        """Handle scalar subtraction from the left."""
        if isinstance(other, (float, int)):

            def shifted_func(*args: float) -> float:
                """Subtract PDF from a constant."""
                return other - self.evaluate(*args)

            return PDF(shifted_func, bounds=self.bounds)

        msg = "Can only subtract another PDF or a scalar."
        raise ValueError(msg)


# Hardcoded bounds
BOUNDS_X = (0, 5)
BOUNDS_Y = (0, 10)


def crystal_ball(x: float, mu: float, sigma: float, beta: float, m: float) -> float:
    """Crystal Ball probability density function (PDF)."""
    z = (x - mu) / sigma

    if z > -beta:
        return np.exp(-(z**2) / 2) * (z > -beta)

    a = ((m / beta) ** m) * np.exp(-(beta**2) / 2)
    b = (m / beta) - beta - z
    return a * (b ** (-m)) * (z <= -beta)


def exponential_decay(x: float, lam: float) -> float:
    """Exponential decay PDF."""
    return lam * np.exp(-lam * x)


def truncated_gaussian(x: float, mu: float, sigma: float, lower: float, upper: float) -> float:
    """Truncated Gaussian PDF."""
    norm_factor = norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)
    return norm.pdf(x, mu, sigma) / norm_factor


def custom_uniform(x: float, x_lower: float, x_upper: float) -> float:
    """Custom uniform PDF."""
    scale = x_upper - x_lower
    return scipy_uniform.pdf(x, loc=x_lower, scale=scale)


# Pre-defined PDFs
def g_s_pdf(params: dict) -> PDF:
    """Truncated Crystal Ball PDF (g_s(X))."""
    return PDF(
        func=lambda x: crystal_ball(
            x=x,
            mu=params["mu"],
            sigma=params["sigma"],
            beta=params["beta"],
            m=params["m"],
        ),
        bounds=[BOUNDS_X],
    ).normalise()


def h_s_pdf(params: dict) -> PDF:
    """Truncated exponential decay PDF (h_s(Y))."""
    return PDF(
        func=lambda x: exponential_decay(x=x, lam=params["lam"]),
        bounds=[BOUNDS_Y],
    ).normalise()


def g_b_pdf() -> PDF:
    """Uniform PDF (g_b(X))."""
    return PDF(
        func=lambda x: custom_uniform(x=x, x_lower=BOUNDS_X[0], x_upper=BOUNDS_X[1]),
        bounds=[BOUNDS_X],
    ).normalise()


def h_b_pdf(params: dict) -> PDF:
    """Truncated Gaussian PDF (h_b(Y))."""
    return PDF(
        func=lambda y: truncated_gaussian(
            y,
            params["mu_b"],
            params["sigma_b"],
            BOUNDS_Y[0],
            BOUNDS_Y[1],
        ),
        bounds=[BOUNDS_Y],
    ).normalise()
