"""Main script for analysis."""

# %%
# ? Imports and setup
from __future__ import annotations

from loguru import logger
from scipy.integrate import dblquad
from scipy.stats import crystalball, expon, truncnorm, uniform

# %%
# ? Constants
BOUNDS_X: tuple[float, float] = (0, 5)
BOUNDS_Y: tuple[float, float] = (0, 10)

# %%
# ? Define PDFs


def g_s(x: float, mu: float, sigma: float, beta: float, m: float) -> float:
    """Crystal Ball PDF.

    Args:
        x: Input variable.
        mu: Mean of the distribution.
        sigma: Standard deviation of the distribution.
        beta: Tail parameter.
        m: Exponent parameter.

    Returns:
        Probability density value.
    """
    return crystalball.pdf(x, beta=beta, m=m, loc=mu, scale=sigma)


def h_s(y: float, lam: float) -> float:
    """Exponential PDF.

    Args:
        y: Input variable.
        lam: Rate parameter.

    Returns:
        Probability density value.
    """
    return expon.pdf(y, loc=0, scale=1 / lam)


def g_b(x: float, x_min: float, x_range: float) -> float:
    """Uniform PDF.

    Args:
        x: Input variable.
        x_min: Minimum value.
        x_range: Range (max - min).

    Returns:
        Probability density value.
    """
    return uniform.pdf(x, loc=x_min, scale=x_range)


def h_b(y: float, mu_b: float, sigma_b: float, y_min: float, y_max: float) -> float:
    """Truncated Normal PDF.

    Args:
        y: Input variable.
        mu_b: Mean of the distribution.
        sigma_b: Standard deviation of the distribution.
        y_min: Minimum truncation bound.
        y_max: Maximum truncation bound.

    Returns:
        Probability density value.
    """
    a: float = (y_min - mu_b) / sigma_b
    b: float = (y_max - mu_b) / sigma_b
    return truncnorm.pdf(y, a=a, b=b, loc=mu_b, scale=sigma_b)


def signal(x: float, y: float, mu: float, sigma: float, beta: float, m: float, lam: float) -> float:  # noqa: PLR0913
    """Signal PDF.

    Args:
        x: X coordinate.
        y: Y coordinate.
        mu: Mean of the signal distribution.
        sigma: Standard deviation of the signal distribution.
        beta: Signal tail parameter.
        m: Signal exponent parameter.
        lam: Signal decay rate.

    Returns:
        Signal probability density value.
    """
    return g_s(x, mu, sigma, beta, m) * h_s(y, lam)


def background(x: float, y: float, mu_b: float, sigma_b: float) -> float:
    """Background PDF.

    Args:
        x: X coordinate.
        y: Y coordinate.
        mu_b: Background mean.
        sigma_b: Background standard deviation.

    Returns:
        Background probability density value.
    """
    return g_b(x, *BOUNDS_X) * h_b(y, mu_b, sigma_b, *BOUNDS_Y)


def total_pdf(  # noqa: PLR0913
    x: float,
    y: float,
    f: float,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    lam: float,
    mu_b: float,
    sigma_b: float,
) -> float:
    """Total combined PDF.

    Args:
        x: X coordinate.
        y: Y coordinate.
        f: Signal fraction.
        mu: Mean of the signal distribution.
        sigma: Standard deviation of the signal distribution.
        beta: Signal tail parameter.
        m: Signal exponent parameter.
        lam: Signal decay rate.
        mu_b: Background mean.
        sigma_b: Background standard deviation.

    Returns:
        Combined probability density value.
    """
    return f * signal(x, y, mu, sigma, beta, m, lam) + (1 - f) * background(x, y, mu_b, sigma_b)


# %%
# ? Normalisation calculations

# Assign parameters directly to variables
mu = 3
sigma = 0.3
beta = 1
m = 1.4
f = 0.6
lam = 0.3
mu_b = 0
sigma_b = 2.5

# Compute the normalizations
signal_norm, _ = dblquad(
    lambda y, x: signal(x=x, y=y, mu=mu, sigma=sigma, beta=beta, m=m, lam=lam),
    BOUNDS_X[0],
    BOUNDS_X[1],
    lambda _: BOUNDS_Y[0],
    lambda _: BOUNDS_Y[1],
)

background_norm, _ = dblquad(
    lambda y, x: background(x=x, y=y, mu_b=mu_b, sigma_b=sigma_b),
    BOUNDS_X[0],
    BOUNDS_X[1],
    lambda _: BOUNDS_Y[0],
    lambda _: BOUNDS_Y[1],
)

total_norm, _ = dblquad(
    lambda y, x: total_pdf(
        x=x,
        y=y,
        f=f,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        lam=lam,
        mu_b=mu_b,
        sigma_b=sigma_b,
    ),
    BOUNDS_X[0],
    BOUNDS_X[1],
    lambda _: BOUNDS_Y[0],
    lambda _: BOUNDS_Y[1],
)

logger.info(f"Normalisation of Signal: {signal_norm:.6f}")
logger.info(f"Normalisation of Background: {background_norm:.6f}")
logger.info(f"Normalisation of Total: {total_norm:.6f}")
# %%
