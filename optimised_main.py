"""Main script for analysis."""

# %%
# ? Imports and setup
from __future__ import annotations

import matplotlib.pyplot as plt
import numba
import numpy as np
from loguru import logger
from scipy.integrate import dblquad

# %%
# ? Constants
BOUNDS_X: tuple[float, float] = (0, 5)
BOUNDS_Y: tuple[float, float] = (0, 10)

# %%
# ? Define PDFs


@numba.njit
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
    # Simplified Crystal Ball approximation
    z = (x - mu) / sigma
    if z > -beta:
        return np.exp(-z * z / 2)
    A = (m / np.abs(beta)) ** m * np.exp(-beta * beta / 2)
    B = m / np.abs(beta) - np.abs(beta) - z
    return A * (B) ** (-m)


@numba.njit
def h_s(y: float, lam: float) -> float:
    """Exponential PDF.

    Args:
        y: Input variable.
        lam: Rate parameter.

    Returns:
        Probability density value.
    """
    return np.exp(-y * lam)


@numba.njit
def g_b(x: float, x_min: float, x_range: float) -> float:
    """Uniform PDF.

    Args:
        x: Input variable.
        x_min: Minimum value.
        x_range: Range (max - min).

    Returns:
        Probability density value.
    """
    return 1.0 / x_range if x_min <= x <= x_min + x_range else 0.0


@numba.njit
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
    if y_min <= y <= y_max:
        z = (y - mu_b) / sigma_b
        return np.exp(-z * z / 2) / (sigma_b * np.sqrt(2 * np.pi))
    return 0.0


@numba.njit
def signal(x: float, y: float, mu: float, sigma: float, beta: float, m: float, lam: float) -> float:
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
    return g_s(x=x, mu=mu, sigma=sigma, beta=beta, m=m) * h_s(y=y, lam=lam)


@numba.njit
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
    return g_b(x=x, x_min=BOUNDS_X[0], x_range=BOUNDS_X[1] - BOUNDS_X[0]) * h_b(
        y=y,
        mu_b=mu_b,
        sigma_b=sigma_b,
        y_min=BOUNDS_Y[0],
        y_max=BOUNDS_Y[1],
    )


@numba.njit
def total_pdf(
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
    return f * signal(x=x, y=y, mu=mu, sigma=sigma, beta=beta, m=m, lam=lam) + (1 - f) * background(
        x=x,
        y=y,
        mu_b=mu_b,
        sigma_b=sigma_b,
    )


# %%
# ? Normalisation calculations and profiling
# Parameters
mu = 3
sigma = 0.3
beta = 1
m = 1.4
f = 0.6
lam = 0.3
mu_b = 0
sigma_b = 2.5

params = {
    "g_s": {"mu": mu, "sigma": sigma, "beta": beta, "m": m},
    "h_s": {"lam": lam},
    "g_b": {"x_min": BOUNDS_X[0], "x_range": BOUNDS_X[1] - BOUNDS_X[0]},
    "h_b": {"mu_b": mu_b, "sigma_b": sigma_b, "y_min": BOUNDS_Y[0], "y_max": BOUNDS_Y[1]},
}
# %%
# Create figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

# Plot g_s (Crystal Ball)
x_g_s = np.linspace(BOUNDS_X[0], BOUNDS_X[1], 200)
axs[0].plot(x_g_s, [g_s(x, **params["g_s"]) for x in x_g_s])
axs[0].set_title("Crystal Ball PDF (g_s)")

# Plot h_s (Exponential)
y_h_s = np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 200)
axs[1].plot(y_h_s, [h_s(y, **params["h_s"]) for y in y_h_s])
axs[1].set_title("Exponential PDF (h_s)")

# Plot g_b (Uniform)
x_g_b = np.linspace(BOUNDS_X[0], BOUNDS_X[1], 200)
axs[2].plot(x_g_b, [g_b(x, **params["g_b"]) for x in x_g_b])
axs[2].set_title("Uniform PDF (g_b)")

# Plot h_b (Truncated Normal)
y_h_b = np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 200)
axs[3].plot(y_h_b, [h_b(y, **params["h_b"]) for y in y_h_b])
axs[3].set_title("Truncated Normal PDF (h_b)")

plt.tight_layout()
plt.show()

# %%
# Compute normalisations


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
