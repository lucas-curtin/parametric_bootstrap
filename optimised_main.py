"""Optimised main running script with array inputs."""

# %%
import matplotlib.pyplot as plt
import numba
import numpy as np
from loguru import logger
from scipy.integrate import quad


@numba.jit
def numba_erf(x: np.ndarray) -> np.ndarray:
    """Error function approximation compatible with Numba."""
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-(x**2))))


@numba.jit
def cumulative_normal(x: np.ndarray) -> np.ndarray:
    """Cumulative density function for the standard normal distribution."""
    return 0.5 * (1 + numba_erf(x / np.sqrt(2)))


@numba.jit
def g_s(  # noqa: PLR0913
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
    a = (m / np.abs(beta)) ** m * np.exp(-(beta**2) / 2)
    b = m / np.abs(beta) - np.abs(beta)
    pdf[beta_minus] = a * (b - z[beta_minus]) ** (-m)
    z_min = (x_min - mu) / sigma
    z_max = (x_max - mu) / sigma
    gauss_min = max(-beta, z_min)
    gauss_max = z_max
    gauss_contrib = cumulative_normal(gauss_max) - cumulative_normal(gauss_min)
    tail_min = z_min
    tail_max = min(-beta, z_max)
    tail_contrib = 0.0
    if tail_min < -beta:
        tail_contrib += (
            (m / np.abs(beta)) ** m
            * np.exp(-(beta**2) / 2)
            / (m - 1)
            * ((tail_max + beta) ** (1 - m) - (tail_min + beta) ** (1 - m))
        )
    normalisation = sigma * (np.sqrt(2 * np.pi) * gauss_contrib + tail_contrib)
    return pdf / normalisation


@numba.jit
def h_s(y: np.ndarray, lam: float, y_min: float, y_max: float) -> np.ndarray:
    """Normalised Truncated Exponential PDF."""
    pdf = np.zeros_like(y)
    within_bounds = (y >= y_min) & (y <= y_max)
    y = y[within_bounds]
    normalisation = np.exp(-lam * y_min) - np.exp(-lam * y_max)
    trunc_pdf = lam * np.exp(-lam * y) / normalisation
    pdf[within_bounds] = trunc_pdf
    return pdf


@numba.jit
def g_b(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Normalised Truncated Uniform PDF."""
    pdf = np.zeros_like(x)
    within_bounds = (x >= x_min) & (x <= x_max)
    normalisation = x_max - x_min
    pdf[within_bounds] = 1.0 / normalisation
    return pdf


@numba.jit
def h_b(y: np.ndarray, mu: float, sigma: float, y_min: float, y_max: float) -> np.ndarray:
    """Normalised Truncated Normal PDF."""
    pdf = np.zeros_like(y)
    within_bounds = (y >= y_min) & (y <= y_max)
    y = y[within_bounds]
    z_min = (y_min - mu) / sigma
    z_max = (y_max - mu) / sigma
    normalisation = cumulative_normal(z_max) - cumulative_normal(z_min)
    z = (y - mu) / sigma
    trunc_pdf = np.exp(-(z**2) / 2) / (sigma * np.sqrt(2 * np.pi) * normalisation)
    pdf[within_bounds] = trunc_pdf
    return pdf


# ? Normalisation calculations and profiling
# %%
BOUNDS_X = [0, 5]
BOUNDS_Y = [0, 10]

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
    "g_s": {
        "mu": mu,
        "sigma": sigma,
        "beta": beta,
        "m": m,
        "x_min": BOUNDS_X[0],
        "x_max": BOUNDS_X[1],
    },
    "h_s": {"lam": lam, "y_min": BOUNDS_Y[0], "y_max": BOUNDS_Y[1]},
    "g_b": {"x_min": BOUNDS_X[0], "x_max": BOUNDS_X[1]},
    "h_b": {"mu": mu_b, "sigma": sigma_b, "y_min": BOUNDS_Y[0], "y_max": BOUNDS_Y[1]},
}

# %%
# Create figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

# Plot g_s (Truncated Crystal Ball)
x_data = np.linspace(BOUNDS_X[0], BOUNDS_X[1], 200)
y_data = np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 200)


g_s_values = g_s(x_data, mu, sigma, beta, m, BOUNDS_X[0], BOUNDS_X[1])
axs[0].plot(x_data, g_s_values)
axs[0].set_title("Truncated Crystal Ball PDF (g_s)")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Density")

# Plot h_s (Truncated Exponential)
h_s_values = h_s(y_data, lam, BOUNDS_Y[0], BOUNDS_Y[1])
axs[1].plot(y_data, h_s_values)
axs[1].set_title("Truncated Exponential PDF (h_s)")
axs[1].set_xlabel("Y")
axs[1].set_ylabel("Density")

# Plot g_b (Uniform)
g_b_values = g_b(x_data, BOUNDS_X[0], BOUNDS_X[1])
axs[2].plot(x_data, g_b_values)
axs[2].set_title("Uniform PDF (g_b)")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Density")

# Plot h_b (Truncated Normal)
h_b_values = h_b(y_data, mu_b, sigma_b, BOUNDS_Y[0], BOUNDS_Y[1])
axs[3].plot(y_data, h_b_values)
axs[3].set_title("Truncated Normal PDF (h_b)")
axs[3].set_xlabel("Y")
axs[3].set_ylabel("Density")

plt.tight_layout()
plt.show()

# %%


result_g_s, _ = quad(
    lambda x: g_s(np.array([x]), mu, sigma, beta, m, BOUNDS_X[0], BOUNDS_X[1])[0],
    BOUNDS_X[0],
    BOUNDS_X[1],
)

result_h_s, _ = quad(
    lambda y: h_s(np.array([y]), lam, BOUNDS_Y[0], BOUNDS_Y[1])[0],
    BOUNDS_Y[0],
    BOUNDS_Y[1],
)

result_g_b, _ = quad(
    lambda x: g_b(np.array([x]), BOUNDS_X[0], BOUNDS_X[1])[0],
    BOUNDS_X[0],
    BOUNDS_X[1],
)
result_h_b, _ = quad(
    lambda y: h_b(np.array([y]), mu, sigma, BOUNDS_Y[0], BOUNDS_Y[1])[0],
    BOUNDS_Y[0],
    BOUNDS_Y[1],
)

# Display results
logger.info("Integration Results:")

logger.info(f"g_s: {result_g_s}")
logger.info(f"h_s: {result_h_s}")
logger.info(f"g_b: {result_g_b}")
logger.info(f"h_b: {result_h_b}")


# %%
