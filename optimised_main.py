"""Optimised main running script with array inputs."""

# %%
from __future__ import annotations

from functools import partial
from math import erf, pi, sqrt
from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from loguru import logger
from numba import njit
from scipy.integrate import dblquad, quad


# %%
# ? Core funcs
@njit
def normal_cdf(x: float) -> float:
    """CDF of a standard normal distribution."""
    return 0.5 * (1 + erf(x / sqrt(2)))


@njit
def g_s_base(
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


# %%
# ? Parameters
mu = 3
sigma = 0.3
beta = 1
m = 1.4
f = 0.6
lam = 0.3
mu_b = 0
sigma_b = 2.5

BOUNDS_X = [0, 5]
BOUNDS_Y = [0, 10]

# Define partial functions for each PDF
g_s = partial(g_s_base, mu=mu, sigma=sigma, beta=beta, m=m, x_min=BOUNDS_X[0], x_max=BOUNDS_X[1])
h_s = partial(h_s_base, lam=lam, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])
g_b = partial(g_b_base, x_min=BOUNDS_X[0], x_max=BOUNDS_X[1])
h_b = partial(h_b_base, mu=mu_b, sigma=sigma_b, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])


# %%
# ? 2D PDFS
def signal_pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signal PDF."""
    g_s_val = g_s(x)
    h_s_val = h_s(y)
    return g_s_val * h_s_val


def background_pdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Background PDF."""
    g_b_val = g_b(x)
    h_b_val = h_b(y)
    return g_b_val * h_b_val


def total_pdf(x: np.ndarray, y: np.ndarray, f: float = 0.5) -> np.ndarray:
    """Total PDF."""
    signal = signal_pdf(x, y)
    background = background_pdf(x, y)
    return f * signal + (1 - f) * background


# %%
# ? Normalisation calculations and profiling


def integrate_function(
    func: callable,
    bounds_x: tuple,
    bounds_y: tuple | None = None,
) -> tuple[
    float,
    float,
]:
    """Generalized integration function.

    Parameters:
    - func: Callable. Function to integrate. For 2D, the function should accept y and x in that
    order.
    - bounds_x: tuple. Bounds for x, in the form (x_min, x_max).
    - bounds_y: tuple or None. Bounds for y, in the form (y_min, y_max). If None, perform 1D
    integration.

    Returns:
    - result: float. Integration result.
    - error: float. Estimated integration error.
    """
    if bounds_y is None:
        # 1D integration
        result, error = quad(lambda x: func(np.array([x]))[0], bounds_x[0], bounds_x[1])
    else:
        # 2D integration
        result, error = dblquad(
            lambda y, x: func(np.array([x]), np.array([y])),
            bounds_x[0],
            bounds_x[1],
            lambda _: bounds_y[0],
            lambda _: bounds_y[1],
        )
    return result, error


# Example usage with the signal_pdf and background_pdf:

# Integrating g_s and h_s (1D PDFs)
g_s_int, g_s_err = integrate_function(g_s, BOUNDS_X)
h_s_int, h_s_err = integrate_function(h_s, BOUNDS_Y)

# Integrating signal_pdf and background_pdf (2D PDFs)
signal_int, signal_err = integrate_function(signal_pdf, BOUNDS_X, BOUNDS_Y)
bkg_int, bkg_err = integrate_function(background_pdf, BOUNDS_X, BOUNDS_Y)

# Integrating total_pdf (2D PDF)
total_int, total_err = integrate_function(lambda x, y: total_pdf(x, y, f), BOUNDS_X, BOUNDS_Y)

# Display results
logger.info("Integration Results (Generalized):")
logger.info(f"g_s: {g_s_int} ± {g_s_err}")
logger.info(f"h_s: {h_s_int} ± {h_s_err}")
logger.info(f"Signal: {signal_int} ± {signal_err}")
logger.info(f"Background: {bkg_int} ± {bkg_err}")
logger.info(f"Total: {total_int} ± {total_err}")


# %%
# ? Core PDFs
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
# ? 1D Projections
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs = axs.ravel()

# Generate data for plotting
x_data = np.linspace(BOUNDS_X[0], BOUNDS_X[1], 200)
y_data = np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 200)

# Calculate PDFs for X
signal_x = g_s(x_data)
background_x = g_b(x_data)
total_x = f * signal_x + (1 - f) * background_x

# Plot X projections
axs[0].plot(x_data, signal_x, label="Signal (g_s)", linestyle="--")
axs[0].plot(x_data, background_x, label="Background (g_b)", linestyle="-.")
axs[0].plot(x_data, total_x, label="Total PDF", linewidth=2)
axs[0].set_title("1D Projection in X")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Density")
axs[0].legend()

# Calculate PDFs for Y
signal_y = h_s(y_data)
background_y = h_b(y_data)
total_y = f * signal_y + (1 - f) * background_y

# Plot Y projections
axs[1].plot(y_data, signal_y, label="Signal (h_s)", linestyle="--")
axs[1].plot(y_data, background_y, label="Background (h_b)", linestyle="-.")
axs[1].plot(y_data, total_y, label="Total PDF", linewidth=2)
axs[1].set_title("1D Projection in Y")
axs[1].set_xlabel("Y")
axs[1].set_ylabel("Density")
axs[1].legend()

plt.tight_layout()
plt.show()


# %%
# ? 2D Plot
x_grid, y_grid = np.meshgrid(
    np.linspace(BOUNDS_X[0], BOUNDS_X[1], 100),
    np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 100),
)

# Flatten the grid arrays for compatibility with the PDF functions
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()

# Compute the total PDF on the flattened grid
joint_pdf_flat = total_pdf(x_flat, y_flat, f)

# Reshape the result back into the grid shape
joint_pdf = joint_pdf_flat.reshape(x_grid.shape)

# Plot the joint PDF
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(
    x_grid,
    y_grid,
    joint_pdf,
    levels=50,
    cmap="viridis",
)
cbar = plt.colorbar(contour)
ax.set_title("2D Joint Probability Density")
ax.set_xlabel("X")
ax.set_ylabel("Y")
cbar.set_label("Density")

plt.tight_layout()
plt.show()

# %%
# ? Generation Timings


def total_pdf_sampler(
    n_events: int,
    f: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate events from the total PDF using vectorized rejection sampling.

    Parameters:
        n_events (int): Number of events to generate.
        f (float): Fraction parameter for the total PDF function.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of x and y samples.
    """
    accepted_x = []
    accepted_y = []

    while len(accepted_x) < n_events:
        # Generate a large pool of random samples
        x_rand = rng.uniform(
            low=BOUNDS_X[0],
            high=BOUNDS_X[1],
            size=10 * (n_events - len(accepted_x)),
        )
        y_rand = rng.uniform(
            low=BOUNDS_Y[0],
            high=BOUNDS_Y[1],
            size=10 * (n_events - len(accepted_y)),
        )
        p_rand = rng.uniform(low=0, high=1, size=10 * (n_events - len(accepted_x)))

        # Calculate PDF values for the current batch
        pdf_values = total_pdf(x=x_rand, y=y_rand, f=f)
        max_pdf_value = np.max(pdf_values)

        # Apply rejection criterion
        acceptance_mask = p_rand * max_pdf_value <= pdf_values

        # Append accepted samples
        accepted_x.extend(x_rand[acceptance_mask])
        accepted_y.extend(y_rand[acceptance_mask])

    # Convert lists to arrays and return exactly n_events samples
    return np.array(accepted_x[:n_events]), np.array(accepted_y[:n_events])


def total_pdf_wrapper(
    xy: tuple[np.ndarray, np.ndarray],
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    f: float,
    lam: float,
    mu_b: float,
    sigma_b: float,
) -> tuple[float, np.ndarray]:
    """Wrapper for the total PDF."""
    x, y = xy

    # Update partials if parameters change during fitting
    g_s = partial(
        g_s_base,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        x_min=BOUNDS_X[0],
        x_max=BOUNDS_X[1],
    )
    h_s = partial(h_s_base, lam=lam, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])
    g_b = partial(g_b_base, x_min=BOUNDS_X[0], x_max=BOUNDS_X[1])
    h_b = partial(h_b_base, mu=mu_b, sigma=sigma_b, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])

    # Compute PDF values
    pdf_values = f * (g_s(x) * h_s(y)) + (1 - f) * (g_b(x) * h_b(y))
    return len(x), pdf_values


def perform_fit(sampled_x: np.array, sampled_y: np.array) -> None:
    """Complete minuit fit."""
    nll = ExtendedUnbinnedNLL((sampled_x, sampled_y), total_pdf_wrapper)
    minuit = Minuit(
        nll,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        f=f,
        lam=lam,
        mu_b=mu_b,
        sigma_b=sigma_b,
    )
    minuit.limits["beta"] = (0, None)
    minuit.limits["m"] = (1, None)
    minuit.migrad()
    minuit.hesse()


rng = np.random.default_rng(seed=451)
n_events = 100_000
n_runs = 1


sampled_x, sampled_y = total_pdf_sampler(
    n_events=n_events,
    f=f,
    rng=rng,
)

# Benchmark timings
normal_time = timeit(lambda: rng.normal(size=n_events), number=n_runs) / n_runs
sample_generation_time = (
    timeit(
        lambda: total_pdf_sampler(
            n_events=n_events,
            f=f,
            rng=rng,
        ),
        number=n_runs,
    )
    / n_runs
)
fit_time = (
    timeit(
        lambda: perform_fit(sampled_x=sampled_x, sampled_y=sampled_y),
        number=n_runs,
    )
    / n_runs
)

# Log benchmark results
logger.info(f"Benchmark Results (averaged over {n_runs} runs):")
logger.info(f"(i) np.random.normal: {normal_time:.6f} s")
logger.info(
    f"(ii) Sample generation: {sample_generation_time:.6f} s (relative: {(sample_generation_time / normal_time):.2f})",  # noqa: E501
)
logger.info(f"(iii) Fit execution: {fit_time:.6f} s (relative: {(fit_time / normal_time):.2f})")
# %%
# ? Parametric Bootstrapping

sample_sizes = [500, 1000, 2500, 5000, 10000]
n_bootstrap = 250

lam_results = {size: {"values": [], "errors": []} for size in sample_sizes}


def nll(lam: float, y_data: np.ndarray) -> float:
    """Calculate the negative log-likelihood for fitting lambda."""
    pdf_y = h_s(y=y_data, lam=lam)
    return -np.sum(np.log(pdf_y))


for n_events in sample_sizes:
    lambda_estimates = []
    rng = np.random.default_rng(seed=451)

    for _ in range(n_bootstrap):
        sampled_x, sampled_y = total_pdf_sampler(
            n_events=n_events,
            f=f,
            rng=rng,
        )

        minuit = Minuit(
            lambda lam: nll(lam=lam, y_data=sampled_y),  # noqa: B023
            lam=lam,
        )
        minuit.limits["lam"] = (0, 10)
        minuit.migrad()
        if minuit.valid:
            lam_results[n_events]["values"].append(minuit.values["lam"])  # noqa: PD011
            lam_results[n_events]["errors"].append(minuit.errors["lam"])


# Calculate bias and uncertainty from results dictionary
bias_results = [np.mean(lam_results[size]["values"]) - lam for size in sample_sizes]
uncertainty_results = [np.mean(lam_results[size]["errors"]) for size in sample_sizes]

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Bias plot
axs[0].plot(sample_sizes, bias_results, marker="o")
axs[0].set_title("Bias in λ vs Sample Size")
axs[0].set_xlabel("Sample Size")
axs[0].set_ylabel("Bias")
axs[0].grid(visible=True)

# Uncertainty plot
axs[1].plot(sample_sizes, uncertainty_results, marker="o")
axs[1].set_title("Uncertainty in λ vs Sample Size")
axs[1].set_xlabel("Sample Size")
axs[1].set_ylabel("Uncertainty")
axs[1].grid(visible=True)

plt.tight_layout()
plt.show()

# %%
