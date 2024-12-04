"""Optimised main running script with array inputs."""

# %%
from __future__ import annotations

from math import erf, pi, sqrt
from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from loguru import logger
from numba import jit
from scipy.integrate import dblquad, quad
from scipy.stats import poisson


# %%
# ? Core funcs
@jit(nopython=True)
def normal_cdf(x: float) -> float:
    """CDF of a standard normal distribution."""
    return 0.5 * (1 + erf(x / sqrt(2)))


@jit(nopython=True)
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


@jit(nopython=True)
def h_s(y: np.ndarray, lam: float, y_min: float, y_max: float) -> np.ndarray:
    """Normalised Truncated Exponential PDF."""
    pdf = np.zeros_like(y)
    within_bounds = (y >= y_min) & (y <= y_max)
    y = y[within_bounds]
    normalisation = np.exp(-lam * y_min) - np.exp(-lam * y_max)
    trunc_pdf = lam * np.exp(-lam * y) / normalisation
    pdf[within_bounds] = trunc_pdf
    return pdf


@jit(nopython=True)
def g_b(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Normalised Truncated Uniform PDF."""
    pdf = np.zeros_like(x)
    within_bounds = (x >= x_min) & (x <= x_max)
    normalisation = x_max - x_min
    pdf[within_bounds] = 1.0 / normalisation
    return pdf


@jit(nopython=True)
def h_b(y: np.ndarray, mu: float, sigma: float, y_min: float, y_max: float) -> np.ndarray:
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
# ? 2D PDFS
def signal_pdf(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """Signal PDF."""
    g_s_val = g_s(x, **params["g_s"])
    h_s_val = h_s(y, **params["h_s"])
    return g_s_val * h_s_val


def background_pdf(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """Background PDF."""
    g_b_val = g_b(x, **params["g_b"])
    h_b_val = h_b(y, **params["h_b"])
    return g_b_val * h_b_val


def total_pdf(x: np.ndarray, y: np.ndarray, f: float, params: dict) -> np.ndarray:
    """Total PDF."""
    signal = signal_pdf(x, y, params)
    background = background_pdf(x, y, params)
    return f * signal + (1 - f) * background


# %%
# ? Normalisation calculations and profiling


g_s_int, _ = quad(
    lambda x: g_s(np.array([x]), mu, sigma, beta, m, BOUNDS_X[0], BOUNDS_X[1]),
    BOUNDS_X[0],
    BOUNDS_X[1],
)

h_s_int, _ = quad(
    lambda y: h_s(np.array([y]), lam, BOUNDS_Y[0], BOUNDS_Y[1])[0],
    BOUNDS_Y[0],
    BOUNDS_Y[1],
)

g_b_int, _ = quad(
    lambda x: g_b(np.array([x]), BOUNDS_X[0], BOUNDS_X[1])[0],
    BOUNDS_X[0],
    BOUNDS_X[1],
)
h_b_int, _ = quad(
    lambda y: h_b(np.array([y]), mu, sigma, BOUNDS_Y[0], BOUNDS_Y[1])[0],
    BOUNDS_Y[0],
    BOUNDS_Y[1],
)

signal_int, _ = dblquad(
    lambda y, x: signal_pdf(np.array([x]), np.array([y]), params),
    params["g_s"]["x_min"],
    params["g_s"]["x_max"],
    lambda _: params["h_s"]["y_min"],
    lambda _: params["h_s"]["y_max"],
)

bkg_int, _ = dblquad(
    lambda y, x: background_pdf(np.array([x]), np.array([y]), params),
    params["g_b"]["x_min"],
    params["g_b"]["x_max"],
    lambda _: params["h_b"]["y_min"],
    lambda _: params["h_b"]["y_max"],
)

total_int, _ = dblquad(
    lambda y, x: total_pdf(np.array([x]), np.array([y]), f, params),
    params["g_b"]["x_min"],
    params["g_b"]["x_max"],
    lambda _: params["h_b"]["y_min"],
    lambda _: params["h_b"]["y_max"],
)

# Display results
logger.info("Integration Results:")

logger.info(f"g_s: {g_s_int}")
logger.info(f"h_s: {h_s_int}")
logger.info(f"g_b: {g_b_int}")
logger.info(f"h_b: {h_b_int}")
logger.info(f"h_s: {signal_int}")
logger.info(f"g_b: {bkg_int}")
logger.info(f"h_b: {total_int}")


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
signal_x = g_s(x_data, **params["g_s"])
background_x = g_b(x_data, **params["g_b"])
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
signal_y = h_s(y_data, **params["h_s"])
background_y = h_b(y_data, **params["h_b"])
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
joint_pdf_flat = total_pdf(x_flat, y_flat, f, params)

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


def generate_sample_from_total_pdf(
    n_events: int,
    params: dict,
    f: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate events from the total PDF."""
    x = rng.uniform(low=BOUNDS_X[0], high=BOUNDS_X[1], size=n_events)
    y = rng.uniform(low=BOUNDS_Y[0], high=BOUNDS_Y[1], size=n_events)
    pdf_values = total_pdf(x=x, y=y, f=f, params=params)
    max_pdf_value = np.max(pdf_values)

    sampled_x = []
    sampled_y = []

    for i in range(n_events):
        x_rand = rng.uniform(low=BOUNDS_X[0], high=BOUNDS_X[1])
        y_rand = rng.uniform(low=BOUNDS_Y[0], high=BOUNDS_Y[1])
        p_rand = rng.uniform(low=0, high=max_pdf_value)

        if p_rand <= total_pdf(x=np.array([x_rand]), y=np.array([y_rand]), f=f, params=params):
            sampled_x.append(x_rand)
            sampled_y.append(y_rand)

    return np.array(sampled_x), np.array(sampled_y)


def total_pdf_wrapper(  # noqa: PLR0913
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
    pdf_values = total_pdf(x=x, y=y, f=f, params=params)
    return len(x), pdf_values


def perform_fit() -> None:
    """Complete minuit fit."""
    sampled_x, sampled_y = generate_sample_from_total_pdf(
        n_events=n_events,
        params=params,
        f=f,
        rng=rng,
    )
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

n_runs = 100


normal_time = timeit(lambda: rng.normal(size=n_events), number=n_runs) / n_runs


sample_generation_time = (
    timeit(
        lambda: generate_sample_from_total_pdf(
            n_events=n_events,
            params=params,
            f=f,
            rng=rng,
        ),
        number=n_runs,
    )
    / n_runs
)


fit_time = timeit(perform_fit, number=n_runs) / n_runs

# Results
relative_generation_time = sample_generation_time / normal_time
relative_fit_time = fit_time / normal_time

logger.info(f"Benchmark Results (averaged over {n_runs} runs):")
logger.info(f"(i) np.random.normal: {normal_time:.6f} s")
logger.info(
    f"(ii) Sample generation: {sample_generation_time:.6f} s (relative: {relative_generation_time:.2f})",  # noqa: E501
)
logger.info(f"(iii) Fit execution: {fit_time:.6f} s (relative: {relative_fit_time:.2f})")
# %%
# ? Lam test
true_params = {
    "lam": 0.3,
    "mu": 3,
    "sigma": 0.3,
    "beta": 1,
    "m": 1.4,
    "f": 0.6,
    "mu_b": 0,
    "sigma_b": 2.5,
}

sample_sizes = [500, 1000, 2500, 5000, 10000]
n_bootstrap = 250

# Store results
results = {size: {"lambda_estimates": [], "sample_sizes": []} for size in sample_sizes}

# Generate and fit function
rng = np.random.default_rng(seed=42)


def fit_data(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit the model to the data and return the estimate for lambda and its uncertainty."""

    def pdf_wrapper(xy: tuple[np.ndarray, np.ndarray], lam: float) -> np.ndarray:
        """Wrapper for the total PDF to evaluate it for given lambda."""
        params = true_params.copy()
        params["lam"] = lam
        x, y = xy
        return total_pdf(x, y, f=true_params["f"], params=params)

    nll = ExtendedUnbinnedNLL((x, y), pdf_wrapper)
    minuit = Minuit(nll, lam=true_params["lam"])
    minuit.limits["lam"] = (0, None)
    minuit.migrad()
    return minuit.to_numpy()["lam"], minuit.errors["lam"]


for size in sample_sizes:
    for _ in range(n_bootstrap):
        # Sample size with Poisson variation
        actual_size = poisson.rvs(size, random_state=rng)
        results[size]["sample_sizes"].append(actual_size)

        # Generate data
        x, y = generate_sample_from_total_pdf(actual_size, params, f, rng)

        # Fit data
        lam_est, lam_err = fit_data(x, y)
        results[size]["lambda_estimates"].append(lam_est)

# Calculate bias and uncertainty
bias = []
uncertainty = []

for size in sample_sizes:
    lam_estimates = results[size]["lambda_estimates"]
    bias.append(np.mean(lam_estimates) - true_params["lam"])
    uncertainty.append(np.std(lam_estimates))
# %%
