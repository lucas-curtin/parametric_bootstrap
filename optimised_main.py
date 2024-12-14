"""Optimised main running script with array inputs."""

# %%
from __future__ import annotations

from functools import partial
from pathlib import Path
from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
import statistical_models.pdfs as pdf
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from loguru import logger
from scipy.integrate import dblquad, quad
from sweights import SWeight

image_dir = Path("/Users/lucascurtin/Desktop/latex_bank/S1/images")

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
g_s = partial(
    pdf.g_s_base,
    mu=mu,
    sigma=sigma,
    beta=beta,
    m=m,
    x_min=BOUNDS_X[0],
    x_max=BOUNDS_X[1],
)
h_s = partial(pdf.h_s_base, lam=lam, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])
g_b = partial(pdf.g_b_base, x_min=BOUNDS_X[0], x_max=BOUNDS_X[1])
h_b = partial(pdf.h_b_base, mu=mu_b, sigma=sigma_b, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])


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
fig, axs = plt.subplots(
    2,
    2,
    figsize=(10, 6),
)
axs = axs.ravel()

# Plot g_s (Truncated Crystal Ball)
x_data = np.linspace(BOUNDS_X[0], BOUNDS_X[1], 200)
y_data = np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 200)


axs[0].plot(x_data, g_s(x_data))
axs[0].set_title("Truncated Crystal Ball PDF (g_s)")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Density")

# Plot h_s (Truncated Exponential)
axs[1].plot(y_data, h_s(y_data))
axs[1].set_title("Truncated Exponential PDF (h_s)")
axs[1].set_xlabel("Y")
axs[1].set_ylabel("Density")

# Plot g_b (Uniform)
axs[2].plot(x_data, g_b(x_data))
axs[2].set_title("Uniform PDF (g_b)")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Density")

# Plot h_b (Truncated Normal)
axs[3].plot(y_data, h_b(y_data))
axs[3].set_title("Truncated Normal PDF (h_b)")
axs[3].set_xlabel("Y")
axs[3].set_ylabel("Density")

plt.tight_layout()
plt.savefig(image_dir / "indivdual_pdfs.png", dpi=300)
plt.show()

# %%
# ? 1D Projections
fig, axs = plt.subplots(
    1,
    2,
    figsize=(10, 6),
)
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
plt.savefig(image_dir / "combined_pdfs.png", dpi=300)
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
fig, ax = plt.subplots(figsize=(10, 6))
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
plt.savefig(image_dir / "joint_density.png", dpi=300)
plt.show()

# %%
# ? Generating Estimates


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

    # Update partials if parameters change during fitting
    g_s = partial(
        pdf.g_s_base,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        x_min=BOUNDS_X[0],
        x_max=BOUNDS_X[1],
    )
    h_s = partial(pdf.h_s_base, lam=lam, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])
    g_b = partial(pdf.g_b_base, x_min=BOUNDS_X[0], x_max=BOUNDS_X[1])
    h_b = partial(pdf.h_b_base, mu=mu_b, sigma=sigma_b, y_min=BOUNDS_Y[0], y_max=BOUNDS_Y[1])

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
    return minuit


rng = np.random.default_rng(seed=451)
n_events = 100_000


sampled_x, sampled_y = total_pdf_sampler(
    n_events=n_events,
    f=f,
    rng=rng,
)

extended_minuit = perform_fit(sampled_x, sampled_y)


# %%
# ? Plotting results
actual_values = {
    "mu": mu,
    "sigma": sigma,
    "beta": beta,
    "m": m,
    "f": f,
    "lam": lam,
    "mu_b": mu_b,
    "sigma_b": sigma_b,
}

# Extract parameter names, estimated values, and uncertainties from extended_minuit
param_names = extended_minuit.parameters
estimated_values = [extended_minuit.values[name] for name in param_names]  # noqa: PD011
estimated_errors = [extended_minuit.errors[name] for name in param_names]

# Match actual values order to the parameter names
actual_values_ordered = [actual_values[name] for name in param_names]

# X positions for the bars
x_positions = np.arange(len(param_names))

# Bar width
bar_width = 0.4

plt.figure(figsize=(10, 6))

# Plot actual values
plt.bar(
    x_positions - bar_width / 2,
    actual_values_ordered,
    width=bar_width,
    color="red",
    alpha=0.7,
    label="Actual Values",
)

# Plot estimated values with error bars
plt.bar(
    x_positions + bar_width / 2,
    estimated_values,
    yerr=estimated_errors,
    width=bar_width,
    color="blue",
    alpha=0.7,
    capsize=5,
    label="Estimated Values",
)

# Formatting
plt.xticks(x_positions, param_names, rotation=45, ha="right", fontsize=12)
plt.xlabel("Parameter Names", fontsize=14)
plt.ylabel("Parameter Values", fontsize=14)
plt.title("Comparison of Actual and Estimated Parameter Values", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig(image_dir / "param_estimates.png", dpi=300)
plt.show()

# %%
# ? Generation Timings

n_runs = 100
# Number of batches for uncertainty estimation
n_batches = 10
n_runs_per_batch = n_runs // n_batches

# Timing results storage
normal_times = []
sample_generation_times = []
fit_times = []

# Benchmark each operation
for _ in range(n_batches):
    normal_times.append(
        timeit(lambda: rng.normal(size=n_events), number=n_runs_per_batch) / n_runs_per_batch,
    )
    sample_generation_times.append(
        timeit(
            lambda: total_pdf_sampler(
                n_events=n_events,
                f=f,
                rng=rng,
            ),
            number=n_runs_per_batch,
        )
        / n_runs_per_batch,
    )
    fit_times.append(
        timeit(
            lambda: perform_fit(sampled_x=sampled_x, sampled_y=sampled_y),
            number=n_runs_per_batch,
        )
        / n_runs_per_batch,
    )

# Convert results to numpy arrays
normal_times = np.array(normal_times)
sample_generation_times = np.array(sample_generation_times)
fit_times = np.array(fit_times)

# Compute mean and standard deviation
normal_time_mean = np.mean(normal_times)
normal_time_std = np.std(normal_times, ddof=1)
sample_generation_time_mean = np.mean(sample_generation_times)
sample_generation_time_std = np.std(sample_generation_times, ddof=1)
fit_time_mean = np.mean(fit_times)
fit_time_std = np.std(fit_times, ddof=1)

# Compute relative times and their uncertainties
relative_sample_generation = sample_generation_time_mean / normal_time_mean
relative_sample_generation_uncertainty = relative_sample_generation * np.sqrt(
    (sample_generation_time_std / sample_generation_time_mean) ** 2
    + (normal_time_std / normal_time_mean) ** 2,
)

relative_fit = fit_time_mean / normal_time_mean
relative_fit_uncertainty = relative_fit * np.sqrt(
    (fit_time_std / fit_time_mean) ** 2 + (normal_time_std / normal_time_mean) ** 2,
)

# Log benchmark results with uncertainties
logger.info(f"Benchmark Results (averaged over {n_batches * n_runs_per_batch} runs):")
logger.info(f"(i) np.random.normal: {normal_time_mean:.6f} ± {normal_time_std:.6f} s")
logger.info(
    f"(ii) Sample generation: {sample_generation_time_mean:.6f} ± {sample_generation_time_std:.6f} s "
    f"(relative: {relative_sample_generation:.2f} ± {relative_sample_generation_uncertainty:.2f})",
)
logger.info(
    f"(iii) Fit execution: {fit_time_mean:.6f} ± {fit_time_std:.6f} s "
    f"(relative: {relative_fit:.2f} ± {relative_fit_uncertainty:.2f})",
)
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
    rng = np.random.default_rng(seed=451)  # Use the RNG with a fixed seed for reproducibility

    for _ in range(n_bootstrap):
        # Introduce Poisson variation to the sample size
        actual_n_events = rng.poisson(n_events)

        # Generate the sample using the varied sample size
        sampled_x, sampled_y = total_pdf_sampler(
            n_events=actual_n_events,  # Use Poisson-varied sample size
            f=f,
            rng=rng,
        )

        # Perform the MLE fit for lambda
        minuit = Minuit(
            lambda lam: nll(lam=lam, y_data=sampled_y),  # noqa: B023
            lam=lam,
        )

        minuit.migrad()
        if minuit.valid:
            lam_results[n_events]["values"].append(minuit.values["lam"])  # noqa: PD011
            lam_results[n_events]["errors"].append(minuit.errors["lam"])


# Calculate bias and uncertainty from results dictionary
bias = [np.mean(lam_results[size]["values"]) - lam for size in sample_sizes]
uncertainty = [np.mean(lam_results[size]["errors"]) for size in sample_sizes]

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# Bias plot
axs[0].plot(sample_sizes, bias, marker="o")
axs[0].set_title("Bias in λ vs Sample Size")
axs[0].set_xlabel("Sample Size")
axs[0].set_ylabel("Bias")
axs[0].grid(visible=True)

# Uncertainty plot
axs[1].plot(sample_sizes, uncertainty, marker="o")
axs[1].set_title("Uncertainty in λ vs Sample Size")
axs[1].set_xlabel("Sample Size")
axs[1].set_ylabel("Uncertainty")
axs[1].grid(visible=True)

plt.tight_layout()
plt.savefig(image_dir / "parametric_bootstrapping.png", dpi=300)
plt.show()


# %%
# ? Extended maximum likelihood fit in X and calculation of sWeights for Y


def nll_x(mu: float, sigma: float, beta: float, m: float, f: float) -> float:
    """Negative log-likelihood for fitting in X."""
    signal_pdf = pdf.g_s_base(sampled_x, mu, sigma, beta, m, BOUNDS_X[0], BOUNDS_X[1])
    background_pdf = pdf.g_b_base(sampled_x, BOUNDS_X[0], BOUNDS_X[1])
    total_pdf_x = f * signal_pdf + (1 - f) * background_pdf
    return -np.sum(np.log(total_pdf_x))


weighted_lam_results = {size: {"values": [], "errors": []} for size in sample_sizes}

for n_events in sample_sizes:
    sampled_x, sampled_y = total_pdf_sampler(
        n_events=n_events,  # Use Poisson-varied sample size
        f=f,
        rng=rng,
    )

    # Set up Minuit for the extended fit
    minuit_x = Minuit(
        nll_x,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        f=f,
    )
    minuit_x.limits["beta"] = (0, None)  # Ensure beta > 0
    minuit_x.limits["m"] = (1, None)  # Ensure m > 1
    minuit_x.limits["f"] = (0, 1)  # Signal fraction must be between 0 and 1
    minuit_x.migrad()
    minuit_x.hesse()

    # Extract fit parameters
    fit_params_x = minuit_x.values  # noqa: PD011
    fit_signal_fraction = fit_params_x["f"]
    fit_signal_count = fit_signal_fraction * len(sampled_x)
    fit_background_count = (1 - fit_signal_fraction) * len(sampled_x)

    # Create the sweighter using spdf and bpdf
    sweighter = SWeight(
        data=sampled_x,  # Dataset (reshaped if necessary)
        pdfs=[
            lambda x,
            mu=fit_params_x["mu"],
            sigma=fit_params_x["sigma"],
            beta=fit_params_x["beta"],
            m=fit_params_x["m"]: pdf.g_s_base(
                np.atleast_1d(x),
                mu,
                sigma,
                beta,
                m,
                BOUNDS_X[0],
                BOUNDS_X[1],
            ),
            lambda x: pdf.g_b_base(np.atleast_1d(x), BOUNDS_X[0], BOUNDS_X[1]),
        ],  # Signal and background PDFs
        yields=[fit_signal_count, fit_background_count],  # Signal and background yields
        discvarranges=[(BOUNDS_X[0], BOUNDS_X[1])],
        method="summation",
        compnames=["sig", "bkg"],
        verbose=True,
        checks=True,
    )

    # Retrieve signal weights
    signal_weights = sweighter.get_weight(0, sampled_x)

    # Perform a weighted fit in Y to extract lambda
    def nll_y_weighted(lam: float) -> float:
        """Weighted negative log-likelihood for fitting in Y."""
        pdf_y = pdf.h_s_base(sampled_y, lam, BOUNDS_Y[0], BOUNDS_Y[1])  # noqa: B023
        weighted_log_likelihood = signal_weights * np.log(pdf_y)  # noqa: B023
        return -np.sum(weighted_log_likelihood)

    # Set up Minuit for the weighted fit in Y
    minuit_y_weighted = Minuit(nll_y_weighted, lam=lam)
    minuit_y_weighted.limits["lam"] = (0, None)  # Decay constant lambda must be > 0
    minuit_y_weighted.migrad()
    minuit_y_weighted.hesse()

    weighted_lam_results[n_events]["values"].append(minuit_y_weighted.values["lam"])  # noqa: PD011
    weighted_lam_results[n_events]["errors"].append(minuit_y_weighted.errors["lam"])


weighted_bias = [np.mean(weighted_lam_results[size]["values"]) - lam for size in sample_sizes]
weighted_uncertainty = [np.mean(weighted_lam_results[size]["errors"]) for size in sample_sizes]


# %%
# Plot lambda results
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# Bias plot
axs[0].plot(sample_sizes, bias, marker="o", label="Unweighted")
axs[0].plot(sample_sizes, weighted_bias, marker="s", label="Weighted")
axs[0].set_title("Bias in λ vs Sample Size")
axs[0].set_xlabel("Sample Size")
axs[0].set_ylabel("Bias")
axs[0].legend()
axs[0].grid(visible=True)

# Uncertainty plot
axs[1].plot(sample_sizes, uncertainty, marker="o", label="Unweighted")
axs[1].plot(sample_sizes, weighted_uncertainty, marker="s", label="Weighted")
axs[1].set_title("Uncertainty in λ vs Sample Size")
axs[1].set_xlabel("Sample Size")
axs[1].set_ylabel("Uncertainty")
axs[1].legend()
axs[1].grid(visible=True)

plt.tight_layout()
plt.savefig(image_dir / "sweights.png", dpi=300)
plt.show()

# %%
