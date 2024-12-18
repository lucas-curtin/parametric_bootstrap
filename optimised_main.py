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
tp = {
    "mu": 3,
    "sigma": 0.3,
    "beta": 1,
    "m": 1.4,
    "f": 0.6,
    "lam": 0.3,
    "mu_b": 0,
    "sigma_b": 2.5,
    "bounds_x": [0, 5],
    "bounds_y": [0, 10],
}


# Define partial functions for each PDF using values from tp
g_s = partial(
    pdf.g_s_base,
    mu=tp["mu"],
    sigma=tp["sigma"],
    beta=tp["beta"],
    m=tp["m"],
    x_min=tp["bounds_x"][0],
    x_max=tp["bounds_x"][1],
)
h_s = partial(
    pdf.h_s_base,
    lam=tp["lam"],
    y_min=tp["bounds_y"][0],
    y_max=tp["bounds_y"][1],
)
g_b = partial(pdf.g_b_base, x_min=tp["bounds_x"][0], x_max=tp["bounds_x"][1])
h_b = partial(
    pdf.h_b_base,
    mu=tp["mu_b"],
    sigma=tp["sigma_b"],
    y_min=tp["bounds_y"][0],
    y_max=tp["bounds_y"][1],
)


# %%
# 2D PDFs
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
g_s_int, g_s_err = integrate_function(g_s, tp["bounds_x"])
h_s_int, h_s_err = integrate_function(h_s, tp["bounds_y"])

# Integrating signal_pdf and background_pdf (2D PDFs)
signal_int, signal_err = integrate_function(signal_pdf, tp["bounds_x"], tp["bounds_x"])
bkg_int, bkg_err = integrate_function(background_pdf, tp["bounds_x"], tp["bounds_x"])

# Integrating total_pdf (2D PDF)
total_int, total_err = integrate_function(
    lambda x, y: total_pdf(x, y, tp["f"]),
    tp["bounds_x"],
    tp["bounds_x"],
)

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
x_data = np.linspace(tp["bounds_x"][0], tp["bounds_x"][1], 200)
y_data = np.linspace(tp["bounds_x"][0], tp["bounds_x"][1], 200)


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
x_data = np.linspace(tp["bounds_x"][0], tp["bounds_x"][1], 200)
y_data = np.linspace(tp["bounds_x"][0], tp["bounds_x"][1], 200)

# Calculate PDFs for X
signal_x = g_s(x_data)
background_x = g_b(x_data)
total_x = tp["f"] * signal_x + (1 - tp["f"]) * background_x

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
total_y = tp["f"] * signal_y + (1 - tp["f"]) * background_y

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
    np.linspace(tp["bounds_x"][0], tp["bounds_x"][1], 100),
    np.linspace(tp["bounds_x"][0], tp["bounds_x"][1], 100),
)

# Flatten the grid arrays for compatibility with the PDF functions
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()

# Compute the total PDF on the flattened grid
joint_pdf_flat = total_pdf(x_flat, y_flat, tp["f"])

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


def total_pdf_sampler(  # noqa: PLR0913
    n_events: int,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    lam: float,
    mu_b: float,
    sigma_b: float,
    f: float,
    rng: np.random.Generator,
    bounds_x: list = tp["bounds_x"],
    bounds_y: list = tp["bounds_y"],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate events from the total PDF using vectorised rejection sampling."""
    g_s = partial(
        pdf.g_s_base,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        x_min=bounds_x[0],
        x_max=bounds_x[1],
    )
    h_s = partial(pdf.h_s_base, lam=lam, y_min=bounds_y[0], y_max=bounds_y[1])
    g_b = partial(pdf.g_b_base, x_min=bounds_x[0], x_max=bounds_x[1])
    h_b = partial(pdf.h_b_base, mu=mu_b, sigma=sigma_b, y_min=bounds_y[0], y_max=bounds_y[1])

    accepted_x = []
    accepted_y = []

    while len(accepted_x) < n_events:
        # Generate random samples
        x_rand = rng.uniform(
            low=bounds_x[0],
            high=bounds_x[1],
            size=10 * (n_events - len(accepted_x)),
        )
        y_rand = rng.uniform(
            low=bounds_y[0],
            high=bounds_y[1],
            size=10 * (n_events - len(accepted_y)),
        )
        p_rand = rng.uniform(low=0, high=1, size=10 * (n_events - len(accepted_x)))

        g_s_val = g_s(x_rand)
        h_s_val = h_s(y_rand)
        signal = g_s_val * h_s_val

        g_b_val = g_b(x_rand)
        h_b_val = h_b(y_rand)
        background = g_b_val * h_b_val

        total_pdf_values = f * signal + (1 - f) * background
        max_pdf_value = np.max(total_pdf_values)

        acceptance_mask = p_rand * max_pdf_value <= total_pdf_values

        accepted_x.extend(x_rand[acceptance_mask])
        accepted_y.extend(y_rand[acceptance_mask])

    return np.array(accepted_x[:n_events]), np.array(accepted_y[:n_events])


def total_density(  # noqa: PLR0913
    xy: tuple[np.ndarray, np.ndarray],
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    f: float,
    lam: float,
    mu_b: float,
    sigma_b: float,
    n_events: float,  # Total number of events (expected number of events)
) -> tuple[float, np.ndarray]:
    """Wrapper for the total PDF with N_total, returning both the integral and density."""
    x, y = xy

    # Precompute the individual PDFs
    g_s_vals = pdf.g_s_base(
        x,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        x_min=tp["bounds_x"][0],
        x_max=tp["bounds_x"][1],
    )
    h_s_vals = pdf.h_s_base(y, lam=lam, y_min=tp["bounds_y"][0], y_max=tp["bounds_y"][1])
    g_b_vals = pdf.g_b_base(x, x_min=tp["bounds_x"][0], x_max=tp["bounds_x"][1])
    h_b_vals = pdf.h_b_base(
        y,
        mu=mu_b,
        sigma=sigma_b,
        y_min=tp["bounds_y"][0],
        y_max=tp["bounds_y"][1],
    )

    # Combine PDFs
    signal_pdf = g_s_vals * h_s_vals
    background_pdf = g_b_vals * h_b_vals

    # Total PDF
    pdf_values = f * signal_pdf + (1 - f) * background_pdf

    # Return integral (n_events) and scaled density
    return n_events, n_events * pdf_values


def perform_fit(  # noqa: PLR0913
    sampled_x: np.array,
    sampled_y: np.array,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    f: float,
    lam: float,
    mu_b: float,
    sigma_b: float,
    n_events: int,
) -> Minuit:
    """Complete minuit fit using extended unbinned likelihood."""
    # Define the NLL using the provided total_density function
    nll = ExtendedUnbinnedNLL((sampled_x, sampled_y), total_density)

    # Initialize Minuit with parameters perturbed by noise
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
        n_events=n_events,
    )

    # Set parameter limits
    minuit.limits["beta"] = (0, None)  # Beta must be positive
    minuit.limits["m"] = (1, None)  # Mass m must be >= 1

    # Perform the minimization and error analysis
    minuit.migrad()
    minuit.hesse()

    return minuit


rng = np.random.default_rng(seed=451)


tp["n_events"] = 100_000

sampled_x, sampled_y = total_pdf_sampler(
    n_events=tp["n_events"],
    f=tp["f"],
    rng=rng,
    mu=tp["mu"],
    sigma=tp["sigma"],
    beta=tp["beta"],
    m=tp["m"],
    lam=tp["lam"],
    mu_b=tp["mu_b"],
    sigma_b=tp["sigma_b"],
    bounds_x=tp["bounds_x"],
    bounds_y=tp["bounds_y"],
)

extended_minuit = perform_fit(
    sampled_x=sampled_x,
    sampled_y=sampled_y,
    mu=tp["mu"],
    sigma=tp["sigma"],
    beta=tp["beta"],
    m=tp["m"],
    f=tp["f"],
    lam=tp["lam"],
    mu_b=tp["mu_b"],
    sigma_b=tp["sigma_b"],
    n_events=tp["n_events"],
)


# %%
# ? Plotting results


param_names = extended_minuit.parameters

actual_values_scaled = [
    tp[name] / tp["n_events"] if name == "n_events" else tp[name] for name in param_names
]

estimated_values_scaled = [
    extended_minuit.values[name] / tp["n_events"]  # noqa: PD011
    if name == "n_events"
    else extended_minuit.values[name]  # noqa: PD011
    for name in param_names
]

estimated_errors_scaled = [
    extended_minuit.errors[name] / tp["n_events"]
    if name == "n_events"
    else extended_minuit.errors[name]
    for name in param_names
]
actual_values_ordered = [tp[name] for name in param_names]

x_positions = np.arange(len(param_names))
bar_width = 0.4

plt.figure(figsize=(10, 6))

plt.bar(
    x_positions - bar_width / 2,
    actual_values_scaled,
    width=bar_width,
    color="red",
    alpha=0.7,
    label="Actual Values",
)

plt.bar(
    x_positions + bar_width / 2,
    estimated_values_scaled,
    yerr=estimated_errors_scaled,
    width=bar_width,
    color="blue",
    alpha=0.7,
    capsize=5,
    label="Estimated Values",
)

param_names_latex = [
    r"$\mu$",
    r"$\sigma$",
    r"$\beta$",
    r"$m$",
    r"$f$",
    r"$\lambda$",
    r"$\mu_b$",
    r"$\sigma_b$",
    r"$\frac{n_{\text{total}}}{n_{\text{events}}}$",
]

plt.xticks(
    x_positions,
    param_names_latex,  # Use the LaTeX formatted parameter names
    rotation=45,
    ha="right",
    fontsize=12,
)
plt.xlabel("Parameter Names", fontsize=14)
plt.ylabel("Parameter Values (Scaled)", fontsize=14)
plt.title("Comparison of Actual and Estimated Parameter Values (Scaled)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig(image_dir / "param_estimates_scaled.png", dpi=300)
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
                f=tp["f"],
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
    f"(ii) Sample generation: {sample_generation_time_mean:.6f} ± {sample_generation_time_std:.6f}s"
    f"(relative: {relative_sample_generation:.2f} ± {relative_sample_generation_uncertainty:.2f})",
)
logger.info(
    f"(iii) Fit execution: {fit_time_mean:.6f} ± {fit_time_std:.6f} s "
    f"(relative: {relative_fit:.2f} ± {relative_fit_uncertainty:.2f})",
)


# %%
# ? Parametric Bootstrapping
rng = np.random.default_rng(seed=451)

sample_sizes = [500, 1000, 2500, 5000, 10000]
n_toys = 250

results = {}


def total_pdf_x(  # noqa: PLR0913
    x: np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    m: float,
    f: float,
    n_events: float,  # Total number of events (expected number of events)
) -> tuple[float, np.ndarray]:
    """Wrapper for the total PDF with N_total, returning both the integral and density."""
    # Precompute the individual PDFs
    signal_pdf = pdf.g_s_base(
        x,
        mu=mu,
        sigma=sigma,
        beta=beta,
        m=m,
        x_min=tp["bounds_x"][0],
        x_max=tp["bounds_x"][1],
    )
    background_pdf = pdf.g_b_base(x, x_min=tp["bounds_x"][0], x_max=tp["bounds_x"][1])

    # Total PDF
    pdf_values = f * signal_pdf + (1 - f) * background_pdf

    # Return integral (n_events) and scaled density
    return n_events, n_events * pdf_values


for sample_size in sample_sizes:
    n_events = rng.poisson(sample_size)

    # Fit the model to the initial sample
    sample_x, sample_y = total_pdf_sampler(
        n_events=n_events,
        f=tp["f"],
        rng=rng,
        mu=tp["mu"],
        sigma=tp["sigma"],
        beta=tp["beta"],
        m=tp["m"],
        lam=tp["lam"],
        mu_b=tp["mu_b"],
        sigma_b=tp["sigma_b"],
    )

    nll = ExtendedUnbinnedNLL((sample_x, sample_y), total_density)
    mi = Minuit(
        nll,
        mu=tp["mu"],
        sigma=tp["sigma"],
        beta=tp["beta"],
        m=tp["m"],
        f=tp["f"],
        lam=tp["lam"],
        mu_b=tp["mu_b"],
        sigma_b=tp["sigma_b"],
        n_events=sample_size,
    )

    mi.limits["beta"] = (0, None)
    mi.limits["m"] = (1, None)

    mi.migrad()
    mi.hesse()

    fit_params = {key: mi.values[key] for key in mi.parameters}  # noqa: PD011

    toys = [
        total_pdf_sampler(
            n_events=n_poisson,
            f=fit_params["f"],
            rng=rng,
            mu=fit_params["mu"],
            sigma=fit_params["sigma"],
            beta=fit_params["beta"],
            m=fit_params["m"],
            lam=fit_params["lam"],
            mu_b=fit_params["mu_b"],
            sigma_b=fit_params["sigma_b"],
        )
        for n_poisson in rng.poisson(int(fit_params["n_events"]), n_toys)
    ]

    results[sample_size] = {
        "values": [],
        "errors": [],
        "lambda_weighted": None,
        "lambda_uncertainty": None,
    }

    # Fit each toy sample, with a minimiser the same as the one to make the original data.
    for toy in toys:
        nll_t = ExtendedUnbinnedNLL((toy[0], toy[1]), total_density)
        mi_t = Minuit(
            nll_t,
            mu=tp["mu"],
            sigma=tp["sigma"],
            beta=tp["beta"],
            m=tp["m"],
            f=tp["f"],
            lam=tp["lam"],
            mu_b=tp["mu_b"],
            sigma_b=tp["sigma_b"],
            n_events=n_events,
        )

        mi_t.limits["beta"] = (0.01, None)  # Beta must be positive
        mi_t.limits["m"] = (1.01, None)
        mi_t.limits["sigma"] = (0.01, None)
        mi_t.limits["sigma_b"] = (0.01, None)

        mi_t.migrad()
        mi_t.hesse()

        # Store values and errors in the results dictionary
        results[sample_size]["values"].append(mi_t.values["lam"])  # noqa: PD011
        results[sample_size]["errors"].append(mi_t.errors["lam"])

    # Perform the lambda calculation using s-weights
    minuit_x = Minuit(
        ExtendedUnbinnedNLL(
            sample_x,
            total_pdf_x,
        ),
        mu=tp["mu"],
        sigma=tp["sigma"],
        beta=tp["beta"],
        m=tp["m"],
        f=tp["f"],
        n_events=len(sample_x),
    )

    minuit_x.limits["beta"] = (0, None)  # Ensure beta > 0
    minuit_x.limits["m"] = (1, None)  # Ensure m > 1
    minuit_x.limits["f"] = (0, 1)  # Signal fraction must be between 0 and 1
    minuit_x.migrad()
    minuit_x.hesse()

    fit_params_x = minuit_x.values  # noqa: PD011
    fit_signal_fraction = fit_params_x["f"]
    fit_signal_count = fit_signal_fraction * len(sample_x)
    fit_background_count = (1 - fit_signal_fraction) * len(sample_x)

    # Define signal and background PDFs based on fit parameters
    signal_pdf = lambda x: pdf.g_s_base(  # noqa: E731
        np.atleast_1d(x),
        fit_params_x["mu"],  # noqa: B023
        fit_params_x["sigma"],  # noqa: B023
        fit_params_x["beta"],  # noqa: B023
        fit_params_x["m"],  # noqa: B023
        tp["bounds_x"][0],
        tp["bounds_x"][1],
    )
    background_pdf = lambda x: pdf.g_b_base(  # noqa: E731
        np.atleast_1d(x),
        tp["bounds_x"][0],
        tp["bounds_x"][1],
    )

    # Create the sWeight object
    sweighter = SWeight(
        data=sample_x,  # Dataset
        pdfs=[signal_pdf, background_pdf],  # Signal and background PDFs
        yields=[fit_signal_count, fit_background_count],  # Signal and background yields
        discvarranges=[(tp["bounds_x"][0], tp["bounds_x"][1])],
        method="summation",
        compnames=["sig", "bkg"],
        verbose=True,
        checks=True,
    )

    signal_weights = sweighter.get_weight(0, sample_x)

    def nll_y_weighted(lam: float) -> float:
        """Weighted negative log likelihood."""
        pdf_y = pdf.h_s_base(sample_y, lam, tp["bounds_y"][0], tp["bounds_y"][1])  # noqa: B023
        weighted_log_likelihood = signal_weights * np.log(pdf_y)  # noqa: B023
        return -np.sum(weighted_log_likelihood)

    minuit_y_weighted = Minuit(nll_y_weighted, lam=1.0)
    minuit_y_weighted.limits["lam"] = (0, None)
    minuit_y_weighted.migrad()
    minuit_y_weighted.hesse()

    results[sample_size]["lambda_weighted"] = minuit_y_weighted.values["lam"]  # noqa: PD011
    results[sample_size]["lambda_uncertainty"] = minuit_y_weighted.errors["lam"]


# %%
# ? Plotting
true_lambda = mi.values["lam"]  # True lambda value from the fit  # noqa: PD011
sample_sizes = list(results.keys())
n_rows = len(sample_sizes)

# Create subplots
fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, sample_size in enumerate(sample_sizes):
    values = np.array(results[sample_size]["values"])
    errors = np.array(results[sample_size]["errors"])
    lambda_weighted = results[sample_size]["lambda_weighted"]
    lambda_uncertainty = results[sample_size]["lambda_uncertainty"]

    # Filter out NaN uncertainties and their corresponding values
    valid_mask = ~np.isnan(errors)
    values = values[valid_mask]
    errors = errors[valid_mask]

    # Compute derived quantities
    biases = values - true_lambda
    uncertainties = errors

    # Compute statistics and their uncertainties
    def compute_stats_with_uncertainty(xvals: np.array) -> tuple:
        """Calc uncertainties."""
        mean = np.mean(xvals)
        std = np.std(xvals, ddof=1)
        mean_error = std / np.sqrt(len(xvals))
        std_error = std / np.sqrt(2 * len(xvals) - 1)
        return mean, std, mean_error, std_error

    mean_values, std_values, me_values, se_values = compute_stats_with_uncertainty(values)
    mean_bias, std_bias, me_bias, se_bias = compute_stats_with_uncertainty(biases)
    mean_uncertainties, std_uncertainties, me_uncertainties, se_uncertainties = (
        compute_stats_with_uncertainty(uncertainties)
    )

    # Format values to significant figures
    def format_with_uncertainty(value: float, uncertainty: float) -> str:
        """Format our values."""
        return f"{value:.2g} ± {uncertainty:.1g}"

    # Column 1: Distribution of fitted values
    hist_values, bin_edges = np.histogram(values, bins="auto", density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    errors_hist = np.sqrt(hist_values / len(values))  # Error bars for histogram

    axs[i, 0].bar(
        bin_centers,
        hist_values,
        width=bin_widths,
        alpha=0.6,
    )
    axs[i, 0].errorbar(bin_centers, hist_values, yerr=errors_hist, fmt="o")
    axs[i, 0].axvline(true_lambda, color="red", linestyle="--", label=r"True $\lambda$")
    axs[i, 0].axvline(
        mean_values,
        color="blue",
        linestyle="--",
        label=rf"$\mu = {format_with_uncertainty(mean_values, me_values)}$",
    )
    axs[i, 0].axvline(
        lambda_weighted,
        color="green",
        linestyle="-.",
        label=rf"Weighted $\lambda = {lambda_weighted:.2g}$",
    )
    axs[i, 0].set_ylabel("Density")
    axs[i, 0].set_title(
        f"Fitted λ (Sample Size {sample_size})\n"
        f"σ = {format_with_uncertainty(std_values, se_values)}",  # noqa: RUF001
    )
    axs[i, 0].legend()
    axs[i, 0].set_xlabel("Fitted λ")

    # Column 2: Distribution of bias
    hist_bias, bin_edges_bias = np.histogram(biases, bins="auto", density=True)
    bin_centers_bias = 0.5 * (bin_edges_bias[:-1] + bin_edges_bias[1:])
    bin_widths_bias = np.diff(bin_edges_bias)
    errors_bias_hist = np.sqrt(hist_bias / len(biases))  # Error bars for histogram

    axs[i, 1].bar(
        bin_centers_bias,
        hist_bias,
        width=bin_widths_bias,
        alpha=0.6,
    )
    axs[i, 1].errorbar(bin_centers_bias, hist_bias, yerr=errors_bias_hist, fmt="o")
    axs[i, 1].axvline(0, color="red", linestyle="--", label="Zero Bias")
    axs[i, 1].axvline(
        mean_bias,
        color="blue",
        linestyle="--",
        label=rf"$\mu = {format_with_uncertainty(mean_bias, me_bias)}$",
    )
    axs[i, 1].axvline(
        lambda_weighted - true_lambda,
        color="green",
        linestyle="-.",
        label=rf"Weighted Bias = {lambda_weighted - true_lambda:.2g}$",
    )
    axs[i, 1].set_ylabel("Density")
    axs[i, 1].set_title(
        f"Bias (Sample Size {sample_size})\nσ = {format_with_uncertainty(std_bias, se_bias)}",  # noqa: RUF001
    )
    axs[i, 1].legend()
    axs[i, 1].set_xlabel("Bias")

    # Column 3: Distribution of uncertainties
    hist_unc, bin_edges_unc = np.histogram(uncertainties, bins="auto", density=True)
    bin_centers_unc = 0.5 * (bin_edges_unc[:-1] + bin_edges_unc[1:])
    bin_widths_unc = np.diff(bin_edges_unc)
    errors_unc_hist = np.sqrt(hist_unc / len(uncertainties))  # Error bars for histogram

    axs[i, 2].bar(
        bin_centers_unc,
        hist_unc,
        width=bin_widths_unc,
        alpha=0.6,
    )
    axs[i, 2].errorbar(bin_centers_unc, hist_unc, yerr=errors_unc_hist, fmt="o")
    axs[i, 2].axvline(mi.errors["lam"], color="red", linestyle="--", label=r"True Uncertainty")
    axs[i, 2].axvline(
        mean_uncertainties,
        color="blue",
        linestyle="--",
        label=rf"$\mu = {format_with_uncertainty(mean_uncertainties, me_uncertainties)}$",
    )
    axs[i, 2].axvline(
        lambda_uncertainty,
        color="green",
        linestyle="-.",
        label=rf"Weighted Uncertainty = {lambda_uncertainty:.2g}$",
    )
    axs[i, 2].set_ylabel("Density")
    axs[i, 2].set_title(
        f"Uncertainty (Sample Size {sample_size})\n"
        f"σ = {format_with_uncertainty(std_uncertainties, se_uncertainties)}",  # noqa: RUF001
    )
    axs[i, 2].legend()
    axs[i, 2].set_xlabel("Uncertainty")

    uncertainties_lims = np.append(uncertainties, [mi.errors["lam"], lambda_uncertainty])
    axs[i, 2].set_xlim([min(uncertainties_lims) * 0.95, max(uncertainties_lims) * 1.05])

plt.tight_layout()
plt.show()


# %%
