"""Main script for analysis."""

# %% Imports and setup
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from loguru import logger
from statistical_models.pdfs import BOUNDS_X, BOUNDS_Y, PDF, g_b_pdf, g_s_pdf, h_b_pdf, h_s_pdf

# %%
# ? Choose params Parameters

params = {
    "mu": 3,
    "sigma": 0.3,
    "beta": 1,
    "m": 1.4,
    "f": 0.6,
    "lam": 0.3,
    "mu_b": 0,
    "sigma_b": 2.5,
}

# %%
# ? Create 1D PDFs

g_s = g_s_pdf(params=params)
h_s = h_s_pdf(params=params)

g_b = g_b_pdf()
h_b = h_b_pdf(params=params)


signal = PDF(
    func=lambda x, y: g_s.evaluate(x) * h_s.evaluate(y),
    bounds=[BOUNDS_X, BOUNDS_Y],
).normalise()

# Combine g_b and h_b into a 2D background PDF
background = PDF(
    func=lambda x, y: g_b.evaluate(x) * h_b.evaluate(y),
    bounds=[BOUNDS_X, BOUNDS_Y],
).normalise()


def total_pdf(f: float, signal: PDF, background: PDF) -> PDF:
    """Calculates the total PDF by combining signal and background."""
    return (f * signal) + ((1 - f) * background)


total = total_pdf(f=params["f"], signal=signal, background=background)


pdf_dict = {
    "g_s(X)": g_s,
    "h_s(Y)": h_s,
    "g_b(X)": g_b,
    "h_b(Y)": h_b,
    "Signal": signal,
    "Background": background,
    "Total": total,
}

# %%
# ? Normalisation Checks

for name, pdf in pdf_dict.items():
    normalisation = pdf.integrate()
    logger.info(f"Normalisation of {name}: {normalisation:.6f}")
# %%
# ? Evaluate and Plot the PDFs

x_range = np.linspace(BOUNDS_X[0], BOUNDS_X[1], 500)
y_range = np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 500)

fixed_y = BOUNDS_Y[1] / 2
fixed_x = BOUNDS_X[1] / 2

# Dictionary to store data
pdf_data = {
    "Signal": {
        "x": [signal.evaluate(x, fixed_y) for x in x_range],
        "y": [signal.evaluate(fixed_x, y) for y in y_range],
    },
    "Background": {
        "x": [background.evaluate(x, fixed_y) for x in x_range],
        "y": [background.evaluate(fixed_x, y) for y in y_range],
    },
    "Total": {
        "x": [total.evaluate(x, fixed_y) for x in x_range],
        "y": [total.evaluate(fixed_x, y) for y in y_range],
    },
}

# Scaling factors for signal and background
signal_scale = params["f"]
background_scale = 1 - params["f"]

# Scale the data
for key in ["Signal", "Background"]:
    pdf_data[key]["x"] = np.array(pdf_data[key]["x"]) * (
        signal_scale if key == "Signal" else background_scale
    )
    pdf_data[key]["y"] = np.array(pdf_data[key]["y"]) * (
        signal_scale if key == "Signal" else background_scale
    )

# %%
# ? Plot the 1D Projection in X

plt.figure(figsize=(8, 6))
for name, data in pdf_data.items():
    plt.plot(x_range, data["x"], label=f"{name} PDF")
plt.title(f"1D Projection in X for Y={fixed_y}")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(8, 6))
for name, data in pdf_data.items():
    plt.plot(y_range, data["y"], label=f"{name} PDF")
plt.title(f"1D Projection in Y for X={fixed_x}")
plt.xlabel("Y")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# %%
# ? 2D Joint Probability Density

x_grid, y_grid = np.meshgrid(x_range, y_range)
joint_pdf = np.array([[total.evaluate(x, y) for x in x_range] for y in y_range])


plt.figure(figsize=(10, 8))
contour = plt.contourf(x_grid, y_grid, joint_pdf, levels=50, cmap="viridis")
plt.colorbar(contour, label="Probability Density")
plt.title("2D Joint Probability Density")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(visible=False)
plt.show()


# %%
# ? Generate sample


def generate_sample(
    pdf: PDF,
    total_events: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a high-statistics sample from the joint PDF."""
    x_samples = rng.uniform(0, 5, size=total_events)
    y_samples = rng.uniform(0, 10, size=total_events)
    probabilities = np.array([pdf.evaluate(x, y) for x, y in zip(x_samples, y_samples)])
    sampled_indices = rng.choice(
        np.arange(total_events),
        size=total_events,
        p=probabilities / probabilities.sum(),
    )
    return x_samples[sampled_indices], y_samples[sampled_indices]


def negative_log_likelihood(params: dict, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """Compute the extended negative log-likelihood for the total PDF."""
    # Dynamically reconstruct signal and background PDFs
    g_s = g_s_pdf(params=params)
    h_s = h_s_pdf(params=params)

    g_b = g_b_pdf()
    h_b = h_b_pdf(params=params)

    signal = PDF(
        func=lambda x, y: g_s.evaluate(x) * h_s.evaluate(y),
        bounds=[BOUNDS_X, BOUNDS_Y],
    ).normalise()

    background = PDF(
        func=lambda x, y: g_b.evaluate(x) * h_b.evaluate(y),
        bounds=[BOUNDS_X, BOUNDS_Y],
    ).normalise()

    pdf = total_pdf(params["f"], signal, background)

    # Extract the expected number of events
    n_expected = params["n_expected"]

    # Compute the observed number of events
    n_observed = len(x_data)

    # Compute the negative extended log-likelihood
    log_likelihood = (
        -n_expected
        + n_observed * np.log(n_expected)
        - np.sum(np.log([pdf.evaluate(x, y) for x, y in zip(x_data, y_data)]))
    )

    return -log_likelihood


def perform_fit_iminuit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    initial_params: dict,
) -> Minuit:
    """Perform an extended maximum likelihood fit using iminuit."""

    def wrapped_neg_log_likelihood(  # noqa: PLR0913
        mu: float,
        sigma: float,
        beta: float,
        m: float,
        f: float,
        lam: float,
        mu_b: float,
        sigma_b: float,
        n_expected: float,
    ) -> float:
        """A wrapper function for the negative log-likelihood."""
        current_params = {
            "mu": mu,
            "sigma": sigma,
            "beta": beta,
            "m": m,
            "f": f,
            "lam": lam,
            "mu_b": mu_b,
            "sigma_b": sigma_b,
            "n_expected": n_expected,
        }
        return negative_log_likelihood(current_params, x_data, y_data)

    # Add initial value for n_expected
    initial_params = {**initial_params, "n_expected": len(x_data)}

    # Instantiate Minuit using dictionary unpacking for parameters
    m = Minuit(
        wrapped_neg_log_likelihood,
        **initial_params,
    )

    # Specify parameter bounds using lists
    m.limits = [
        (None, None),  # mu: no bounds
        (None, None),  # sigma: no bounds
        (0, None),  # beta: beta > 0
        (1, None),  # m: m > 1
        (0, 1),  # f: mixing fraction in [0, 1]
        (None, None),  # lam: no bounds
        (None, None),  # mu_b: no bounds
        (None, None),  # sigma_b: no bounds
        (0, None),  # n_expected: must be positive
    ]

    m.errordef = Minuit.LIKELIHOOD
    m.migrad()  # Perform the fit
    m.hesse()  # Compute uncertainties
    return m


# %%
# ? iminuit tests

rng = np.random.default_rng(seed=451)

total_events = 100_000

# Generate sample
x_data, y_data = generate_sample(pdf=total, total_events=total_events, rng=rng)

# Perform the fit
best_params = perform_fit_iminuit(x_data=x_data, y_data=y_data, initial_params=params)

# Output the fit results
logger.info(f"Best fit parameters:\n{best_params.params}")

# %%
