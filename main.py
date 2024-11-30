"""Main script for analysis."""

# %% Imports and setup

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import statistical_models.pdfs as pdf
from loguru import logger
from statistical_models.integrator import integrate_pdf

# %% Part (a) - Numeric Validation of the Normalisation Constant


params = {
    "mu": 3,  # Mean of the signal Crystal Ball PDF
    "sigma": 0.3,  # Standard deviation of the signal Crystal Ball PDF
    "beta": 1,  # Transition point for the signal Crystal Ball PDF
    "m": 1.4,  # Power-law slope for the signal Crystal Ball PDF
    "f": 0.6,  # Signal fraction in the total PDF
    "lambda": 0.3,  # Decay constant for the signal exponential PDF
    "mu_b": 0,  # Mean of the background Gaussian PDF
    "sigma_b": 2.5,  # Standard deviation of the background Gaussian PDF
}

n_inverse = pdf.crystal_ball_inverse_normalisation(
    sigma=params["sigma"],
    beta=params["beta"],
    m=params["m"],
)


# %% Part (b) - Total PDF Normalisation Check


# Define the PDF functions for normalisation checks, including bounds
pdf_functions = {
    "g_s(X)": {
        "pdf": partial(
            pdf.truncated_crystal_ball,
            mu=params["mu"],
            sigma=params["sigma"],
            beta=params["beta"],
            m=params["m"],
            x_lower=pdf.X_LOWER,
            x_upper=pdf.X_UPPER,
        ),
        "x_bounds": (pdf.X_LOWER, pdf.X_UPPER),
        "y_bounds": (None, None),  # None indicates single integration over x
    },
    "h_s(Y)": {
        "pdf": partial(
            pdf.truncated_exponential_decay,
            lam=params["lambda"],
            y_lower=pdf.Y_LOWER,
            y_upper=pdf.Y_UPPER,
        ),
        "x_bounds": (None, None),  # None indicates single integration over y
        "y_bounds": (pdf.Y_LOWER, pdf.Y_UPPER),
    },
    "g_b(X)": {
        "pdf": pdf.uniform_pdf,
        "x_bounds": (pdf.X_LOWER, pdf.X_UPPER),
        "y_bounds": (None, None),
    },
    "h_b(Y)": {
        "pdf": partial(pdf.truncated_gaussian_pdf, mu=params["mu_b"], sigma=params["sigma_b"]),
        "x_bounds": (None, None),
        "y_bounds": (pdf.Y_LOWER, pdf.Y_UPPER),
    },
    "s(X, Y)": {
        "pdf": partial(
            pdf.signal_pdf,
            params=params,
            y_lower=pdf.Y_LOWER,
            y_upper=pdf.Y_UPPER,
            x_lower=pdf.X_LOWER,
            x_upper=pdf.X_UPPER,
        ),
        "x_bounds": (pdf.X_LOWER, pdf.X_UPPER),
        "y_bounds": (pdf.Y_LOWER, pdf.Y_UPPER),
    },
    "b(X, Y)": {
        "pdf": partial(pdf.background_pdf, params=params),
        "x_bounds": (pdf.X_LOWER, pdf.X_UPPER),
        "y_bounds": (pdf.Y_LOWER, pdf.Y_UPPER),
    },
    "f(X, Y)": {
        "pdf": partial(
            pdf.total_pdf,
            params=params,
            y_lower=pdf.Y_LOWER,
            y_upper=pdf.Y_UPPER,
            x_lower=pdf.X_LOWER,
            x_upper=pdf.X_UPPER,
        ),
        "x_bounds": (pdf.X_LOWER, pdf.X_UPPER),
        "y_bounds": (pdf.Y_LOWER, pdf.Y_UPPER),
    },
}

# Iterate over all PDFs and check their normalisation
for name, details in pdf_functions.items():
    pdf_function = details["pdf"]
    x_bounds = details["x_bounds"]
    y_bounds = details["y_bounds"]

    normalisation = integrate_pdf(
        pdf_function,
        x_bounds[0],
        x_bounds[1],
        y_bounds[0],
        y_bounds[1],
    )

    logger.info(f"Normalisation of {name}: {normalisation:.6f}")


# %%

# ? Evaluate the PDFs
x_range = np.linspace(pdf.X_LOWER, pdf.X_UPPER, 500)
y_range = np.linspace(pdf.Y_LOWER, pdf.Y_UPPER, 500)

g_s_x = [pdf_functions["g_s(X)"]["pdf"](x) for x in x_range]
h_s_y = [pdf_functions["h_s(Y)"]["pdf"](y) for y in y_range]

g_b_x = [pdf_functions["g_b(X)"]["pdf"](x) for x in x_range]
h_b_y = [pdf_functions["h_b(Y)"]["pdf"](y) for y in y_range]

# Compute signal and background components for X and Y

total_x = params["f"] * np.array(g_s_x) + (1 - params["f"]) * np.array(g_b_x)
total_y = params["f"] * np.array(h_s_y) + (1 - params["f"]) * np.array(h_b_y)

# %%
# ? Plot the 1D projection in X
plt.figure(figsize=(8, 6))
plt.plot(x_range, total_x, label="Total PDF", linewidth=2)
plt.plot(x_range, params["f"] * np.array(g_s_x), "--", label="Signal PDF", linewidth=1.5)
plt.plot(
    x_range,
    (1 - params["f"]) * np.array(g_b_x),
    "--",
    label="Background PDF",
    linewidth=1.5,
)
plt.title("1D Projection in X")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# %%
# ? Plot the 1D projection in Y
plt.figure(figsize=(8, 6))
plt.plot(y_range, total_y, label="Total PDF", linewidth=2)
plt.plot(y_range, params["f"] * np.array(h_s_y), "--", label="Signal PDF", linewidth=1.5)
plt.plot(
    y_range,
    (1 - params["f"]) * np.array(h_b_y),
    "--",
    label="Background PDF",
    linewidth=1.5,
)
plt.title("1D Projection in Y")
plt.xlabel("Y")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# %%
# ? 2D plot of the joint probability density
x_grid, y_grid = np.meshgrid(x_range, y_range)
joint_pdf = np.array([[pdf_functions["f(X, Y)"]["pdf"](x, y) for x in x_range] for y in y_range])


plt.figure(figsize=(10, 8))
plt.contourf(x_grid, y_grid, joint_pdf, levels=50, cmap="viridis")
plt.colorbar(label="Probability Density")
plt.title("2D Joint Probability Density")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
# %%
