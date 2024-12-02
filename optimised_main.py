"""Main script for analysis."""

# %% Imports and setup
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numba_stats.crystalball import pdf as crystal_ball_pdf
from numba_stats.expon import pdf as expon_pdf
from numba_stats.truncnorm import pdf as truncnorm_pdf
from numba_stats.uniform import pdf as uniform_pdf
from scipy.integrate import nquad

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

BOUNDS_X = (0, 5)
BOUNDS_Y = (0, 10)

# %%
# ? Define PDFs


# 1D PDFs for X and Y
def g_s(x: float) -> float:
    """Truncated Crystal Ball PDF."""
    return crystal_ball_pdf(
        x,
        loc=params["mu"],
        scale=params["sigma"],
        beta=params["beta"],
        m=params["m"],
    )


def h_s(y: float) -> float:
    """Truncated exponential decay PDF."""
    return expon_pdf(y, 0, 1 / params["lam"])  # Positional arguments for loc and scale


def g_b(x: float) -> float:
    """Uniform PDF."""
    return uniform_pdf(
        x,
        BOUNDS_X[0],
        BOUNDS_X[1] - BOUNDS_X[0],
    )  # Positional arguments for loc and scale


def h_b(y: float) -> float:
    """Truncated Gaussian PDF."""
    a = (BOUNDS_Y[0] - params["mu_b"]) / params["sigma_b"]
    b = (BOUNDS_Y[1] - params["mu_b"]) / params["sigma_b"]
    return truncnorm_pdf(
        y,
        a,
        b,
        params["mu_b"],
        params["sigma_b"],
    )  # Positional arguments for a, b, loc, scale


# %%
# ? 2D PDFs and Combined Model


def signal(x: float, y: float) -> float:
    """Signal 2D PDF."""
    return g_s(x) * h_s(y)


def background(x: float, y: float) -> float:
    """Background 2D PDF."""
    return g_b(x) * h_b(y)


def total_pdf(f: float) -> callable:
    """Total combined PDF."""

    def total_func(x: float, y: float) -> float:
        return f * signal(x, y) + (1 - f) * background(x, y)

    return total_func


total = total_pdf(f=params["f"])

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


def integrate_pdf(pdf: callable, bounds: list[tuple]) -> float:
    """Integrate a given PDF over its bounds."""
    return nquad(pdf, bounds)[0]


for name, pdf in pdf_dict.items():
    bounds = (
        [BOUNDS_X, BOUNDS_Y]
        if name in ["Signal", "Background", "Total"]
        else [BOUNDS_X if name.startswith("g_") else BOUNDS_Y]
    )
    norm = integrate_pdf(pdf, bounds)
    logger.info(f"Normalisation of {name}: {norm:.6f}")

# %%
# %% Evaluate and Plot the PDFs


# Define ranges for X and Y
x_range = np.linspace(BOUNDS_X[0], BOUNDS_X[1], 500)
y_range = np.linspace(BOUNDS_Y[0], BOUNDS_Y[1], 500)

fixed_y = BOUNDS_Y[1] / 2
fixed_x = BOUNDS_X[1] / 2

# Evaluate individual PDFs
gs_data = [g_s(x) for x in x_range]
hs_data = [h_s(y) for y in y_range]
gb_data = [g_b(x) for x in x_range]
hb_data = [h_b(y) for y in y_range]

# Evaluate combined PDFs
signal_x = [signal(x, fixed_y) for x in x_range]
signal_y = [signal(fixed_x, y) for y in y_range]

background_x = [background(x, fixed_y) for x in x_range]
background_y = [background(fixed_x, y) for y in y_range]

total_x = [total(x, fixed_y) for x in x_range]
total_y = [total(fixed_x, y) for y in y_range]

# %%
# ? Plot Individual PDFs

plt.figure(figsize=(8, 6))
plt.plot(x_range, gs_data, label="$g_s(X)$ (Crystal Ball)")
plt.plot(x_range, gb_data, label="$g_b(X)$ (Uniform)")
plt.title("Individual PDFs in X")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y_range, hs_data, label="$h_s(Y)$ (Exponential Decay)")
plt.plot(y_range, hb_data, label="$h_b(Y)$ (Truncated Normal)")
plt.title("Individual PDFs in Y")
plt.xlabel("Y")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# %%
# ? Plot Combined PDFs

plt.figure(figsize=(8, 6))
plt.plot(x_range, signal_x, label="Signal in X")
plt.plot(x_range, background_x, label="Background in X")
plt.plot(x_range, total_x, label="Total in X")
plt.title(f"1D Projection in X for Y={fixed_y}")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y_range, signal_y, label="Signal in Y")
plt.plot(y_range, background_y, label="Background in Y")
plt.plot(y_range, total_y, label="Total in Y")
plt.title(f"1D Projection in Y for X={fixed_x}")
plt.xlabel("Y")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()

# %%
# ? 2D Joint Probability Density

x_grid, y_grid = np.meshgrid(x_range, y_range)
joint_pdf = np.array([[total(x, y) for x in x_range] for y in y_range])

plt.figure(figsize=(10, 8))
contour = plt.contourf(x_grid, y_grid, joint_pdf, levels=50, cmap="viridis")
plt.colorbar(contour, label="Probability Density")
plt.title("2D Joint Probability Density")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(visible=False)
plt.show()
# %%
