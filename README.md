# Statistical Model

This repository provides tools for defining, normalising, and testing probability density functions (PDFs) used in statistical analysis, with a focus on signal and background modelling. The package includes efficient methods for **data generation**, **parameter estimation**, and **bias/uncertainty studies**. It is designed to be computationally efficient, leveraging Python libraries such as `scipy`, `numpy`, and `iminuit`.

## Features

### Core PDFs
The package defines key PDFs for both signal and background distributions:
- **Signal PDFs**:
  - **Truncated Crystal Ball (`g_s`)**: Combines a Gaussian core with a power-law tail, designed for asymmetric distributions.
  - **Truncated Exponential Decay (`h_s`)**: Exponential decay function normalised over a defined range.
- **Background PDFs**:
  - **Uniform Distribution (`g_b`)**: A flat distribution over a specified range.
  - **Truncated Normal Distribution (`h_b`)**: A Gaussian distribution confined to a specific range.

These functions are implemented with normalisation and truncation built-in, ensuring they integrate to unity over their specified domains.

## Tools and Capabilities

### Normalisation Validation
- All PDFs are normalised over their specified bounds.
- Numerical integration ensures correctness.

### Data Generation
- **Vectorised Rejection Sampling**: Used to generate synthetic datasets from the total PDF. This method ensures efficient sampling of signal and background events while maintaining the desired PDF shape.

### Parameter Estimation
- **Extended Maximum Likelihood Fits**:
  - Implemented using the `iminuit` library.
  - Supports simultaneous fitting of all parameters, including shape and scale, with uncertainty propagation.
- **Weighted Fits**:
  - Incorporates sWeights to isolate signal contributions.

### Bias and Uncertainty Studies
- Includes parametric bootstrapping to assess bias and variability of fitted parameters across different sample sizes.

### Visualisations
- Visual representations of PDFs, projections, and parameter comparisons are included.

## Installation
Install the package and its dependencies locally:

```bash
pip install -e .
```

## Usage

### Define PDFs
Define and evaluate distributions using the `pdfs` module. For example, compute the value of a truncated Crystal Ball PDF:

```python
from statistical_models.pdfs import g_s_base

x = 2.5
mu = 3
sigma = 0.3
beta = 1
m = 1.4
x_min = 0
x_max = 5

pdf_value = g_s_base(x, mu, sigma, beta, m, x_min, x_max)
print(f"PDF value: {pdf_value}")
```

### Generate Synthetic Data
Generate datasets using the vectorised rejection sampler:

```python
from optimised_main import total_pdf_sampler
import numpy as np

rng = np.random.default_rng(seed=451)
n_events = 1000
sampled_x, sampled_y = total_pdf_sampler(
    n_events=n_events,
    f=0.6,
    rng=rng,
    mu=3,
    sigma=0.3,
    beta=1,
    m=1.4,
    lam=0.3,
    mu_b=0,
    sigma_b=2.5,
    bounds_x=[0, 5],
    bounds_y=[0, 10],
)
```

### Perform Parameter Estimation
Perform parameter estimation using extended maximum likelihood fits:

```python
from optimised_main import perform_fit

fit_results = perform_fit(
    sampled_x=sampled_x,
    sampled_y=sampled_y,
    mu=3,
    sigma=0.3,
    beta=1,
    m=1.4,
    f=0.6,
    lam=0.3,
    mu_b=0,
    sigma_b=2.5,
    n_events=n_events,
)

print(f"Fitted parameters: {fit_results.values}")
print(f"Uncertainties: {fit_results.errors}")
```

### Running main.py

There are .py scripts with the #%% ipython terminals used. This is largely to help ensure clean code is written with good formatting and linting. These have then been converted into notebooks using the **jupytext** module instead with the **dev** dependencies. This can be utilised as follows:

```bash
jupytext --to notebook your_file.py
```

## AI Declaration
This project was developed with assistance from OpenAI's ChatGPT, which was used for drafting, optimising code structures, and generating this README. All implementation, validation, and final edits were performed by the author. ChatGPT was used extensively for writing up handwritten notes into Latex, as well as helping gather resources to learn more about certain topics such as sWeights. Furthermore, it was used to help teach how to implement numba-jit. Lastly, it was used to help with the documentation of functions and making sleek plots to best show the information from the analysis completed in this project. 
