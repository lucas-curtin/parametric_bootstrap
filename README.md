# Statistical Model

This package provides tools to define, normalise, and test probability distributions commonly used in statistical studies, including signal and background models.

## Features

- **Crystal Ball PDF**: Gaussian core with a power-law tail for modelling asymmetric distributions.
- **Exponential Decay PDF**: Ideal for modelling decaying processes.
- **Truncated PDFs**:
  - **Truncated Crystal Ball**: Normalised over a defined range.
  - **Truncated Exponential Decay**: Normalised over a defined range.
  - **Truncated Gaussian**: A normal distribution confined to a specific range.
  - **Uniform PDF**: Flat distribution over a specified range.
- **Integration Utilities**: Tools for numerical integration to validate and check the normalisation of complex multidimensional PDFs.
- **Composite Models**:
  - **Signal PDFs**: Product of a Crystal Ball distribution and an exponential decay.
  - **Background PDFs**: Combination of uniform and Gaussian distributions.
  - **Total PDFs**: Weighted sum of signal and background models.
- **Testing Framework**: Includes normalisation tests to ensure accuracy and correctness of the PDFs.

## Installation

Install the package locally in editable mode:

```bash
pip install -e .
```

## Usage

### 1. Compute and Evaluate PDFs
Use the `pdfs` module to define and evaluate distributions. For example, compute the value of a Crystal Ball PDF at a given point:

```python
from statistical_model.pdfs import crystal_ball

mu = 3
sigma = 0.3
beta = 1
m = 1.4
x = 2.5

pdf_value = crystal_ball(x, mu, sigma, beta, m)
print(f"Crystal Ball PDF value: {pdf_value}")
```

### 2. Validate Normalisation
Use the `integrator` module to check the normalisation of signal, background, and total PDFs:

```python
from statistical_model.integrator import check_normalisation

params = {
    "mu": 3,
    "sigma": 0.3,
    "beta": 1,
    "m": 1.4,
    "f": 0.6,
    "lambda": 0.3,
    "mu_b": 0,
    "sigma_b": 2.5,
}

signal_norm, background_norm, total_norm = check_normalisation(params)

print(f"Signal Normalisation: {signal_norm}")
print(f"Background Normalisation: {background_norm}")
print(f"Total Normalisation: {total_norm}")
```

### 3. Running Tests
The package includes unit tests to verify the accuracy of the PDFs and their normalisation. Run the tests using `pytest`:

```bash
pytest -s tests/
```

### Example Test Script

Here's a sample test to validate the normalisation:

```python
from statistical_model.integrator import check_normalisation

def test_normalisation():
    params = {
        "mu": 3,
        "sigma": 0.3,
        "beta": 1,
        "m": 1.4,
        "f": 0.6,
        "lambda": 0.3,
        "mu_b": 0,
        "sigma_b": 2.5,
    }
    
    signal_norm, background_norm, total_norm = check_normalisation(params)
    
    if not abs(signal_norm - 1.0) < 1e-6:
        raise ValueError(f"Signal PDF is not properly normalised: {signal_norm}")
    
    if not abs(background_norm - 1.0) < 1e-6:
        raise ValueError(f"Background PDF is not properly normalised: {background_norm}")
    
    if not abs(total_norm - 1.0) < 1e-6:
        raise ValueError(f"Total PDF is not properly normalised: {total_norm}")
```

### 4. Extending the Package
The package is designed to be extensible. You can add more custom PDFs or composite models by defining new functions in the `pdfs` module and using the utilities in `integrator` for validation.

---

## Project Structure

```
statistical_model/
├── statistical_model/
│   ├── __init__.py
│   ├── pdfs.py          # Defines PDFs like Crystal Ball, Exponential, etc.
│   ├── integrator.py    # Utilities for integration and normalisation checks.
│   ├── utils.py         # Additional helpers (if needed).
├── tests/
│   ├── __init__.py
│   ├── test_pdfs.py     # Unit tests for PDFs.
│   ├── test_integrator.py # Unit tests for integration and normalisation.
├── pyproject.toml       # Build configuration.
├── README.md            # Documentation.
```

## Dependencies

The package uses the following Python libraries:
- `scipy`: For numerical integration and statistical distributions.
- `numpy`: For efficient numerical computations.

Install dependencies using `pip`:

```bash
pip install scipy numpy
```

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the functionality or extend the features of the package.

---

## License

This package is licensed under the MIT License. See the `LICENSE` file for details.
