"""Tests the integration module to ensure PDFs are properly normalised."""

from statistical_model.integrator import check_normalisation


def test_normalisation() -> None:
    """Tests that the signal, background, and total PDFs are properly normalised.

    Verifies that:
    - Signal PDF integrates to 1 within the range: x ∈ [0, 5], y ∈ [0, 10].
    - Background PDF integrates to 1 within the range: x ∈ [0, 5], y ∈ [0, 10].
    - Total PDF integrates to 1 within the range: x ∈ [0, 5], y ∈ [0, 10].

    Raises:
        ValueError: If any of the PDFs fail the normalisation test.
    """
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

    signal_norm, background_norm, total_norm = check_normalisation(params)

    if not abs(signal_norm - 1.0) < 1e-6:
        msg = f"Signal PDF is not properly normalised: {signal_norm}"
        raise ValueError(msg)

    if not abs(background_norm - 1.0) < 1e-6:
        msg = f"Background PDF is not properly normalised: {background_norm}"
        raise ValueError(msg)

    if not abs(total_norm - 1.0) < 1e-6:
        msg = f"Total PDF is not properly normalised: {total_norm}"
        raise ValueError(msg)
