import math

def bandwidth_from_box_height(
    box_height_norm: float,
    sample_rate: float,
    *,
    range_fraction: float = 1.0
) -> float:
    """
    Estimate LoRa signal bandwidth (Hz) from a normalized bounding-box height.

    Parameters
    ----------
    box_height_norm : float
        Height of detected bounding box as fraction of full spectrogram height (0..1).
    sample_rate : float
        SDR sampling rate in Hz (e.g., 32e6).
    range_fraction : float
        Divisor applied to sample_rate to obtain the vertical full range mapped
        by the spectrogram. Example: range_fraction=1 -> full range = sample_rate;
        range_fraction=2 -> full range = sample_rate/2 (Nyquist). Use other
        values for larger/smaller mapping.

    Returns
    -------
    float
        Estimated bandwidth in Hz (continuous, NOT quantized to FFT bins).
    """
    if box_height_norm is None:
        return 0.0
    # clamp to reasonable range
    h = float(box_height_norm)
    if h < 0:
        h = 0.0
    # avoid division by zero for range_fraction
    rf = float(range_fraction) if float(range_fraction) != 0.0 else 1.0
    full_range_hz = sample_rate / rf
    # continuous estimate (no rounding/quantization)
    bw_hz = h * full_range_hz
    return bw_hz


def sf_from_proportional_scaling(
    class_sf: int,
    class_bw_hz: float,
    measured_bw_hz: float
) -> int:
    """
    Compute predicted SF from proportional box scaling (BW and T scale together).

    Formula:
        SF_pred = SF_class + round(2 * log2(R))
    where
        R = measured_bw_hz / class_bw_hz

    Args:
        class_sf: nominal SF from class label.
        class_bw_hz: nominal bandwidth (Hz) of the class (e.g. 125e3).
        measured_bw_hz: measured bandwidth (Hz) from the detected box.

    Returns:
        Integer predicted SF (nearest discrete SF step).
    """
    if class_bw_hz <= 0 or measured_bw_hz <= 0:
        raise ValueError("Bandwidths must be positive numbers.")

    r = measured_bw_hz / class_bw_hz
    delta_sf = round(2.0 * math.log(r, 2))
    return class_sf + delta_sf
