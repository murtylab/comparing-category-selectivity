#!/usr/bin/env python3
"""Lightweight utilities for PIL images used in the stimulus pipeline.

This module offers single-image helpers extracted from the original stimulus
creation scripts so that filtering, scrambling, and inversion can be reused in
other notebooks or scripts without copying the large procedural workflow.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageOps


def _to_float_grayscale(image: Image.Image) -> np.ndarray:
    """Return the image as a float32 numpy array in [0, 1]."""
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _to_uint8_image(array: np.ndarray) -> Image.Image:
    """Scale a float array to 8-bit grayscale and wrap as a PIL image."""
    arr_min = float(array.min())
    arr_max = float(array.max())
    if math.isclose(arr_max, arr_min):
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        scaled = ((array - arr_min) / (arr_max - arr_min) * 255.0).clip(0, 255)
        scaled = scaled.astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def DoLogGabor(
    image: np.ndarray,
    FreqPeak: float,
    FreqSigma: float,
    theta: float,
    ThetaSigma: float,
    graphics: bool = False,
) -> np.ndarray:
    """Log-Gabor filtering ported from the original stimulus scripts."""
    del graphics  # retained for API compatibility

    n, m = image.shape
    x, y = np.meshgrid(np.arange(m) - m // 2, np.arange(n) - n // 2)
    polar_grid = x + y * 1j
    center_dist = np.abs(polar_grid)
    center_ang = np.angle(polar_grid)

    ds = np.sin(center_ang) * np.cos(theta) - np.cos(center_ang) * np.sin(theta)
    dc = np.cos(center_ang) * np.cos(theta) + np.sin(center_ang) * np.sin(theta)
    dtheta = np.arctan2(ds, dc)
    ang_spread = np.exp(-0.5 * (dtheta / ThetaSigma) ** 2)

    freq_sd_absolute = 1 / (2**FreqSigma)
    rfo = (FreqPeak / min(m, n)) / 0.5
    with np.errstate(divide="ignore"):
        sf_spread = np.exp(
            -0.5 * (np.log(center_dist / rfo) / np.log(freq_sd_absolute)) ** 2
        )
    sf_spread[n // 2, m // 2] = 0

    filter_ = sf_spread * ang_spread
    filtered_image = np.fft.ifftshift(filter_) * np.fft.fft2(image)
    filtered_image = np.fft.ifft2(filtered_image)
    return filtered_image


def orientation_filter(
    image: Image.Image,
    angle_deg: float,
    freq_peak: float = 100.0,
    freq_sigma: float = 1000.0,
    theta_sigma_deg: float = 15.0,
    standardize_input: bool = True,
) -> Image.Image:
    """Filter a PIL image with a log-Gabor filter tuned to a specific angle.

    Parameters
    ----------
    image:
        Input PIL image. It is internally converted to grayscale.
    angle_deg:
        Orientation of the filter in degrees. The function wraps angles to
        [0, 180) so that 0 and 180 represent the same orientation.
    freq_peak:
        Peak spatial frequency value forwarded to ``DoLogGabor``.
    freq_sigma:
        Bandwidth parameter for the log-Gabor filter.
    theta_sigma_deg:
        Angular bandwidth (degrees) used to build the filter.
    standardize_input:
        When ``True`` the grayscale image is z-score normalised before
        filtering, matching the behaviour in the original script.

    Returns
    -------
    PIL.Image.Image
        The filtered image with intensities scaled to uint8.
    """
    theta = math.radians(angle_deg % 180.0)
    theta_sigma = math.radians(theta_sigma_deg)

    image_arr = _to_float_grayscale(image)
    if standardize_input:
        image_arr = image_arr - image_arr.mean()
        std = image_arr.std()
        if std > 0:
            image_arr = image_arr / std

    filtered = DoLogGabor(
        image_arr,
        FreqPeak=freq_peak,
        FreqSigma=freq_sigma,
        theta=theta,
        ThetaSigma=theta_sigma,
        graphics=False,
    )

    return _to_uint8_image(np.real(filtered))


def scramble_image(
    image: Image.Image,
    *,
    normalize_output: bool = True,
    seed: Optional[int] = None,
) -> Image.Image:
    """Generate a phase-scrambled version of the input image.

    Parameters
    ----------
    image:
        Input PIL image to scramble.
    normalize_output:
        When ``True`` the resulting scrambled image is scaled back to [0, 1]
        before converting to uint8 pixel values.
    seed:
        Optional random seed for reproducible scrambling.
    """
    arr = _to_float_grayscale(image)

    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    # Re-implementing the logic keeps the function self-contained.
    spectrum = np.fft.fft2(arr)
    amplitude = np.abs(spectrum)
    random_phase = np.angle(np.fft.fft2(np.random.randn(*arr.shape)))
    scrambled = np.fft.ifft2(amplitude * np.exp(1j * random_phase)).real

    if seed is not None and rng_state is not None:
        np.random.set_state(rng_state)

    if normalize_output:
        scrambled = scrambled - scrambled.min()
        ptp = np.ptp(scrambled)
        if ptp > 0:
            scrambled = scrambled / ptp

    return _to_uint8_image(scrambled)


def invert_image(image: Image.Image, method: str = "rotate") -> Image.Image:
    """Return an inverted variant of ``image``.

    Parameters
    ----------
    image:
        Input PIL image.
    method:
        * ``"rotate"`` (default) rotates the image by 180 degrees,
          matching the stimulus inversion used in the original pipeline.
        * ``"contrast"`` performs contrast negation on the luminance
          channel.
    """
    method = method.lower()
    if method == "rotate":
        return image.transpose(Image.ROTATE_180)
    if method == "contrast":
        grey = image.convert("L")
        inverted = ImageOps.invert(grey)
        if image.mode != "L":
            inverted = inverted.convert(image.mode)
        return inverted
    raise ValueError(f"Unknown inversion method '{method}'.")


__all__ = ["orientation_filter", "scramble_image", "invert_image"]
