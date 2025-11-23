# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Tests for YOLO-LoRa spectrogram dataset and trainer.

This module contains tests for the SpectrogramDataset and SpectrogramTrainer classes.
"""

import numpy as np
import pytest


def test_spectrogram_dataset_import():
    """Test that SpectrogramDataset can be imported."""
    from ultralytics.models.yololora import SpectrogramDataset

    assert SpectrogramDataset is not None


def test_spectrogram_trainer_import():
    """Test that SpectrogramTrainer can be imported."""
    from ultralytics.models.yololora import SpectrogramTrainer

    assert SpectrogramTrainer is not None


def test_build_spectrogram_dataset_import():
    """Test that build_spectrogram_dataset can be imported."""
    from ultralytics.data import build_spectrogram_dataset

    assert build_spectrogram_dataset is not None


def test_spectrogram_preprocessing():
    """Test spectrogram preprocessing functionality."""
    from ultralytics.models.yololora.dataset import SpectrogramDataset
    from unittest.mock import Mock

    # Create a mock dataset instance with proper initialization
    dataset = Mock(spec=SpectrogramDataset)
    dataset.spectrogram_normalize = True
    dataset.spectrogram_clip_range = None

    # Use the actual preprocessing method
    preprocess = SpectrogramDataset.preprocess_spectrogram.__get__(dataset, SpectrogramDataset)

    # Test with random spectrogram data
    spectrogram = np.random.rand(100, 100, 3).astype(np.float32) * 100

    # Process the spectrogram
    processed = preprocess(spectrogram)

    # Verify output properties
    assert processed.dtype == np.uint8
    assert processed.shape == spectrogram.shape
    assert processed.min() >= 0
    assert processed.max() <= 255


def test_spectrogram_preprocessing_with_clipping():
    """Test spectrogram preprocessing with clipping."""
    from ultralytics.models.yololora.dataset import SpectrogramDataset
    from unittest.mock import Mock

    # Create a mock dataset instance with clipping
    dataset = Mock(spec=SpectrogramDataset)
    dataset.spectrogram_normalize = True
    dataset.spectrogram_clip_range = (10.0, 90.0)

    # Use the actual preprocessing method
    preprocess = SpectrogramDataset.preprocess_spectrogram.__get__(dataset, SpectrogramDataset)

    # Test with random spectrogram data
    spectrogram = np.random.rand(100, 100, 3).astype(np.float32) * 100

    # Process the spectrogram
    processed = preprocess(spectrogram)

    # Verify output properties
    assert processed.dtype == np.uint8
    assert processed.shape == spectrogram.shape
    assert processed.min() >= 0
    assert processed.max() <= 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
