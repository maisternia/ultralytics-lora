#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Example script for training YOLO on 3-channel spectrogram data.

This script demonstrates how to use the SpectrogramTrainer and SpectrogramDataset
classes for training YOLO models on LoRa spectrogram data.

Usage:
    python examples/yololora_train.py
"""

from pathlib import Path

from ultralytics.models.yololora import SpectrogramTrainer


def train_spectrogram_model():
    """Train a YOLO model on spectrogram data."""
    # Configuration for spectrogram training
    config = {
        # Model and data
        "model": "yolo11n.pt",  # Pre-trained model
        "data": "lora_spectrograms.yaml",  # Dataset configuration
        # Training parameters
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        # Spectrogram-specific parameters
        "spectrogram_normalize": True,
        "spectrogram_clip_range": None,  # Optional: (min, max) for clipping
        # Optimization
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        # Augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        # Other settings
        "patience": 50,
        "save": True,
        "val": True,
        "plots": True,
    }

    # Initialize trainer
    trainer = SpectrogramTrainer(overrides=config)

    # Train the model
    print("Starting spectrogram model training...")
    results = trainer.train()

    print(f"Training completed. Best model saved to: {trainer.best}")
    return results


def train_with_custom_preprocessing():
    """Train with custom spectrogram preprocessing."""
    config = {
        "model": "yolo11n.pt",
        "data": "lora_spectrograms.yaml",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        # Custom preprocessing with clipping
        "spectrogram_normalize": True,
        "spectrogram_clip_range": (10.0, 90.0),  # Clip values between 10 and 90
    }

    trainer = SpectrogramTrainer(overrides=config)
    results = trainer.train()
    return results


def create_example_dataset_config():
    """Create an example dataset configuration file."""
    yaml_content = """# LoRa Spectrogram Dataset Configuration

# Dataset paths
path: /path/to/lora_spectrograms  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')

# Class names
names:
  0: lora_signal
  1: noise
  2: interference

# Number of channels
channels: 3

# Spectrogram-specific parameters
spectrogram_normalize: true
spectrogram_clip_range: null  # Optional: [min, max] for clipping
"""

    config_path = Path("lora_spectrograms.yaml")
    if not config_path.exists():
        with open(config_path, "w") as f:
            f.write(yaml_content)
        print(f"Example dataset config created: {config_path}")
    else:
        print(f"Dataset config already exists: {config_path}")


if __name__ == "__main__":
    # Create example dataset configuration
    create_example_dataset_config()

    print("\nExample 1: Basic spectrogram training")
    print("=" * 50)
    # Uncomment to run training:
    # train_spectrogram_model()

    print("\nExample 2: Training with custom preprocessing")
    print("=" * 50)
    # Uncomment to run training:
    # train_with_custom_preprocessing()

    print("\nTo run training, uncomment the training calls in this script.")
    print("Make sure you have:")
    print("1. Created the lora_spectrograms.yaml configuration file")
    print("2. Prepared your spectrogram dataset in the correct format")
    print("3. Installed all required dependencies")
