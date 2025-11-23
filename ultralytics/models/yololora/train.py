# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Spectrogram trainer class for YOLO-LoRa.

This module provides a custom trainer class for training YOLO models on 3-channel
spectrogram data for LoRa signal detection tasks.
"""

from __future__ import annotations

from typing import Any

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yololora.dataset import SpectrogramDataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class SpectrogramTrainer(DetectionTrainer):
    """
    A trainer class for training YOLO models on 3-channel spectrogram data.

    This trainer extends DetectionTrainer to handle spectrogram-specific data loading
    and preprocessing for LoRa signal detection tasks.

    Attributes:
        spectrogram_normalize (bool): Whether to normalize spectrogram values.
        spectrogram_clip_range (tuple): Min and max values for clipping spectrograms.

    Methods:
        build_dataset: Build spectrogram dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.

    Examples:
        >>> from ultralytics.models.yololora import SpectrogramTrainer
        >>> args = dict(model="yolo11n.pt", data="lora_spectrograms.yaml", epochs=10)
        >>> trainer = SpectrogramTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize SpectrogramTrainer with configuration and overrides.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        # Extract spectrogram-specific parameters from overrides
        if overrides is None:
            overrides = {}

        self.spectrogram_normalize = overrides.pop("spectrogram_normalize", True)
        self.spectrogram_clip_range = overrides.pop("spectrogram_clip_range", None)

        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Build spectrogram dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing spectrogram images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (SpectrogramDataset): Spectrogram dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return SpectrogramDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=self.args,
            rect=self.args.rect or mode == "val",  # rectangular batches
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            spectrogram_normalize=self.spectrogram_normalize,
            spectrogram_clip_range=self.spectrogram_clip_range,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the spectrogram dataset.
            batch_size (int): Number of spectrograms per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            from ultralytics.utils import LOGGER

            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )
