# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Spectrogram dataset class for YOLO-LoRa.

This module provides a custom dataset class for loading 3-channel spectrogram data
for object detection tasks using YOLO models.
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.patches import imread


class SpectrogramDataset(YOLODataset):
    """
    Dataset class for loading 3-channel spectrogram data in YOLO format.

    This class extends YOLODataset to handle spectrogram data with 3 channels,
    suitable for LoRa signal detection and classification tasks.

    The spectrograms are expected to be stored as images with 3 channels representing
    different frequency or time characteristics of the signal.

    Attributes:
        spectrogram_normalize (bool): Whether to normalize spectrogram values.
        spectrogram_clip_range (tuple): Min and max values for clipping spectrograms.

    Methods:
        load_image: Load a spectrogram image with special handling for 3-channel data.
        preprocess_spectrogram: Apply spectrogram-specific preprocessing.

    Examples:
        >>> dataset = SpectrogramDataset(
        ...     img_path="path/to/spectrograms", data={"names": {0: "lora_signal"}}, task="detect"
        ... )
    """

    def __init__(
        self,
        *args,
        spectrogram_normalize: bool = True,
        spectrogram_clip_range: tuple[float, float] | None = None,
        **kwargs,
    ):
        """
        Initialize SpectrogramDataset with spectrogram-specific parameters.

        Args:
            *args: Positional arguments passed to parent YOLODataset.
            spectrogram_normalize (bool): Whether to normalize spectrogram values to [0, 1].
            spectrogram_clip_range (tuple[float, float], optional): Min and max values for clipping.
            **kwargs: Keyword arguments passed to parent YOLODataset.
        """
        self.spectrogram_normalize = spectrogram_normalize
        self.spectrogram_clip_range = spectrogram_clip_range
        super().__init__(*args, **kwargs)

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load a spectrogram image from dataset index 'i' with spectrogram-specific processing.

        Args:
            i (int): Index of the spectrogram to load.
            rect_mode (bool): Whether to use rectangular resizing.

        Returns:
            im (np.ndarray): Loaded spectrogram as a NumPy array with shape (H, W, 3).
            hw_original (tuple[int, int]): Original dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the spectrogram file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy spectrogram file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR

            if im is None:
                raise FileNotFoundError(f"Spectrogram Not Found {f}")

            # Apply spectrogram-specific preprocessing
            im = self.preprocess_spectrogram(im)

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(int(w0 * r), self.imgsz), min(int(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Ensure 3-channel format
            if im.ndim == 2:
                # If grayscale, replicate to 3 channels
                im = np.stack([im] * 3, axis=-1)
            elif im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def preprocess_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply spectrogram-specific preprocessing.

        This method handles:
        - Value clipping to specified range
        - Normalization to [0, 255] for uint8 format

        Args:
            spectrogram (np.ndarray): Raw spectrogram data.

        Returns:
            (np.ndarray): Preprocessed spectrogram in uint8 format.
        """
        # Convert to float for processing
        spec = spectrogram.astype(np.float32)

        # Apply clipping if specified
        if self.spectrogram_clip_range is not None:
            vmin, vmax = self.spectrogram_clip_range
            spec = np.clip(spec, vmin, vmax)

        # Normalize to [0, 1] range
        if self.spectrogram_normalize:
            spec_min = spec.min()
            spec_max = spec.max()
            if spec_max > spec_min:
                spec = (spec - spec_min) / (spec_max - spec_min)
            else:
                spec = np.zeros_like(spec)

        # Convert to uint8 [0, 255] range
        spec = (spec * 255).astype(np.uint8)

        return spec
