# YOLO-LoRa: 3-Channel Spectrogram Data Loader

## Overview

YOLO-LoRa provides specialized classes for loading and training YOLO models on 3-channel spectrogram data, particularly designed for LoRa signal detection tasks.

## Key Components

### 1. SpectrogramDataset

A custom dataset class that extends `YOLODataset` to handle 3-channel spectrogram data.

**Features:**
- Automatic spectrogram normalization
- Optional value clipping
- Support for standard YOLO data augmentation
- Compatible with all YOLO detection features

**Parameters:**
- `spectrogram_normalize` (bool): Whether to normalize spectrogram values to [0, 1] range (default: True)
- `spectrogram_clip_range` (tuple[float, float], optional): Min and max values for clipping spectrograms before normalization

### 2. SpectrogramTrainer

A custom trainer class that extends `DetectionTrainer` to handle spectrogram-specific data loading.

**Features:**
- Seamless integration with YOLO training pipeline
- Support for distributed training
- Custom preprocessing for spectrogram data

## Usage Examples

### Basic Training

```python
from ultralytics.models.yololora import SpectrogramTrainer

# Initialize trainer with configuration
trainer = SpectrogramTrainer(
    overrides={
        'model': 'yolo11n.pt',
        'data': 'lora_spectrograms.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'spectrogram_normalize': True,
    }
)

# Train the model
trainer.train()
```

### Custom Dataset Loading

```python
from ultralytics.models.yololora import SpectrogramDataset
from ultralytics.cfg import IterableSimpleNamespace

# Configuration
cfg = IterableSimpleNamespace({
    'imgsz': 640,
    'cache': False,
    'rect': False,
    'single_cls': False,
    'task': 'detect',
    'classes': None,
    'fraction': 1.0,
})

# Create dataset
dataset = SpectrogramDataset(
    img_path='path/to/spectrograms',
    imgsz=640,
    batch_size=16,
    augment=True,
    hyp=cfg,
    data={'names': {0: 'lora_signal'}, 'channels': 3},
    task='detect',
    spectrogram_normalize=True,
    spectrogram_clip_range=(10.0, 90.0),  # Optional clipping
)
```

### Using build_spectrogram_dataset Function

```python
from ultralytics.data import build_spectrogram_dataset
from ultralytics.cfg import IterableSimpleNamespace

# Configuration
cfg = IterableSimpleNamespace({
    'imgsz': 640,
    'cache': False,
    'rect': False,
    'single_cls': False,
    'task': 'detect',
    'classes': None,
    'fraction': 1.0,
    'spectrogram_normalize': True,
    'spectrogram_clip_range': None,
})

# Build dataset
dataset = build_spectrogram_dataset(
    cfg=cfg,
    img_path='path/to/spectrograms',
    batch=16,
    data={'names': {0: 'lora_signal'}, 'channels': 3},
    mode='train',
    stride=32,
)
```

### Data Format

The spectrogram dataset expects the following directory structure:

```
lora_spectrograms/
├── images/
│   ├── train/
│   │   ├── spec_001.png
│   │   ├── spec_002.png
│   │   └── ...
│   └── val/
│       ├── spec_101.png
│       └── ...
└── labels/
    ├── train/
    │   ├── spec_001.txt
    │   ├── spec_002.txt
    │   └── ...
    └── val/
        ├── spec_101.txt
        └── ...
```

### Dataset Configuration (YAML)

Create a `lora_spectrograms.yaml` file:

```yaml
# Dataset configuration for LoRa spectrograms
path: /path/to/lora_spectrograms  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')

# Classes
names:
  0: lora_signal
  1: noise
  2: interference

# Number of channels (for spectrograms)
channels: 3

# Spectrogram-specific parameters (optional)
spectrogram_normalize: true
spectrogram_clip_range: [10.0, 90.0]  # Optional: [min, max] for clipping
```

### CLI Training

```bash
# Train with spectrogram trainer
yolo train \
    model=yolo11n.pt \
    data=lora_spectrograms.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16
```

Note: For CLI usage, you need to configure the trainer in your training script to use `SpectrogramTrainer` instead of the default trainer.

## Technical Details

### Spectrogram Preprocessing

The `SpectrogramDataset` applies the following preprocessing steps:

1. **Loading**: Spectrograms are loaded as images (PNG, JPG, etc.) or numpy arrays (.npy)
2. **Clipping** (optional): Values are clipped to the specified range if `spectrogram_clip_range` is provided
3. **Normalization**: Values are normalized to [0, 1] range if `spectrogram_normalize=True`
4. **Conversion**: Normalized values are converted to uint8 [0, 255] range
5. **Channel handling**: Ensures 3-channel format (replicates grayscale if needed)

### Integration with YOLO Pipeline

The spectrogram dataset integrates seamlessly with the YOLO pipeline:

- Supports all YOLO augmentation techniques (mosaic, mixup, etc.)
- Compatible with rectangular training
- Supports caching to RAM or disk
- Works with distributed training

## Advanced Usage

### Custom Preprocessing

You can extend `SpectrogramDataset` to add custom preprocessing:

```python
from ultralytics.models.yololora.dataset import SpectrogramDataset
import numpy as np

class CustomSpectrogramDataset(SpectrogramDataset):
    def preprocess_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        # Apply custom preprocessing
        spec = super().preprocess_spectrogram(spectrogram)
        
        # Add your custom processing here
        # Example: Apply histogram equalization
        spec = cv2.equalizeHist(spec[:,:,0])
        
        return spec
```

### Training with Validation

```python
from ultralytics.models.yololora import SpectrogramTrainer

trainer = SpectrogramTrainer(
    overrides={
        'model': 'yolo11n.pt',
        'data': 'lora_spectrograms.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'patience': 50,  # Early stopping patience
        'save': True,     # Save checkpoints
        'val': True,      # Enable validation
    }
)

# Train with validation
results = trainer.train()
```

## Performance Considerations

1. **Caching**: Use `cache='ram'` or `cache='disk'` for faster training on repeated epochs
2. **Workers**: Adjust `workers` parameter based on CPU count for optimal data loading
3. **Batch Size**: Use the largest batch size that fits in GPU memory
4. **Image Size**: Balance between detection accuracy and training speed

## Troubleshooting

### Issue: "Spectrogram Not Found"
- Ensure spectrogram files exist in the specified directory
- Check file permissions
- Verify image file formats are supported (PNG, JPG, etc.)

### Issue: Training is slow
- Enable caching: `cache='ram'` or `cache='disk'`
- Increase number of workers
- Use smaller image size for faster training

### Issue: Out of memory
- Reduce batch size
- Reduce image size
- Use gradient accumulation
- Use mixed precision training (AMP)

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)
- [Custom Datasets](https://docs.ultralytics.com/datasets/)
