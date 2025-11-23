# YOLO-LoRa Directory - Path-Specific Instructions

## Directory Overview

**IMPORTANT**: The `YOLO-LoRa/` directory is for **experimental research only** and is **NOT part of the standard ultralytics-yolo package**. This directory exists only in the `lora-dev` branch for custom experiments and testing.

**Purpose**: Custom YOLO experiments for LoRa signal detection from spectrograms, including:
- Custom datasets and data preprocessing
- Custom loss functions (e.g., height-precision loss)
- Custom trainers and models
- Testing YOLO under different conditions
- Inference and validation scripts

**Location**: Available only in `lora-dev` branch
**Maintainer**: Personal experiments by repository owner
**Not for Production**: Code here is experimental and not subject to the same quality standards as the main ultralytics package

## Directory Structure

```
YOLO-LoRa/
├── datasets/              # Custom datasets for experiments
├── InferenceDS/          # Inference test datasets
├── tests/                # Experimental tests
├── yolo-lora.py         # Main training script with custom components
├── loss.py              # Custom loss functions (e.g., LoRaDetectionLoss)
├── trainer.py           # Custom trainer implementations (e.g., TrainerLoRa)
├── infer.py             # Inference scripts
├── validate.py          # Validation scripts
├── convert.py           # Conversion utilities
├── hyper_param.yaml     # Custom hyperparameters for experiments
└── test_import.py       # Import tests
```

## Guidelines for Working in YOLO-LoRa

### DO:

1. **Experiment Freely**: This is a sandbox for experimentation. Feel free to:
   - Try new loss functions and training strategies
   - Test different model architectures
   - Create custom datasets and preprocessing pipelines
   - Modify hyperparameters aggressively
   - Break things to learn

2. **Document Your Experiments**: Add comments explaining:
   - What you're testing and why
   - Expected vs actual results
   - Hyperparameter choices and their rationale
   - Any interesting findings or observations

3. **Keep Self-Contained**: 
   - Store all experimental data, models, and results within YOLO-LoRa/
   - Use relative paths when referencing files in this directory
   - Don't modify the main ultralytics package unless absolutely necessary

4. **Use Custom Components**:
   ```python
   # Example: Custom Loss
   from loss import LoRaDetectionLoss
   
   # Example: Custom Trainer
   from trainer import TrainerLoRa
   model.train(trainer=TrainerLoRa, ...)
   ```

5. **Organize by Experiment**:
   - Use descriptive names for model runs (e.g., "RUN4_fraction0.25-scale0.5-box7.5-ph0.2")
   - Keep track of what parameters changed between runs
   - Store results in organized subdirectories

### DON'T:

1. **Don't Apply Production Standards**: 
   - No need for comprehensive unit tests
   - Code doesn't need to follow strict style guidelines
   - Experimental code can be messy - that's OK
   - Don't spend time on edge cases or error handling unless needed

2. **Don't Commit Large Files**:
   - Model weights (*.pt files)
   - Large datasets
   - Inference results and visualizations
   - Use .gitignore to exclude these (already configured)

3. **Don't Break the Main Package**:
   - Changes here should not affect ultralytics/ code
   - If you need to import from ultralytics, use standard imports
   - Don't modify ultralytics package behavior globally

4. **Don't Merge to Main**:
   - YOLO-LoRa should stay in lora-dev branch
   - Never merge experimental code to the main branch
   - This directory is for research, not production

## Common Workflows

### Training with Custom Loss

```python
# yolo-lora.py example
from ultralytics import YOLO
from loss import LoRaDetectionLoss
from trainer import TrainerLoRa

# Load pretrained model
model = YOLO("path/to/pretrained.pt")

# Train with custom trainer and loss
results = model.train(
    trainer=TrainerLoRa,
    data="YOLO-LoRa/datasets/your-dataset/data.yaml",
    cfg="YOLO-LoRa/hyper_param.yaml",
    epochs=100,
    imgsz=1024,
    batch=16,
)
```

### Creating Custom Loss Functions

```python
# loss.py example
from ultralytics.utils.loss import v8DetectionLoss
import torch.nn.functional as F

class MyCustomLoss(v8DetectionLoss):
    def __init__(self, *args, custom_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weight = custom_weight
    
    def __call__(self, preds, batch):
        # Get base YOLO losses
        loss, loss_items = super().__call__(preds, batch)
        
        # Add your custom loss term
        custom_term = self.compute_custom_loss(preds, batch)
        loss += self.custom_weight * custom_term
        
        return loss, loss_items
```

### Custom Trainer Setup

```python
# trainer.py example
from ultralytics.models.yolo.detect import DetectionTrainer
from loss import MyCustomLoss

class MyCustomTrainer(DetectionTrainer):
    def get_loss(self):
        return MyCustomLoss(self.model, self.hyp, self.device)
    
    def set_metrics(self):
        # Extend loss names for logging
        self.loss_names = ['box', 'cls', 'dfl', 'obj', 'custom']
```

### Running Inference

```python
# infer.py usage
from infer import run_inference

# Run inference with trained model
run_inference(
    model_path="path/to/best.pt",
    MODEL_RUN="experiment_name",
    SOURCE="YOLO-LoRa/InferenceDS/test_images",
    CONF_THRESH=0.7,
    IMGSZ=1024,
)
```

## Hyperparameter Configuration

The `hyper_param.yaml` file contains experiment-specific hyperparameters:

```yaml
# Key parameters for LoRa experiments
task: detect
data: YOLO-LoRa/datasets/your-dataset/data.yaml
epochs: 100
batch: 16
imgsz: 1024
device: mps  # or 'cuda', 'cpu'
workers: 0   # keep 0 for reproducible research
seed: 0      # keep 0 for reproducible research
deterministic: true

# Custom augmentation
scale: 0.5
box: 7.5     # adjust for your use case
mosaic: 0.0
mixup: 0.0
```

## Dataset Organization

```
YOLO-LoRa/datasets/
└── experiment-name/
    ├── data.yaml           # Dataset configuration
    ├── images/
    │   ├── train/         # Training images
    │   └── val/           # Validation images
    ├── labels/
    │   ├── train/         # YOLO format labels
    │   └── val/
    └── runs/              # Training outputs
        └── detect/
            └── run-name/
                ├── weights/
                │   ├── best.pt
                │   └── last.pt
                └── results.csv
```

## Testing and Validation

- Unit tests in `YOLO-LoRa/tests/` are optional and for your convenience
- Focus on empirical validation: does the model perform well on your task?
- Use `validate.py` to run validation with custom metrics
- Compare results across different experiment configurations

## Tips for Experimentation

1. **Start Small**: Test with `fraction=0.1` for quick iterations
2. **Resume Training**: Use `resume=True` to continue interrupted runs
3. **Track Everything**: Keep notes on what works and what doesn't
4. **Version Your Configs**: Save hyper_param.yaml for each experiment
5. **Visualize Results**: Use `plots=True` to generate training visualizations
6. **Compare Systematically**: Change one variable at a time when possible

## Common Issues

### Import Errors
- Make sure ultralytics is installed: `pip install -e .`
- Check that you're importing from the correct paths
- YOLO-LoRa modules should use relative imports when referencing each other

### Training Issues
- Check dataset paths in data.yaml
- Verify image size matches your dataset (imgsz parameter)
- Ensure sufficient disk space for checkpoints and logs
- Monitor GPU/CPU memory usage

### Model Loading
- Verify pretrained model path exists
- Check that model architecture matches pretrained weights
- Use absolute paths or paths relative to script location

## Integration with Main Ultralytics Package

When you need to use ultralytics functionality:

```python
# Standard imports from ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

# Your custom extensions
from loss import LoRaDetectionLoss  # Local to YOLO-LoRa
from trainer import TrainerLoRa      # Local to YOLO-LoRa
```

## Version Compatibility

- This experimental code is designed for Ultralytics YOLO v8.3.223+
- May require adjustments for different YOLO versions
- Test compatibility when updating the base ultralytics package

## Support and Resources

- **Main Docs**: See root-level README.md and ultralytics documentation
- **Questions**: This is personal experimental code - troubleshoot independently
- **Examples**: Look at existing scripts in YOLO-LoRa for patterns
- **Ultralytics API**: https://docs.ultralytics.com for base functionality

## Remember

This directory is YOUR experimental playground in the lora-dev branch:
- ✅ Break things and learn
- ✅ Try unconventional approaches
- ✅ Keep experiments organized
- ✅ Document interesting findings
- ❌ Don't commit large files
- ❌ Don't affect the main ultralytics package
- ❌ Don't merge to main branch

**Have fun experimenting! 🚀**
