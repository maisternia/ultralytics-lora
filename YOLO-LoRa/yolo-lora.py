# yolo-lora.py
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from typing import Optional
from ultralytics.utils import RANK
import torch
from torch import nn
# from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.tal import make_anchors
from infer import run_inference

# 1. Custom Loss
class MyCustomLoss(v8DetectionLoss):
    def __init__(self, model, height_loss_weight=0.5, *args, **kwargs):
        """Initialize with additional height loss weight parameter."""
        super().__init__(model, *args, **kwargs)
        self.height_loss_weight = height_loss_weight
        self.height_loss = nn.SmoothL1Loss(reduction='none')

    def __call__(self, preds, batch):
        loss, loss_items = super().__call__(preds, batch)
 
        return loss, loss_items

# 2. Custom Model
class MyCustomModel(DetectionModel):
    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """
        Ensure DetectionModel is initialized properly by calling super().__init__ with cfg/weights.
        Any extra args/kwargs are forwarded to the base class.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        # Return an instance of the custom loss bound to this model
        return MyCustomLoss(self)

# 3. Custom Trainer
class MyCustomTrainer(DetectionTrainer):
    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = MyCustomModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


if __name__ == "__main__":

    BASE_DIR = "YOLO-LoRa"
    MODELS_DIR = f"{BASE_DIR}/datasets"
    DS_NAME = "4-E2L1P20_FFT4096_BWSF125_9_725_12_POW-30_0_SM"
    RUNS_SUBDIR = "runs/detect"
    model_name = "RUN4_fraction0.25-scale0.5-box7.5-ph0.2"

    train_path = f"{MODELS_DIR}/{DS_NAME}/{RUNS_SUBDIR}/{model_name}"

    # Try to resume an incomplete run if a checkpoint exists; otherwise fall back to fresh training
    last_ckpt = Path(train_path) / "weights" / "last.pt"
    results = None
    if last_ckpt.exists():
        try:
            model = YOLO(str(last_ckpt))
            results = model.train(resume=True)
        except Exception as e:
            # resume failed -> will run a fresh training below
            # (keep message short; caller can inspect exception if needed)
            results = None

    if results is None:
        # Fresh training; ensure outputs go to the same train_path (project + name)
        pretrained_model="/Volumes/SSD-MLData/Unchirp/Products/datasets/4.2-added_-30db/runs/detect/4-n_size_tune-100/weights/best.pt"
        model = YOLO(pretrained_model)  # load a pretrained model
        results = model.train(
            # trainer=MyCustomTrainer,          # ← substitute standard Trainer if desired
            fraction=0.25,
            project=f"{MODELS_DIR}/{DS_NAME}/{RUNS_SUBDIR}",
            name=model_name,
            cfg="YOLO-LoRa/hyper_param.yaml"
        )

    # Get the path to the best weights from the results object
    # This assumes results.save_dir and best.pt exist in that directory
    if hasattr(results, "save_dir"):
        model_path = str(Path(results.save_dir) / "weights" / "best.pt")
    else:
        raise RuntimeError("Could not determine model_path from YOLO().train() results.")

    # Optionally, you can also get the model instance if needed:
    # model = results.model if hasattr(results, "model") else None

    # Now call run_inference with the trained model path
    run_inference(model_path, MODEL_RUN=model_name)