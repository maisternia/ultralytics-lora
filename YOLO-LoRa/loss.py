# loss.py
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss

class LoRaDetectionLoss(v8DetectionLoss):
    """Add physics-guided box-height precision to YOLO11 detection loss."""

    def __init__(self, *args, height_weight: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.height_weight = height_weight

    def __call__(self, preds, batch):
        # Run base YOLO detection losses (box, cls, dfl, obj)
        loss, loss_items = super().__call__(preds, batch)

        # Extract normalized predicted vs GT boxes from batch (base stores them)
        pbox, tbox = getattr(self, "pred_boxes", None), getattr(self, "target_boxes", None)
        if pbox is None or tbox is None:
            raise RuntimeError("LoRaDetectionLoss: pred_boxes/target_boxes missing — check Ultralytics version.")
        pred_h = torch.clamp(pbox[:, 3], 1e-9)
        true_h = torch.clamp(tbox[:, 3], 1e-9)

        # Log-space SmoothL1 for scale-invariant height precision
        height_loss = F.smooth_l1_loss(torch.log(pred_h), torch.log(true_h))
        loss += self.height_weight * height_loss

        # Append for logging
        loss_items = torch.cat([loss_items, height_loss.detach().unsqueeze(0)])
        return loss, loss_items
