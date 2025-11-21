# trainer.py
from ultralytics.models.yolo.detect import DetectionTrainer
from loss import LoRaDetectionLoss

class TrainerLoRa(DetectionTrainer):
    """Trainer using custom height-precision loss."""

    def get_loss(self):
        # CLI/YAML parameter automatically injected into self.args
        height_weight = getattr(self.args, "height_weight", 2.0)
        return LoRaDetectionLoss(self.model, self.hyp, self.device,
                               height_weight=height_weight)

    def set_metrics(self):
        # extend loss names for console/TB logging
        # YOLO normally logs ['box', 'cls', 'dfl', 'obj']
        self.loss_names = ['box', 'cls', 'dfl', 'obj', 'height']
