import os
import sys
import types
import torch
import torch.nn.functional as F

# Ensure project package dir is importable
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Create a minimal stub for ultralytics.yolo.engine.loss if not present
uh_module_name = "ultralytics.yolo.engine.loss"
if uh_module_name not in sys.modules:
    # Build module chain
    sys.modules.setdefault("ultralytics", types.ModuleType("ultralytics"))
    sys.modules.setdefault("ultralytics.yolo", types.ModuleType("ultralytics.yolo"))
    sys.modules.setdefault("ultralytics.yolo.engine", types.ModuleType("ultralytics.yolo.engine"))
    fake_loss_mod = types.ModuleType(uh_module_name)
    class ComputeLoss:
        def __init__(self, *args, **kwargs):
            pass
        def forward(self, preds, batch):
            # placeholder; will be patched in test
            raise NotImplementedError
    fake_loss_mod.ComputeLoss = ComputeLoss
    sys.modules[uh_module_name] = fake_loss_mod

# Import the class under test
from loss import ComputeLossLoRa  # noqa: E402

def test_compute_loss_lora_height_loss():
    # Patch the parent ComputeLoss.forward to return fixed scalar losses and a loss_items vector
    base_module = sys.modules[uh_module_name]
    BaseComputeLoss = base_module.ComputeLoss
    orig_forward = BaseComputeLoss.forward

    def dummy_forward(self, preds, batch):
        # lbox, lcls, lobj, loss_items_vector
        return torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), torch.tensor([0.1, 0.2, 0.3])

    BaseComputeLoss.forward = dummy_forward

    try:
        # Instantiate without running __init__
        inst = object.__new__(ComputeLossLoRa)
        # set weight
        inst.height_weight = 2.0

        # set matched boxes (xywh normalized). two boxes for the test
        inst.pred_boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2],
                                        [0.2, 0.2, 0.3, 0.5]], dtype=torch.float32)
        inst.target_boxes = torch.tensor([[0.1, 0.1, 0.15, 0.25],
                                          [0.2, 0.2, 0.3, 0.45]], dtype=torch.float32)

        # Call forward (super().forward is patched)
        total_loss, loss_items = ComputeLossLoRa.forward(inst, preds=None, batch=None)

        # Compute expected height loss: smooth_l1_loss(log(pred_h_safe), log(true_h_safe))
        pred_h = inst.pred_boxes[:, 3]
        true_h = inst.target_boxes[:, 3]
        pred_h_safe = torch.clamp(pred_h, 1e-9)
        true_h_safe = torch.clamp(true_h, 1e-9)
        expected_height_loss = F.smooth_l1_loss(torch.log(pred_h_safe), torch.log(true_h_safe))
        expected_total = (1.0 + 2.0 + 3.0) + inst.height_weight * expected_height_loss

        # Assertions
        assert torch.allclose(total_loss, torch.tensor(expected_total), atol=1e-6)
        # original loss_items length was 3, now should be 4
        assert loss_items.numel() == 4
        # last element should equal detached height loss
        assert torch.allclose(loss_items[-1], expected_height_loss.detach(), atol=1e-6)
    finally:
        # Restore original method to avoid side effects
        BaseComputeLoss.forward = orig_forward