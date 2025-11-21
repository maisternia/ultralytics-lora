from ultralytics import YOLO
from convert import bandwidth_from_box_height, sf_from_proportional_scaling
import yaml
from pathlib import Path
import math
import statistics

# Constants (centralized configuration)
MODELS_DIR = "datasets"
RUNS_SUBDIR = "runs/detect"
WEIGHTS_REL = "weights/best.pt"
CONF_FILENAME = "conf.yaml"

DS_NAME = "4-E2L1P20_FFT4096_BWSF125_9_725_12_POW-30_0_SM"
DS_CONF = f"{MODELS_DIR}/{DS_NAME}/{CONF_FILENAME}"

# model runs
RUN1 = "4.2_tune-100-scale_0.1"
RUN2 = "4.2_tune-100-scale_0.5"
RUN3 = "4.2_tune-100-scale_0.05"
RUN4 = "4.2_tune-100-cont"
DEFAULT_MODEL_RUN = RUN4  # choose default run here

# inference / dataset sources
INFERENCE_DIR = "InferenceDS"
INFERENCE_DS = "BW800SF7"
SOURCE_FILENAME = "FMRecording(13).bb19.png"  # change filename here if needed
SOURCE = f"{INFERENCE_DIR}/{INFERENCE_DS}"#/{SOURCE_FILENAME}"

# prediction / processing parameters
IMGSZ = 1024                 # image size used for predict
CONF_THRESH = 0.7            # detection confidence threshold
SAVE_ANNOTATIONS = True
SAVE_TXT = False
PROJECT_DIR = "runs/predict"
EXP_NAME = "exp_test"
STREAM = True

# signal processing constants
SAMPLE_RATE = 32e6  # Hz, change to your real sampling rate
MODEL_FFT_SIZE = 4096     # FFT size used for spectrogram generation
SOURCE_FFT_SIZE = 1024  # FFT size used for spectrogram generation of input images

# Decimation rate = 2^(decimation factor)
MODEL_DECIMATION_FACTOR = 2 # Time decimation rate used in model training
SOURCE_DECIMATION_FACTOR = 0 # Time decimation rate used in source spectrograms

# spectrum mapping: divisor to produce the mapped vertical frequency span from sample_rate
# e.g. 1 => full range = sample_rate, 2 => full range = sample_rate/2 (Nyquist)
SPECTROGRAM_RANGE_FRACTION = SOURCE_FFT_SIZE / IMGSZ

# Accuracy target + tolerances for measured bandwidth (adjust as needed)
TARGET_BW_HZ = 812e3        # expected target bandwidth (Hz) to compare against
TOLERANCE_HZ = 50e3         # absolute tolerance in Hz
TOLERANCE_PCT = 0.05        # relative tolerance (5%)

# Accumulators for accuracy estimation
measured_bw_list = []
abs_errors = []
sq_errors = []
rel_errors = []
within_tol = 0
total_meas = 0

# new: load class pairs stored under 'names' in conf.yaml as list[(float,int)]
def load_class_pairs(conf_path):
    import re
    def parse_entry(v):
        # Handle list/tuple like [125, 9]
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return (float(v[0]), int(v[1]))
            except Exception:
                return None
        # Handle string like "125, 9" or "125 9" or "125,9"
        if isinstance(v, str):
            parts = [p for p in re.split(r"[,\s]+", v.strip()) if p != ""]
            if len(parts) >= 2:
                try:
                    return (float(parts[0]), int(float(parts[1])))
                except Exception:
                    return None
        # If already numeric pair in some other form, ignore other scalars
        return None

    p = Path(conf_path)
    if not p.exists():
        return []  # no conf file -> empty list
    with p.open("r") as f:
        conf = yaml.safe_load(f) or {}
    names = conf.get("names", conf.get("classes", {}))

    pairs = []
    if isinstance(names, dict):
        # try to sort by numeric class id when possible, else keep insertion order
        try:
            items = sorted(names.items(), key=lambda kv: int(kv[0]))
        except Exception:
            items = list(names.items())
        for _, v in items:
            entry = parse_entry(v)
            if entry is not None:
                pairs.append(entry)
    elif isinstance(names, list):
        for v in names:
            entry = parse_entry(v)
            if entry is not None:
                pairs.append(entry)
    return pairs

CLASS_PAIRS = load_class_pairs(DS_CONF)
# print("Loaded class pairs (float,int):", CLASS_PAIRS)

# build model path from constants
MODEL = DEFAULT_MODEL_RUN
model_path = f"{MODELS_DIR}/{DS_NAME}/{RUNS_SUBDIR}/{MODEL}/{WEIGHTS_REL}"
model = YOLO(model_path)

# use constants in predict call
results = model.predict(
    source=SOURCE,
    imgsz=IMGSZ,
    conf=CONF_THRESH,
    save=SAVE_ANNOTATIONS,
    save_txt=SAVE_TXT,
    project=PROJECT_DIR,
    name=EXP_NAME,
    stream=STREAM,
    show=False,      # do not open image display windows
    verbose=False    # suppress console logging/output from predict()
)

for result in results:
    if not CLASS_PAIRS:
        print("Warning: CLASS_PAIRS is empty — cannot map detections to (float,int) pairs.")
        continue

    for box in result.boxes:
        # use normalized coordinates (xyxyn) to avoid integer rounding
        coords_n = box.xyxyn.cpu().numpy().squeeze()
        if coords_n.size != 4:
            continue
        x1n, y1n, x2n, y2n = coords_n  # normalized [0..1]
        height_norm = float(y2n - y1n)  # normalized height (fraction)
        # optional: turn normalized bbox into pixel values only for display if needed
        try:
            img_w, img_h = int(result.orig_shape[1]), int(result.orig_shape[0])
            x1, y1, x2, y2 = int(x1n * img_w), int(y1n * img_h), int(x2n * img_w), int(y2n * img_h)
            height_px = int(y2 - y1)
        except Exception:
            # fallback to IMGSZ if orig_shape missing
            x1, y1, x2, y2 = None, None, None, None
            height_px = None
        height = height_px  # keep for backwards-compat where used, may be None

        # confidence and class may be tensors/scalars — convert safely
        try:
            conf = float(box.conf.cpu().item())
        except Exception:
            try:
                conf = float(box.conf)
            except Exception:
                conf = None
        try:
            cls_idx = int(box.cls.cpu().item())
        except Exception:
            try:
                cls_idx = int(box.cls)
            except Exception:
                cls_idx = None

        if cls_idx is None or cls_idx < 0 or cls_idx >= len(CLASS_PAIRS):
            print(f"Detection with invalid class index: {cls_idx} (conf={conf})")
            continue

        # original nominal class pair (e.g. [125, 9] meaning 125 kHz, SF=9)
        nominal_bw, nominal_sf = CLASS_PAIRS[cls_idx]

        # Дударек, Г.О. і Мартинюк, С.Є. 2025. Identifying LoRa parameters using convolutional neural networks. Вісті вищих учбових закладів. Радіоелектроніка. (Жов 2025). DOI:https://doi.org/10.20535/S0021347025020013
        # Formula (2.10):
        adjusted_bw = float(nominal_bw) * (MODEL_FFT_SIZE / SOURCE_FFT_SIZE)
        adjusted_sf = int(nominal_sf) - MODEL_DECIMATION_FACTOR + SOURCE_DECIMATION_FACTOR
        # convert nominal bw to Hz (assumes nominal in kHz in your conf.yaml)
        class_bw_hz = float(adjusted_bw) * 1e3

        # measured bandwidth from detected normalized box height (Hz), continuous (no bin-rounding)
        if SAMPLE_RATE and SAMPLE_RATE > 0:
            measured_bw_hz = bandwidth_from_box_height(height_norm, SAMPLE_RATE, range_fraction=SPECTROGRAM_RANGE_FRACTION)
            # predicted SF from proportional scaling
            try:
                sf_pred = sf_from_proportional_scaling(int(adjusted_sf), class_bw_hz, measured_bw_hz)
            except Exception:
                sf_pred = None
            # record accuracy metrics if we got a measurement
            if measured_bw_hz is not None:
                total_meas += 1
                err = measured_bw_hz - TARGET_BW_HZ
                abs_err = abs(err)
                abs_errors.append(abs_err)
                sq_errors.append(err * err)
                rel_errors.append(abs_err / TARGET_BW_HZ if TARGET_BW_HZ != 0 else 0.0)
                measured_bw_list.append(measured_bw_hz)
                if abs_err <= TOLERANCE_HZ or (abs_err / TARGET_BW_HZ) <= TOLERANCE_PCT:
                    within_tol += 1
        else:
            measured_bw_hz = None
            sf_pred = None


        print(
            f"Detected class={cls_idx} conf={conf} bbox_norm=({x1n:.4f},{y1n:.4f},{x2n:.4f},{y2n:.4f}) "
            f"height_norm={height_norm:.6f} bbox_px=({x1},{y1},{x2},{y2}) height_px={height} "
            f"-> nominal_bw={nominal_bw}, nominal_sf={nominal_sf}, adjusted_BW:{adjusted_bw}, adjusted_SF:{adjusted_sf} "
            f"measured_bw_hz={measured_bw_hz} predicted_SF={sf_pred}"
        )

# After loop: print accuracy summary
if total_meas > 0:
    mean_abs = statistics.mean(abs_errors)
    median_abs = statistics.median(abs_errors)
    rmse = math.sqrt(sum(sq_errors) / total_meas)
    mean_rel = statistics.mean(rel_errors) * 100.0
    pct_within = (within_tol / total_meas) * 100.0
    mean_measured = statistics.mean(measured_bw_list)
    print("\nBandwidth accuracy summary vs target {:.0f} kHz:".format(TARGET_BW_HZ/1e3))
    print(f" detections with measurement: {total_meas}")
    print(f" mean measured BW: {mean_measured/1e3:.3f} kHz")
    print(f" mean absolute error: {mean_abs/1e3:.3f} kHz")
    print(f" median absolute error: {median_abs/1e3:.3f} kHz")
    print(f" RMSE: {rmse/1e3:.3f} kHz")
    print(f" mean relative error: {mean_rel:.2f} %")
    print(f" within tolerance ({TOLERANCE_HZ/1e3:.0f} kHz or {TOLERANCE_PCT*100:.0f}%): {pct_within:.1f} %")
else:
    print("\nNo measured bandwidths collected; cannot compute accuracy.")

