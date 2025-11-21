from ultralytics import YOLO
from convert import bandwidth_from_box_height, sf_from_proportional_scaling
import yaml
from pathlib import Path
import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
import csv

def run_inference(
    model_path,
    MODEL_RUN="RUN8",
    DS_NAME="4-E2L1P20_FFT4096_BWSF125_9_725_12_POW-30_0_SM",
    INFERENCE_DS="BW800SF7",
    SOURCE_FILENAME="FMRecording(13).bb19.png",
    IMGSZ=1024,
    CONF_THRESH=0.7,
    SAVE_ANNOTATIONS=False,
    SAVE_TXT=False,
    PROJECT_DIR="runs/predict",
    STREAM=True,
    SAMPLE_RATE=32e6,
    MODEL_FFT_SIZE=4096,
    TARGET_BW_HZ=203e3,
    TOLERANCE_HZ=50e3,
    TOLERANCE_PCT=0.05,
):
    EXP_NAME=MODEL_RUN
    # Constants (centralized configuration)
    BASE_DIR = "YOLO-LoRa"
    MODELS_DIR = f"{BASE_DIR}/datasets"
    # RUNS_SUBDIR, WEIGHTS_REL, CONF_FILENAME are not used after DS_CONF
    DS_CONF = f"{MODELS_DIR}/{DS_NAME}/data.yaml"

    INFERENCE_DIR = f"{BASE_DIR}/InferenceDS"
    SOURCE = f"{INFERENCE_DIR}/{INFERENCE_DS}"

    SPECTROGRAM_RANGE_FRACTION = MODEL_FFT_SIZE / IMGSZ

    # Remove duplicate/unused accumulators
    measured_bw_list = []
    abs_errors = []
    sq_errors = []
    rel_errors = []
    within_tol = 0
    total_meas = 0
    measurement_details = []

    # Ensure experiment name is unique under PROJECT_DIR by appending _1, _2, ... if needed.
    def unique_exp_name(project_dir: str, exp_name: str) -> str:
        """
        Return a non-conflicting experiment name by appending _N if PROJECT_DIR/exp_name exists.
        """
        base = Path(project_dir)
        candidate = base / exp_name
        if not candidate.exists():
            return exp_name
        i = 1
        while True:
            new_name = f"{exp_name}_{i}"
            if not (base / new_name).exists():
                return new_name
            i += 1

    # Make EXP_NAME unique before running predict
    # EXP_NAME_UNIQ = unique_exp_name(PROJECT_DIR, MODEL_RUN)
    EXP_NAME_UNIQ = unique_exp_name(PROJECT_DIR, EXP_NAME)
    print(f"Using experiment name: {EXP_NAME_UNIQ}")

    # signal processing constants
    SAMPLE_RATE = 32e6  # Hz, change to your real sampling rate
    MODEL_FFT_SIZE = 4096     # FFT size used for spectrogram generation

    # spectrum mapping: divisor to produce the mapped vertical frequency span from sample_rate
    # e.g. 1 => full range = sample_rate, 2 => full range = sample_rate/2 (Nyquist)
    SPECTROGRAM_RANGE_FRACTION = MODEL_FFT_SIZE / IMGSZ

    # Accuracy target + tolerances for measured bandwidth (adjust as needed)
    TARGET_BW_HZ = 203e3        # expected target bandwidth (Hz) to compare against
    TOLERANCE_HZ = 50e3         # absolute tolerance in Hz
    TOLERANCE_PCT = 0.05        # relative tolerance (5%)

    # Accumulators for accuracy estimation
    measured_bw_list = []
    abs_errors = []
    sq_errors = []
    rel_errors = []
    within_tol = 0
    total_meas = 0
    measurement_details = []  # Add: list to store (measured_bw_hz, height_norm, confidence)

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

    model = YOLO(model_path)

    results = model.predict(
        source=SOURCE,
        imgsz=IMGSZ,
        conf=CONF_THRESH,
        save=SAVE_ANNOTATIONS,
        save_txt=SAVE_TXT,
        project=PROJECT_DIR,
        name=EXP_NAME_UNIQ,
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

            # convert nominal bw to Hz (assumes nominal in kHz in your conf.yaml)
            class_bw_hz = float(nominal_bw) * 1e3

            # measured bandwidth from detected normalized box height (Hz), continuous (no bin-rounding)
            if SAMPLE_RATE and SAMPLE_RATE > 0:
                measured_bw_hz = bandwidth_from_box_height(height_norm, SAMPLE_RATE, range_fraction=SPECTROGRAM_RANGE_FRACTION)
                # predicted SF from proportional scaling
                try:
                    sf_pred = sf_from_proportional_scaling(int(nominal_sf), class_bw_hz, measured_bw_hz)
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
                    measurement_details.append((measured_bw_hz, height_norm, conf))  # Add: store details
                    if abs_err <= TOLERANCE_HZ or (abs_err / TARGET_BW_HZ) <= TOLERANCE_PCT:
                        within_tol += 1
            else:
                measured_bw_hz = None
                sf_pred = None


            print(
                f"Detected class={cls_idx} conf={conf} bbox_norm=({x1n:.4f},{y1n:.4f},{x2n:.4f},{y2n:.4f}) "
                f"height_norm={height_norm:.6f} bbox_px=({x1},{y1},{x2},{y2}) height_px={height} "
                f"-> nominal_bw={nominal_bw}, nominal_sf={nominal_sf},"
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

    # Append: plot Gaussian distribution of measured_bandwidth - TARGET_BW_HZ
    try:
        if len(measured_bw_list) > 0:
            # diffs in Hz -> convert to kHz for plotting
            diffs = np.array(measured_bw_list, dtype=float) - float(TARGET_BW_HZ)
            diffs_khz = diffs / 1e3

            # Clip to ±20 kHz for plotting (outliers excluded for visualization)
            CLIP_KHZ = 20.0
            diffs_plot = diffs_khz[(diffs_khz >= -CLIP_KHZ) & (diffs_khz <= CLIP_KHZ)]

            if diffs_plot.size == 0:
                print(f"No measured bandwidths within ±{CLIP_KHZ} kHz to plot.")
            else:
                mu = diffs_plot.mean()
                sigma = diffs_plot.std(ddof=0)

                # Prepare figure: histogram (kHz) + Gaussian PDF overlay (fixed x-range)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(diffs_plot, bins=40, density=True, alpha=0.6, color='C0', label='Measured - Target (kHz)')

                x = np.linspace(-CLIP_KHZ, CLIP_KHZ, 400)
                # protect against sigma == 0
                if sigma <= 0:
                    pdf = np.zeros_like(x)
                    pdf[np.abs(x - mu) < 1e-6] = 1.0  # degenerate spike (visual)
                else:
                    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ax.plot(x, pdf, 'r-', lw=2, label=f'Gaussian fit\nmu={mu:.2f} kHz\nsigma={sigma:.2f} kHz')

                ax.axvline(0.0, color='k', linestyle='--', linewidth=1)
                ax.set_xlim(-CLIP_KHZ, CLIP_KHZ)  # fixed horizontal scale ±20 kHz
                ax.set_xlabel('Measured BW - Target BW (kHz)')
                ax.set_ylabel('Density')
                ax.set_title('Distribution of Measured Bandwidth Errors (clipped ±20 kHz)')
                ax.legend()

                # report how many points were excluded for context
                excluded = len(diffs_khz) - len(diffs_plot)
                if excluded > 0:
                    ax.text(
                        0.95,
                        0.95,
                        f'excluded: {excluded}',
                        transform=ax.transAxes,
                        ha='right',
                        va='top',
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
                    )

                # Ensure output directory exists and save (same logic as before)
                out_dir = Path(PROJECT_DIR) / EXP_NAME_UNIQ
                analysis_dir = out_dir / "analysis"
                analysis_dir.mkdir(parents=True, exist_ok=True)
                out_path = analysis_dir / f"{MODEL_RUN}_measured_bw_error_gaussian.png"
                fig.savefig(out_path, bbox_inches='tight', dpi=150)
                plt.close(fig)

                print(f"Saved measured bandwidth error gaussian plot to {out_path}")
                
                # Save accuracy report as text file
                try:
                    report_lines = [
                        "Bandwidth Accuracy Report",
                        "=" * 50,
                        f"Model Run: {MODEL_RUN}",
                        f"Target BW: {TARGET_BW_HZ/1e3:.0f} kHz",
                        f"Tolerance: ±{TOLERANCE_HZ/1e3:.0f} kHz or {TOLERANCE_PCT*100:.0f}%",
                        "",
                        "Summary Statistics",
                        "-" * 50,
                        f"Total detections with measurement: {total_meas}",
                        f"Mean measured BW: {mean_measured/1e3:.3f} kHz",
                        f"Mean absolute error: {mean_abs/1e3:.3f} kHz",
                        f"Median absolute error: {median_abs/1e3:.3f} kHz",
                        f"RMSE: {rmse/1e3:.3f} kHz",
                        f"Mean relative error: {mean_rel:.2f} %",
                        f"Within tolerance: {pct_within:.1f} % ({within_tol}/{total_meas})",
                        "",
                        "Gaussian Fit (clipped ±20 kHz)",
                        "-" * 50,
                        f"Mean (mu): {mu:.2f} kHz",
                        f"Std Dev (sigma): {sigma:.2f} kHz",
                        f"Data points included: {len(diffs_plot)}",
                        f"Data points excluded: {excluded}",
                    ]
                    
                    report_text = "\n".join(report_lines)
                    report_path = analysis_dir / f"{MODEL_RUN}_bandwidth_accuracy_report.txt"
                    with open(report_path, "w") as f:
                        f.write(report_text)
                    print(f"Saved accuracy report to {report_path}")
                except Exception as e:
                    print(f"Failed to save accuracy report: {e}")
                
                # Save measurement details to CSV
                try:
                    csv_path = analysis_dir / f"{MODEL_RUN}_measurement_details.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Measured BW (Hz)", "Height Norm", "Confidence"])
                        for meas_bw, h_norm, conf_val in measurement_details:
                            writer.writerow([f"{meas_bw:.2f}", f"{h_norm:.6f}", f"{conf_val:.4f}" if conf_val is not None else ""])
                    print(f"Saved measurement details to {csv_path}")
                except Exception as e:
                    print(f"Failed to save measurement details CSV: {e}")
        else:
            print("No measured bandwidths to plot.")
    except Exception as e:
        print(f"Failed to plot measured bandwidth gaussian: {e}")

if __name__ == "__main__":
    # Example usage: pass model_path as argument or use default
    BASE_DIR = "YOLO-LoRa"
    MODELS_DIR = f"{BASE_DIR}/datasets"
    DS_NAME = "4-E2L1P20_FFT4096_BWSF125_9_725_12_POW-30_0_SM"
    RUNS_SUBDIR = "runs/detect"
    WEIGHTS_REL = "weights/best.pt"

    RUNS = {
        "RUN1": "4.2_tune-100-scale_0.1",
        "RUN2": "4.2_tune-100-scale_0.5",
        "RUN3": "4.2_tune-100-scale_0.05",
        "RUN4": "4.2_tune-100-cont",
        "RUN5": "4.2_tune-100-heightloss10.0",
        "RUN6": "4.2_tune-100-scale0.5-phys2.0",
        "RUN7": "scale0.3-default_loss",
        "RUN8": "scale0.5-default_loss",
        "RUN9": "relevant-scale0.5-box7.5-height2.0",
    }
    MODEL_RUN = "RUN9"
    MODEL = RUNS[MODEL_RUN]

    model_path = f"{MODELS_DIR}/{DS_NAME}/{RUNS_SUBDIR}/{MODEL}/{WEIGHTS_REL}"
    run_inference(model_path, MODEL_RUN=MODEL_RUN, DS_NAME=DS_NAME)

