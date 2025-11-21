from ultralytics import YOLO

# Load the model you trained
DS_NAME = "4-E2L1P20_FFT4096_BWSF125_9_725_12_POW-30_0_SM"
RUN1 = "4.2_tune-100-scale_0.1"
RUN2 = "4.2_tune-100-scale_0.5"
RUN3 = "4.2_tune-100-scale_0.05"

# Replace single-run validation with multi-run validation and comparison
RUNS = [RUN1, RUN2, RUN3]

def _to_list(m):
	# convert metrics.box.maps to a plain python list of floats
	try:
		return list(m)
	except TypeError:
		try:
			return m.tolist()
		except Exception:
			return [float(m)]

results = {}
for run in RUNS:
	model_path = f"Models/{DS_NAME}/runs/detect/{run}/weights/best.pt"
	print(f"\nValidating run: {run} -> {model_path}")
	model = YOLO(model_path)

	metrics = model.val(
		data=f"Models/{DS_NAME}/conf.yaml",
		imgsz=1024,
		device="mps",
		save=True,
		plots=True
	)

	# collect per-class maps and a mean mAP
	maps_raw = metrics.box.maps
	maps_list = _to_list(maps_raw)
	mean_map = sum(maps_list) / len(maps_list) if len(maps_list) > 0 else float(getattr(metrics.box, "map", 0.0))

	results[run] = {
		"maps": maps_list,
		"mean_map": mean_map,
		"map50": float(getattr(metrics.box, "map50", 0.0)),
		"map75": float(getattr(metrics.box, "map75", 0.0)),
	}

	# print run summary
	print(" per-class mAPs:", maps_list)
	print(f" mean per-class mAP: {mean_map:.6f}")
	print(f" mAP@0.5: {results[run]['map50']:.6f}, mAP@0.75: {results[run]['map75']:.6f}")

# compare runs by mean_map
print("\nComparison by mean per-class mAP:")
sorted_runs = sorted(results.items(), key=lambda x: x[1]["mean_map"], reverse=True)
for idx, (run, info) in enumerate(sorted_runs, 1):
	print(f"{idx}. {run}: mean_map={info['mean_map']:.6f}, map50={info['map50']:.6f}, map75={info['map75']:.6f}")

best_run = sorted_runs[0][0] if sorted_runs else None
if best_run:
	print(f"\nBest run by mean per-class mAP: {best_run}")
else:
	print("\nNo results collected.")

