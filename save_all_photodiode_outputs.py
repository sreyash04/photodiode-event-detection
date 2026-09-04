import os
import glob
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from photodiode import detect_photodiode_prctile_3state
from detect_photodiode_neuralynx import detect_photodiode_neuralynx


# -----------------------------
# Output folders
# -----------------------------
output_dir = "photodiode_outputs"
edf_dir = os.path.join(output_dir, "EDF")
ncs_dir = os.path.join(output_dir, "NCS")

for base_dir in [edf_dir, ncs_dir]:
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "states"), exist_ok=True)


# -----------------------------
# Settings
# -----------------------------
edf_opts = {
    "use_bandstop": True,
    "bandstop_low": 55.0,
    "bandstop_high": 65.0,
    "bandstop_order": 2,
}

ncs_opts = {
    "smooth_ms": 0.5,
    "low_threshold": 1.5,
    "high_threshold": 95.5,
    "refractory_ms": 0,

    "use_notch": False,
    "use_bandstop": True,
    "bandstop_low": 55.0,
    "bandstop_high": 65.0,
    "bandstop_order": 2,
}


def save_outputs(filename, file_type, detector_func, opts, output_base_dir):
    data = sio.loadmat(filename)

    signal = data["signal"].squeeze()
    fs = data["fs"].item()
    pd_state = data["pd_state"].squeeze().astype(int)

    state, on_idx, off_idx, on_time, off_time, dbg = detector_func(signal, fs, opts)
    state = state.astype(int)

    accuracy = np.mean(state == pd_state)
    mismatch_idx = np.where(state != pd_state)[0]

    base = os.path.splitext(os.path.basename(filename))[0]

    state_dir = os.path.join(output_base_dir, "states")
    plot_dir = os.path.join(output_base_dir, "plots")

    # Save arrays
    np.save(os.path.join(state_dir, base + "_detected_state.npy"), state)
    np.save(os.path.join(state_dir, base + "_pd_state.npy"), pd_state)
    np.save(os.path.join(state_dir, base + "_on_idx.npy"), on_idx)
    np.save(os.path.join(state_dir, base + "_off_idx.npy"), off_idx)
    np.save(os.path.join(state_dir, base + "_mismatch_idx.npy"), mismatch_idx)

    # Full comparison plot
    plt.figure(figsize=(14, 4))
    plt.plot(pd_state, label="pd_state reference", linewidth=1)
    plt.plot(state, label="detected state", linewidth=1, alpha=0.7)
    plt.title(f"{file_type}: {filename}\nAccuracy: {accuracy:.6f}, Mismatches: {len(mismatch_idx)}")
    plt.xlabel("Sample index")
    plt.ylabel("State")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, base + "_full_comparison.png"), dpi=150)
    plt.close()

    # Zoom around first mismatch
    if len(mismatch_idx) > 0:
        center = mismatch_idx[0]
        start = max(0, center - 1000)
        end = min(len(state), center + 1000)

        plt.figure(figsize=(14, 4))
        plt.plot(np.arange(start, end), pd_state[start:end], label="pd_state reference", linewidth=1)
        plt.plot(np.arange(start, end), state[start:end], label="detected state", linewidth=1, alpha=0.7)
        plt.title(f"{file_type}: {filename}\nZoom around first mismatch at sample {center}")
        plt.xlabel("Sample index")
        plt.ylabel("State")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, base + "_first_mismatch_zoom.png"), dpi=150)
        plt.close()

    return {
        "file": filename,
        "type": file_type,
        "accuracy": accuracy,
        "mismatches": len(mismatch_idx),
        "on_events": len(on_idx),
        "off_events": len(off_idx),
        "status": "OK",
    }


# -----------------------------
# Run EDF files
# -----------------------------
edf_files = sorted([f for f in glob.glob("*.mat") if "EDF" in f.upper()])
edf_results = []

print("\nProcessing EDF files...")
for filename in edf_files:
    print("Processing:", filename)
    try:
        result = save_outputs(
            filename=filename,
            file_type="EDF",
            detector_func=detect_photodiode_prctile_3state,
            opts=edf_opts,
            output_base_dir=edf_dir,
        )
        edf_results.append(result)
    except Exception as e:
        edf_results.append({
            "file": filename,
            "type": "EDF",
            "accuracy": None,
            "mismatches": None,
            "on_events": None,
            "off_events": None,
            "status": f"ERROR: {e}",
        })


# -----------------------------
# Run NCS files
# -----------------------------
ncs_files = sorted([f for f in glob.glob("*.mat") if "NCS" in f.upper()])
ncs_results = []

print("\nProcessing NCS files...")
for filename in ncs_files:
    print("Processing:", filename)
    try:
        result = save_outputs(
            filename=filename,
            file_type="NCS",
            detector_func=detect_photodiode_neuralynx,
            opts=ncs_opts,
            output_base_dir=ncs_dir,
        )
        ncs_results.append(result)
    except Exception as e:
        ncs_results.append({
            "file": filename,
            "type": "NCS",
            "accuracy": None,
            "mismatches": None,
            "on_events": None,
            "off_events": None,
            "status": f"ERROR: {e}",
        })


# -----------------------------
# Save CSV summaries
# -----------------------------
edf_df = pd.DataFrame(edf_results)
ncs_df = pd.DataFrame(ncs_results)
all_df = pd.concat([edf_df, ncs_df], ignore_index=True)

edf_df.to_csv(os.path.join(edf_dir, "edf_batch_results.csv"), index=False)
ncs_df.to_csv(os.path.join(ncs_dir, "ncs_batch_results.csv"), index=False)
all_df.to_csv(os.path.join(output_dir, "all_photodiode_batch_results.csv"), index=False)

# Save settings
with open(os.path.join(output_dir, "settings_used.txt"), "w") as f:
    f.write("EDF settings:\n")
    for key, value in edf_opts.items():
        f.write(f"{key}: {value}\n")

    f.write("\nNCS/Neuralynx settings:\n")
    for key, value in ncs_opts.items():
        f.write(f"{key}: {value}\n")


print("\nDone.")
print("Saved all outputs to:", output_dir)
print("\nEDF results:")
print(edf_df)
print("\nNCS results:")
print(ncs_df)