import scipy.io as sio
import numpy as np
from detect_photodiode_neuralynx import detect_photodiode_neuralynx

data = sio.loadmat("photodiode_events_DA012_Gambling_NCS.mat")

signal = data["signal"].squeeze()
fs = data["fs"].item()
pd_state = data["pd_state"].squeeze().astype(int)

opts = {
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

state, on_idx, off_idx, on_time, off_time, dbg = detect_photodiode_neuralynx(signal, fs, opts)
state = state.astype(int)

accuracy = np.mean(state == pd_state)
mismatches = np.sum(state != pd_state)

print("FINAL RESULT")
print("file: photodiode_events_DA030_Gambling_NCS.mat")
print("accuracy:", accuracy)
print("mismatches:", mismatches)
print("on events:", len(on_idx))
print("off events:", len(off_idx))
print("thresholds:", dbg["thresholds"])
print("options:", opts)