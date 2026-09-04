import scipy.io as sio
import numpy as np
from photodiode import detect_photodiode_prctile_3state

data = sio.loadmat("photodiode_events_DA004_DA4_SWB_deidentified_EDF.mat")

signal = data["signal"].squeeze()
fs = data["fs"].item()
pd_state = data["pd_state"].squeeze()

# Optional bandstop filter settings.
# This removes frequencies between 55 Hz and 65 Hz.
# Set use_bandstop to False to use only default uniform smoothing.
opts = {
    "use_bandstop": True,
    "bandstop_low": 55.0,
    "bandstop_high": 65.0,
    "bandstop_order": 2,
}

state, on_idx, off_idx, on_time, off_time, dbg = detect_photodiode_prctile_3state(signal, fs, opts)

print("state unique:", np.unique(state))
print("on events:", len(on_idx))
print("off events:", len(off_idx))

accuracy = np.mean(state == pd_state)
print("accuracy vs pd_state:", accuracy)

print("first 10 on_idx:", on_idx[:10])
print("first 10 off_idx:", off_idx[:10])

mismatches = np.where(state != pd_state)[0]
print("number of mismatches:", len(mismatches))

if len(mismatches) > 0:
    print("first 10 mismatches:", mismatches[:10])
else:
    print("Python output perfectly matches reference pd_state.")

print("filtering used:", dbg["filtering"])