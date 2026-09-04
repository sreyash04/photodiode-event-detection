import glob
import scipy.io as sio
import numpy as np
from photodiode import detect_photodiode_prctile_3state

opts = {
    "use_bandstop": True,
    "bandstop_low": 55.0,
    "bandstop_high": 65.0,
    "bandstop_order": 2,
}

files = sorted(glob.glob("*EDF*.mat"))

print("file,accuracy,mismatches,on_events,off_events,status")

for filename in files:
    try:
        data = sio.loadmat(filename)

        signal = data["signal"].squeeze()
        fs = data["fs"].item()
        pd_state = data["pd_state"].squeeze().astype(int)

        state, on_idx, off_idx, on_time, off_time, dbg = detect_photodiode_prctile_3state(signal, fs, opts)
        state = state.astype(int)

        accuracy = np.mean(state == pd_state)
        mismatches = np.sum(state != pd_state)

        print(f"{filename},{accuracy:.12f},{mismatches},{len(on_idx)},{len(off_idx)},OK")

    except Exception as e:
        print(f"{filename},NA,NA,NA,NA,ERROR: {e}")