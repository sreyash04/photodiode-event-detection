import scipy.io as sio
import numpy as np
from detect_photodiode_neuralynx import detect_photodiode_neuralynx

data = sio.loadmat("photodiode_events_DA030_Gambling_NCS.mat")

signal = data["signal"].squeeze()
fs = data["fs"].item()
pd_state = data["pd_state"].squeeze()

configs = []

for low in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]:
    high = 100 - low
    configs.append((low, high))

best = None

for low, high in configs:
    opts = {
        "low_threshold": low,
        "high_threshold": high,
        "use_bandstop": True,
        "bandstop_low": 55.0,
        "bandstop_high": 65.0,
        "bandstop_order": 2,
        "use_notch": False,
    }

    state, on_idx, off_idx, on_time, off_time, dbg = detect_photodiode_neuralynx(signal, fs, opts)

    acc = np.mean(state == pd_state)

    print(
        f"low={low:2d}, high={high:2d}, "
        f"accuracy={acc:.6f}, "
        f"on={len(on_idx)}, off={len(off_idx)}"
    )

    if best is None or acc > best[0]:
        best = (acc, low, high, len(on_idx), len(off_idx))

print("\nBEST:")
print(f"accuracy={best[0]:.6f}, low={best[1]}, high={best[2]}, on={best[3]}, off={best[4]}")