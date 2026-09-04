import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d


def set_defaults(opts=None):
    if opts is None:
        opts = {}

    defaults = {
        "smooth_ms": 4,
        "high_threshold": 90,
        "low_threshold": 10,

        # Prevent fake rapid switches
        "refractory_ms": 1,

        "use_notch": False,
        "notch_freq": 60.0,
        "notch_Q": 30.0,

        "use_bandstop": True,
        "bandstop_low": 55.0,
        "bandstop_high": 65.0,
        "bandstop_order": 2,
    }

    for key, value in defaults.items():
        if key not in opts or opts[key] is None:
            opts[key] = value

    return opts


def detect_photodiode_neuralynx(pd, fs, opts=None):
    opts = set_defaults(opts)

    pd = np.asarray(pd).reshape(-1)
    T = pd.size
    x = pd.astype(float)

    # -----------------------------
    # Filtering
    # -----------------------------
    if opts["use_notch"]:
        b, a = iirnotch(
            w0=opts["notch_freq"],
            Q=opts["notch_Q"],
            fs=fs,
        )
        x = filtfilt(b, a, x)

    elif opts["use_bandstop"]:
        nyquist = fs / 2
        low = opts["bandstop_low"] / nyquist
        high = opts["bandstop_high"] / nyquist

        b, a = butter(
            opts["bandstop_order"],
            [low, high],
            btype="bandstop",
        )

        x = filtfilt(b, a, x)

    # -----------------------------
    # Smoothing
    # -----------------------------
    smoothN = max(0, round(opts["smooth_ms"] * 1e-3 * fs))

    if smoothN >= 3:
        x = uniform_filter1d(x, size=smoothN, mode="nearest")

    # -----------------------------
    # Thresholds
    # -----------------------------
    high_p = max(0, min(100, round(opts["high_threshold"])))
    low_p = max(0, min(100, round(opts["low_threshold"])))

    upth = np.percentile(x, high_p)
    loth = np.percentile(x, low_p)

   # print(f"low_threshold: {loth:.6f},\thigh_threshold: {upth:.6f}")

    # -----------------------------
    # Detection
    # -----------------------------

    refractoryN = max(1, round(opts["refractory_ms"] * 1e-3 * fs))

    state = np.zeros(T, dtype=float)
    on_idx = []
    off_idx = []

    first_detection = True
    prev_state = 0
    start_idx = 0
    last_transition = -refractoryN

    for t in range(T):
        can_switch = (t - last_transition) >= refractoryN

        if not can_switch:
            continue 

        # High threshold means ON
        if x[t] >= upth:
            if first_detection:
                state[start_idx:t] = 0
                state[t] = 1

                on_idx.append(t)
                prev_state = 1
                start_idx = t + 1
                last_transition = t
                first_detection = False

            elif prev_state == 0:
                state[start_idx:t] = 0
                state[t] = 1

                on_idx.append(t)
                prev_state = 1
                start_idx = t + 1
                last_transition = t

        # Low threshold means OFF
        elif x[t] <= loth:
            if first_detection:
                state[start_idx:t] = 1
                state[t] = 0

                off_idx.append(t)
                prev_state = 0
                start_idx = t + 1
                last_transition = t
                first_detection = False

            elif prev_state == 1:
                state[start_idx:t] = 1
                state[t] = 0

                off_idx.append(t)
                prev_state = 0
                start_idx = t + 1
                last_transition = t

    state[start_idx:] = prev_state

    on_idx = np.array(on_idx, dtype=int)
    off_idx = np.array(off_idx, dtype=int)

    on_time = on_idx / fs
    off_time = off_idx / fs

    dbg = {
        "thresholds": [loth, upth],
        "filtering": {
            "use_notch": opts["use_notch"],
            "notch_freq": opts["notch_freq"],
            "notch_Q": opts["notch_Q"],
            "use_bandstop": opts["use_bandstop"],
            "bandstop_low": opts["bandstop_low"],
            "bandstop_high": opts["bandstop_high"],
            "bandstop_order": opts["bandstop_order"],
        },
        "smoothing": {
            "smooth_ms": opts["smooth_ms"],
            "smoothN": smoothN,
        },
        "refractory": {
            "refractory_ms": opts["refractory_ms"],
            "refractoryN": refractoryN,
        },
        "onIdx": on_idx,
        "offIdx": off_idx,
    }

    return state, on_idx, off_idx, on_time, off_time, dbg