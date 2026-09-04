import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, iirnotch


def set_defaults(opts=None):
    if opts is None:
        opts = {}

    defaults = {
        "smooth_ms": 4,
        "minMID_ms": 1,
        "debounceON_ms": 100,
        "debounceOFF_ms": 100,
        "refractory_ms": 100,
        "zscoreBeforeFit": True,
        "extreme_prctiles": [2, 98],
        "threshold_fraction": 5,
        "rngSeed": 0,

        # Optional notch filter for power-line noise
        "use_notch": False,
        "notch_freq": 60.0,
        "notch_Q": 30.0,

        # Optional Butterworth bandstop filter
        "use_bandstop": False,
        "bandstop_low": 55.0,
        "bandstop_high": 65.0,
        "bandstop_order": 2,
    }

    for key, value in defaults.items():
        if key not in opts or opts[key] is None:
            opts[key] = value

    return opts


def detect_photodiode_prctile_3state(pd, fs, opts=None):
    opts = set_defaults(opts)

    pd = np.asarray(pd).reshape(-1)
    T = pd.size

    smoothN = max(0, round(opts["smooth_ms"] * 1e-3 * fs))
    minMID_N = max(1, round(opts["minMID_ms"] * 1e-3 * fs))
    debON_N = max(1, round(opts["debounceON_ms"] * 1e-3 * fs))
    debOFF_N = max(1, round(opts["debounceOFF_ms"] * 1e-3 * fs))
    refr_N = max(0, round(opts["refractory_ms"] * 1e-3 * fs))

    x = pd.astype(float)

    # Optional frequency filtering.
    # Default is no frequency filtering.
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

    # Default smoothing: uniform moving average.
    if smoothN >= 3:
        x_s = uniform_filter1d(x, size=smoothN, mode="nearest")
    else:
        x_s = x.copy()

    if opts["zscoreBeforeFit"]:
        mu0 = np.mean(x_s)
        sd0 = np.std(x_s)
        if sd0 < np.finfo(float).eps:
            sd0 = 1
        x_fit = (x_s - mu0) / sd0
    else:
        mu0 = 0
        sd0 = 1
        x_fit = x_s.copy()

    low_prc = np.percentile(x_fit, opts["extreme_prctiles"][0])
    high_prc = np.percentile(x_fit, opts["extreme_prctiles"][1])

    iprci = high_prc - low_prc

    high_thr = high_prc - iprci / opts["threshold_fraction"]
    low_thr = low_prc + iprci / opts["threshold_fraction"]

    raw_label = np.zeros_like(x_fit, dtype=np.int8)
    raw_label[(x_fit >= low_thr) & (x_fit <= high_thr)] = 1
    raw_label[x_fit > high_thr] = 2

    raw_state = raw_label.astype(np.int8)

    dbg = {
        "x_s": x_s,
        "x_fit": x_fit,
        "mu0": mu0,
        "sd0": sd0,
        "rawLabel": raw_label,
        "rawState": raw_state,
        "thresholds": [low_thr, high_thr],
        "samples": {
            "smoothN": smoothN,
            "minMID_N": minMID_N,
            "debON_N": debON_N,
            "debOFF_N": debOFF_N,
            "refr_N": refr_N,
        },
        "filtering": {
            "use_notch": opts["use_notch"],
            "notch_freq": opts["notch_freq"],
            "notch_Q": opts["notch_Q"],
            "use_bandstop": opts["use_bandstop"],
            "bandstop_low": opts["bandstop_low"],
            "bandstop_high": opts["bandstop_high"],
            "bandstop_order": opts["bandstop_order"],
        },
    }

    state = np.zeros(T, dtype=np.int8)

    if raw_state[0] == 2:
        state[0] = 1
    else:
        state[0] = 0

    on_idx = []
    off_idx = []

    last_on_det = -np.inf
    last_off_det = -np.inf

    mid_active = False
    mid_start = None
    mid_len = 0

    on_cand_active = False
    on_cand_start = None
    on_cand_len = 0

    off_cand_active = False
    off_cand_start = None
    off_cand_len = 0

    on_to_mid_start = None
    on_to_mid_len = 0

    for t in range(1, T):
        rs = raw_state[t]

        state[t] = state[t - 1]

        if state[t - 1] == 0:
            off_cand_active = False
            off_cand_len = 0
            off_cand_start = None

            if rs == 1:
                if not mid_active:
                    mid_active = True
                    mid_start = t
                    mid_len = 1
                else:
                    mid_len += 1

                on_cand_active = False
                on_cand_len = 0
                on_cand_start = None

            elif rs == 2:
                if mid_active and mid_len >= minMID_N:
                    if not on_cand_active:
                        on_cand_active = True
                        on_cand_start = mid_start
                        on_cand_len = 1
                    else:
                        on_cand_len += 1

                    if on_cand_len >= debON_N:
                        onset = on_cand_start

                        if (onset - last_on_det) >= refr_N:
                            state[onset:t + 1] = 1
                            on_idx.append(onset)
                            last_on_det = onset

                        mid_active = False
                        mid_start = None
                        mid_len = 0
                        on_cand_active = False
                        on_cand_start = None
                        on_cand_len = 0

                else:
                    on_cand_active = False
                    on_cand_start = None
                    on_cand_len = 0
                    mid_active = False
                    mid_start = None
                    mid_len = 0

            else:
                mid_active = False
                mid_start = None
                mid_len = 0
                on_cand_active = False
                on_cand_start = None
                on_cand_len = 0

            on_to_mid_start = None
            on_to_mid_len = 0

        else:
            mid_active = False
            mid_start = None
            mid_len = 0
            on_cand_active = False
            on_cand_start = None
            on_cand_len = 0

            if rs == 2:
                off_cand_active = False
                off_cand_len = 0
                off_cand_start = None

                on_to_mid_start = None
                on_to_mid_len = 0

            elif rs == 1:
                if on_to_mid_start is None:
                    on_to_mid_start = t
                    on_to_mid_len = 1
                else:
                    on_to_mid_len += 1

            else:
                if not off_cand_active:
                    off_cand_active = True

                    if on_to_mid_start is not None and on_to_mid_len >= minMID_N:
                        off_cand_start = on_to_mid_start
                    else:
                        off_cand_start = t

                    off_cand_len = 1
                else:
                    off_cand_len += 1

                if off_cand_len >= debOFF_N:
                    off_start = off_cand_start

                    state[off_start:t + 1] = 0
                    off_idx.append(off_start)

                    last_off_det = off_start

                    off_cand_active = False
                    off_cand_len = 0
                    off_cand_start = None

                    on_to_mid_start = None
                    on_to_mid_len = 0

    on_idx = np.array(on_idx, dtype=int)
    off_idx = np.array(off_idx, dtype=int)

    on_time = on_idx / fs
    off_time = off_idx / fs

    dbg["onIdx"] = on_idx
    dbg["offIdx"] = off_idx

    return state, on_idx, off_idx, on_time, off_time, dbg