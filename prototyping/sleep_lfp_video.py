"""
Per-pid LFP sleep explorer.
Computes delta/theta/sleep-score/SWR signals from streamed LFP, caches to NPZ,
loads motion energy, and opens a 10-panel interactive figure.

Usage:
    ipython -i explore_session.py [pid]
    # or from IPython / another script:
    #   from explore_session import explore_session
    #   fig, axes = explore_session(pid)
    #   fig, axes = explore_session(pid, recompute=True)  # force recompute
    #   srt  = make_timestamp_srt(pid)       # subtitle file for mpv (2 s bins)
    #   srt  = make_wheel_srt(pid)           # per-frame wheel-speed SRT (raw resolution)
    #   path = make_annotated_video(pid)     # annotated mp4 with DLC + scores
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import signal, ndimage
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

BASE            = Path(__file__).parent
DEFAULT_PID     = "8169d556-f994-4735-b4c8-f7c85ddc39b0"
SLEEP_THRESHOLD = 0.5

# ── LFP pipeline parameters ───────────────────────────────────────────────────
LAST_N_SEC    = 600          # analyse last 10 minutes
WIN_SEC       = 4.0          # sliding window length
STEP_SEC      = 2.0          # sliding window step
DELTA         = (0.5,  4.0)  # NREM delta + slow oscillations (mouse)
THETA         = (6.0, 10.0)  # hippocampal theta (mouse)
RIPPLE        = (100,  250)  # CA1 sharp-wave ripples (Buzsáki 2015)
TOTAL         = (0.5, 50.0)  # denominator band for power ratios
N_TOP         = 5            # most-outward non-void/root channels
N_BOT         = 5            # deepest  non-void/root channels
N_CA1_PLOT    = 10           # max CA1 channels in waterfall
DECIMATE      = 10           # LFP downsample factor for plotting (2500 → 250 Hz)


# ── signal helpers ────────────────────────────────────────────────────────────

def _bandpass(x, fs, lo, hi, order=3):
    b, a = signal.butter(order, [lo / (fs/2), hi / (fs/2)], btype="band")
    return signal.filtfilt(b, a, x)


def detect_ripples(lfp_ca1, fs, thresh_on=3.0, thresh_peak=5.0,
                   min_dur=0.015, max_dur=0.150):
    """MAD z-score envelope detector for 100-250 Hz SWR events."""
    env = np.abs(signal.hilbert(_bandpass(lfp_ca1, fs, *RIPPLE)))
    med = np.median(env)
    z   = (env - med) / (1.4826 * np.median(np.abs(env - med)))
    events = []
    for i in range(1, ndimage.label(z > thresh_on)[1] + 1):
        idx = np.flatnonzero(ndimage.label(z > thresh_on)[0] == i)
        dur = len(idx) / fs
        if min_dur <= dur <= max_dur and z[idx].max() >= thresh_peak:
            events.append(idx[np.argmax(z[idx])] / fs)   # store peak time
    return np.array(events)


# ── plot helpers ──────────────────────────────────────────────────────────────

def shade_sleep(ax, t, score, color="royalblue", alpha=0.13):
    """Shade contiguous epochs where score > SLEEP_THRESHOLD."""
    asleep, in_sleep = score > SLEEP_THRESHOLD, False
    for j in range(len(t)):
        if asleep[j] and not in_sleep:
            x0 = t[j]; in_sleep = True
        if (not asleep[j] or j == len(t) - 1) and in_sleep:
            ax.axvspan(x0, t[j], color=color, alpha=alpha, zorder=1)
            in_sleep = False


def lfp_waterfall(ax, t, lfp, axial_pos, color, spacing=3.0):
    """Stack amplitude-normalised LFP traces with depth labels."""
    norm   = lfp / (np.percentile(np.abs(lfp), 90, axis=0) + 1e-12)
    single = isinstance(color, str)
    for ch in range(lfp.shape[1]):
        ax.plot(t, norm[:, ch] + ch * spacing,
                color=color if single else color[ch], lw=0.3, alpha=0.85)
    ax.set_yticks(np.arange(lfp.shape[1]) * spacing)
    ax.set_yticklabels([f"{v:.0f} µm" for v in axial_pos], fontsize=6)


def norm01(x):
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    return (x - lo) / (hi - lo + 1e-12)


# ── computation pipeline ──────────────────────────────────────────────────────

def _compute(pid, one):
    """Stream LFP + behaviour, compute all sleep signals, return savez-ready dict."""
    ssl      = SpikeSortingLoader(pid=pid, one=one)
    channels = ssl.load_channels()
    sr_lf    = ssl.raw_electrophysiology(band="lf", stream=True)
    fs       = sr_lf.fs

    eid, _   = one.pid2eid(pid)
    trials   = one.load_object(eid, "trials")

    # window starts at the last trial and runs for LAST_N_SEC;
    # cap t_end at the earliest data limit across LFP / camera / wheel,
    # then slide t_start back so the window stays LAST_N_SEC wide
    t_last_trial = np.nanmax(trials["feedback_times"])
    t_end_lfp    = sr_lf.shape[0] / fs
    try:
        cam_t_full = one.load_dataset(eid, "_ibl_leftCamera.times.npy", collection="alf")
        t_end_cam  = float(cam_t_full[-1])
    except Exception:
        t_end_cam  = t_end_lfp
    try:
        wheel_full  = one.load_object(eid, "wheel")
        t_end_wheel = float(wheel_full["timestamps"][-1])
    except Exception:
        t_end_wheel = t_end_lfp
    t_end      = min(t_end_lfp, t_end_cam, t_end_wheel)
    t_start    = t_end - LAST_N_SEC
    print(f"Window: {t_start/60:.1f}–{t_end/60:.1f} min  "
          f"(last trial at {t_last_trial/60:.1f} min, "
          f"post-trial rest in window: {(t_end - t_last_trial)/60:.1f} min)")
    lfp      = sr_lf[int(t_start * fs): int(t_end * fs), :-1].astype(np.float32)
    print(f"LFP shape: {lfp.shape}")

    # channel selection: top-N_TOP most-outward + bottom-N_BOT deepest (non-void/root)
    axial, raw_inds, acronym = channels["axial_um"], channels["rawInd"].astype(int), \
                               np.array(channels["acronym"])
    valid      = ~np.isin(acronym, ["void", "root"])
    order_desc = np.argsort(axial)[::-1]; order_desc = order_desc[valid[order_desc]]
    top_idx    = np.concatenate([order_desc[:N_TOP], order_desc[::-1][:N_BOT]])
    top_raw    = raw_inds[top_idx];  top_axial = axial[top_idx]
    print(f"Top-{N_TOP}: {np.unique(acronym[order_desc[:N_TOP]])}  "
          f"Bot-{N_BOT}: {np.unique(acronym[order_desc[::-1][:N_BOT]])}")

    # CA1 channels for ripple detection and waterfall
    ca1_mask   = np.array(["CA1" in a for a in acronym])
    ca1_raw    = raw_inds[ca1_mask]; ca1_raw = ca1_raw[ca1_raw < lfp.shape[1]]
    ca1_axial_vals = axial[ca1_mask][raw_inds[ca1_mask] < lfp.shape[1]]
    n_ca1      = len(ca1_raw)

    # vectorised Welch PSD over all windows and channels simultaneously
    win_samp   = int(WIN_SEC * fs);  step_samp = int(STEP_SEC * fs)
    n_win      = (lfp.shape[0] - win_samp) // step_samp + 1
    win_times  = t_start + (np.arange(n_win) * step_samp + win_samp // 2) / fs
    segs       = np.stack([lfp[i*step_samp: i*step_samp+win_samp, top_raw].astype(np.float64)
                           for i in range(n_win)], axis=1)   # (win_samp, n_win, N_CH)
    print(f"Computing PSD ({n_win} windows × {N_TOP+N_BOT} channels)...")
    f_ax, psd  = signal.welch(segs, fs=fs, nperseg=min(win_samp, int(fs*2)), axis=0)
    band_power = lambda band: np.trapezoid(psd[(f_ax >= band[0]) & (f_ax <= band[1])],
                                           f_ax[(f_ax >= band[0]) & (f_ax <= band[1])], axis=0)
    total      = band_power(TOTAL) + 1e-12
    delta_ratio = band_power(DELTA) / total
    theta_ratio = band_power(THETA) / total
    del segs, psd

    # behavioural signals resampled to window grid
    def to_windows(t_sig, sig):
        if t_sig is None: return np.full(n_win, np.nan)
        m = (t_sig >= t_start) & (t_sig <= t_end)
        return np.interp(win_times, t_sig[m], sig[m]) if m.sum() >= 2 else np.full(n_win, np.nan)

    def _try(fn):
        try:    return fn()
        except: return None

    cam     = _try(lambda: one.load_object(eid, "_ibl_leftCamera",
                                           attribute=["times", "features"]))
    wheel   = _try(lambda: one.load_object(eid, "wheel"))
    pupil_win = to_windows(cam["times"]  if cam   else None,
                           cam["features"]["pupilDiameter_smooth"].values if cam else None)
    wheel_win = to_windows(wheel["timestamps"] if wheel else None,
                           np.abs(np.gradient(wheel["position"], wheel["timestamps"]))
                           if wheel else None)

    # per-channel sleep score: mean of z-scored [delta, -pupil, -wheel]
    zscore    = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    pupil_z   = -zscore(pupil_win);  wheel_z = -zscore(wheel_win)
    N_CH      = N_TOP + N_BOT
    sleep_score = np.stack([np.nanmean(np.stack([zscore(delta_ratio[:, ch]),
                                                  pupil_z, wheel_z]), axis=0)
                             for ch in range(N_CH)], axis=1)

    # sharp-wave ripple detection on CA1sp (pyramidal layer) preferred
    if n_ca1 > 0:
        sp_raw = raw_inds[acronym == "CA1sp"]; sp_raw = sp_raw[sp_raw < lfp.shape[1]]
        det    = sp_raw if len(sp_raw) > 0 else ca1_raw
        print(f"Ripple detection on {'CA1sp' if len(sp_raw)>0 else 'CA1'} ({len(det)} ch)")
        peaks  = detect_ripples(lfp[:, det].mean(axis=1).astype(np.float64), fs) + t_start
        ripple_rate = np.array([np.sum((peaks >= win_times[i] - WIN_SEC/2) &
                                       (peaks <  win_times[i] + WIN_SEC/2)) / WIN_SEC
                                for i in range(n_win)])
        print(f"Detected {len(peaks)} ripples")
        ca1_plot = np.argsort(ca1_axial_vals)[::-1][:N_CA1_PLOT]
        lfp_ca1_ds = signal.decimate(lfp[:, ca1_raw[ca1_plot]].astype(np.float32),
                                     DECIMATE, axis=0, zero_phase=True)
        ca1_axial_plot = ca1_axial_vals[ca1_plot]
    else:
        ripple_rate = np.full(n_win, np.nan)
        lfp_ca1_ds  = np.zeros((0, 0), dtype=np.float32)
        ca1_axial_plot = np.array([])
        print("No CA1 channels found")

    # trial labels
    mean_score   = sleep_score.mean(axis=1)
    in_win       = (trials["stimOn_times"] >= t_start) & (trials["stimOn_times"] <= t_end)
    trial_times  = trials["stimOn_times"][in_win]
    trial_ends   = trials["feedback_times"][in_win]
    asleep_mask  = np.interp(trial_times, win_times, mean_score) > SLEEP_THRESHOLD
    print(f"Trials: {in_win.sum()}  |  asleep: {asleep_mask.sum()}")

    # downsample top/bot LFP for plotting
    lfp_ds  = signal.decimate(lfp[:, top_raw].astype(np.float32), DECIMATE, axis=0, zero_phase=True)
    fs_ds   = fs / DECIMATE
    t_ds    = np.arange(lfp_ds.shape[0]) / fs_ds + t_start

    return dict(win_times=win_times, sleep_score=sleep_score,
                delta_ratio=delta_ratio, theta_ratio=theta_ratio,
                pupil_win=pupil_win, wheel_win=wheel_win,
                trial_times=trial_times, trial_end_times=trial_ends,
                asleep_mask=asleep_mask, ripple_rate=ripple_rate,
                lfp_ds=lfp_ds, lfp_ca1_ds=lfp_ca1_ds,
                ca1_axial=ca1_axial_plot, t_ds=t_ds,
                top_axial=top_axial, fs_ds=np.array(fs_ds))


# ── main entry point ──────────────────────────────────────────────────────────

def explore_session(pid, one=None, recompute=False, save_png=False):
    """
    Compute (or load cached) sleep signals and open interactive 10-panel figure.
    Set recompute=True to ignore existing cache and re-stream LFP.
    Set save_png=True to also save the figure as a PNG next to the NPZ.
    Returns (fig, axes).
    """
    if one is None:
        one = ONE()
    eid, _ = one.pid2eid(pid)

    # compute or load from cache
    npz_path = BASE / f"sleep_score_{pid[:8]}.npz"
    if recompute or not npz_path.exists():
        print(f"Computing signals for {pid[:8]}...")
        d = _compute(pid, one)
        np.savez(npz_path, **d)
        print(f"Cached: {npz_path.name}")
    else:
        d = dict(np.load(npz_path))
        print(f"Loaded cache: {npz_path.name}")

    # unpack
    win_times, sleep_score  = d["win_times"], d["sleep_score"]
    delta_ratio, theta_ratio = d["delta_ratio"], d["theta_ratio"]
    pupil_win, wheel_win    = d["pupil_win"], d["wheel_win"]
    trial_times             = d["trial_times"]
    trial_end_times         = d["trial_end_times"]
    asleep_mask             = d["asleep_mask"]
    ripple_rate             = d["ripple_rate"]
    lfp_ds, lfp_ca1_ds      = d["lfp_ds"], d["lfp_ca1_ds"]
    ca1_axial, top_axial    = d["ca1_axial"], d["top_axial"]
    t_ds                    = d["t_ds"]

    t0          = win_times[0]
    to_min      = lambda t: (t - t0) / 60
    t_min       = to_min(win_times)
    t_ds_min    = to_min(t_ds)
    mean_score  = sleep_score.mean(axis=1)
    n_ch        = sleep_score.shape[1]
    n_ca1       = lfp_ca1_ds.shape[1] if lfp_ca1_ds.ndim == 2 else 0
    half        = n_ch // 2                         # top-5 / bot-5 boundary
    ch_color    = lambda ch: "steelblue" if ch < half else "darkorange"

    # motion energy from ONE (graceful fallback per camera)
    def load_me(cam):
        try:
            t  = one.load_dataset(eid, f"_ibl_{cam}Camera.times.npy",     collection="alf")
            me = one.load_dataset(eid, f"{cam}Camera.ROIMotionEnergy.npy", collection="alf")
            return t, (me[:, 0] if me.ndim > 1 else me).astype(np.float64)
        except Exception as e:
            print(f"  {cam}Camera ME unavailable: {e}"); return None, None

    print("Loading motion energy...")
    t_l, me_l = load_me("left")
    t_r, me_r = load_me("right")
    t_b, me_b = load_me("body")

    t_lo = win_times[0] - (win_times[1] - win_times[0]) / 2
    t_hi = win_times[-1] + (win_times[1] - win_times[0]) / 2
    def resample(t_sig, sig):
        if t_sig is None: return np.full(len(win_times), np.nan)
        m = (t_sig >= t_lo) & (t_sig <= t_hi)
        return np.interp(win_times, t_sig[m], sig[m]) if m.sum() >= 2 \
               else np.full(len(win_times), np.nan)

    me_left  = norm01(resample(t_l, me_l))
    me_right = norm01(resample(t_r, me_r))
    me_body  = norm01(resample(t_b, me_b))

    # ── figure ────────────────────────────────────────────────────────────────
    ca1_h   = max(1.0, min(n_ca1, 10) * 0.25 + 0.5)
    heights = [0.35, 0.9, 0.9, 1.2, 1.8, 1.8, 1.8, 0.9, ca1_h, 2.5]

    plt.ion()
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    try: fig.canvas.manager.window.showMaximized()
    except Exception: pass

    gs   = gridspec.GridSpec(10, 1, hspace=0.06, height_ratios=heights,
                             left=0.06, right=0.99, top=0.98, bottom=0.08)
    ax0  = fig.add_subplot(gs[0])
    axes = [ax0] + [fig.add_subplot(gs[i], sharex=ax0) for i in range(1, 10)]

    # 0 — trial epochs (red = asleep, grey = awake)
    for ts, te, sl in zip(to_min(trial_times), to_min(trial_end_times), asleep_mask):
        axes[0].axvspan(ts, te, color="tomato" if sl else "silver", alpha=0.85)
    axes[0].set_yticks([]); axes[0].set_ylabel("Trials", fontsize=8)

    # 1 — wheel speed
    axes[1].plot(t_min, wheel_win, color="saddlebrown", lw=0.8)
    shade_sleep(axes[1], t_min, mean_score); axes[1].set_ylabel("Wheel\nspeed", fontsize=8)

    # 2 — pupil diameter
    axes[2].plot(t_min, pupil_win, color="purple", lw=0.8)
    shade_sleep(axes[2], t_min, mean_score); axes[2].set_ylabel("Pupil\ndiameter", fontsize=8)

    # 3 — motion energy (left/right whisker + body, normalised 0-1)
    ax = axes[3]
    ax.plot(t_min, me_left,  color="steelblue", lw=0.8, alpha=0.85, label="left whisker")
    ax.plot(t_min, me_right, color="darkorange", lw=0.8, alpha=0.85, label="right whisker")
    ax.plot(t_min, me_body,  color="seagreen",   lw=0.8, alpha=0.85, label="body")
    shade_sleep(ax, t_min, mean_score)
    ax.set_ylabel("Motion\nenergy", fontsize=8); ax.legend(fontsize=6, loc="upper right", ncol=3)

    # 4-6 — sleep score / delta / theta  (steelblue = top-5 channels, darkorange = bot-5)
    for panel, data, ylabel in [(4, sleep_score, "Sleep score"),
                                 (5, delta_ratio, "Delta ratio\n(0.5–4 Hz)"),
                                 (6, theta_ratio, "Theta ratio\n(6–10 Hz)")]:
        ax = axes[panel]
        for ch in range(n_ch):
            ax.plot(t_min, data[:, ch], color=ch_color(ch), lw=0.6, alpha=0.7)
        if panel == 4:
            ax.axhline(SLEEP_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.5,
                       label=f"thr={SLEEP_THRESHOLD}")
            ax.legend(fontsize=6, loc="upper right")
        shade_sleep(ax, t_min, mean_score); ax.set_ylabel(ylabel, fontsize=8)

    # 7 — SWR rate (CA1sp preferred, 100-250 Hz, MAD z-score)
    ax = axes[7]
    if not np.all(np.isnan(ripple_rate)):
        ax.fill_between(t_min, ripple_rate, color="darkorange", alpha=0.7, lw=0)
        ax.set_ylabel("SWR rate\n(evt/s)", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No CA1 channels", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        ax.set_yticks([]); ax.set_ylabel("SWR rate", fontsize=8)
    shade_sleep(ax, t_min, mean_score)

    ms_ds = np.interp(t_ds, win_times, mean_score)   # mean score on LFP time grid

    # 8 — CA1 LFP waterfall (up to N_CA1_PLOT topmost CA1 channels)
    ax = axes[8]
    if n_ca1 > 0:
        shade_sleep(ax, t_ds_min, ms_ds)
        lfp_waterfall(ax, t_ds_min, lfp_ca1_ds, ca1_axial, "darkorange")
        ax.set_ylabel("CA1 LFP\n(axial pos.)", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No CA1 channels", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        ax.set_yticks([]); ax.set_ylabel("CA1 LFP", fontsize=8)

    # 9 — top-5 + bottom-5 LFP waterfall with gap and dashed separator
    ax = axes[9]
    shade_sleep(ax, t_ds_min, ms_ds)
    sp   = 3.0
    norm = lfp_ds / (np.percentile(np.abs(lfp_ds), 90, axis=0) + 1e-12)
    ticks, labels = [], []
    for ch in range(n_ch):
        offset = (ch + 1 if ch >= half else ch) * sp   # +1 slot gap between groups
        ax.plot(t_ds_min, norm[:, ch] + offset, color=ch_color(ch), lw=0.3, alpha=0.85)
        ticks.append(offset); labels.append(f"{top_axial[ch]:.0f} µm")
    ax.axhline(half * sp, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylabel("Top/bot 5 LFP\n(axial pos.)", fontsize=8)
    ax.set_xlabel("Time in window (min)", fontsize=9)

    # secondary x-axis: absolute session MM:SS — same clock as video
    secax = ax.secondary_xaxis("bottom",
                                functions=(lambda x: x*60 + t0, lambda s: (s - t0)/60))
    secax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda s, _: f"{int(s)//60:02d}:{int(s)%60:02d}"))
    secax.tick_params(labelsize=7); secax.set_xlabel("Session time (MM:SS)", fontsize=9)

    # shared formatting
    n_asleep = int(asleep_mask.sum()); n_trials = len(trial_times)
    fig.suptitle(f"pid={pid}  |  last 10 min  |  trials: {n_trials}  |  "
                 f"asleep: {n_asleep} ({100*n_asleep/max(n_trials,1):.0f}%)", fontsize=9)
    ax0.set_xlim(t_min[0], t_min[-1])
    for i, ax in enumerate(axes):
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if i < len(axes) - 1:
            ax.set_xticklabels([])

    if save_png:
        png_path = BASE / f"sleep_score_{pid[:8]}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved PNG: {png_path}")

    plt.show()
    print(f"Interactive figure open — pid={pid[:8]}  |  use returned (fig, axes) to inspect.")
    return fig, axes


# ── video annotation helpers ──────────────────────────────────────────────────

def make_timestamp_srt(pid, one=None):
    """
    Write an SRT subtitle file with session MM:SS and behavioural scores per frame.
    VLC auto-loads it if the .srt sits next to the .mp4 with the same stem.
    Scores are interpolated from the pre-computed NPZ (run explore_session first).
    """
    if one is None:
        one = ONE()
    eid, _ = one.pid2eid(pid)
    cam_t  = one.load_dataset(eid, "_ibl_leftCamera.times.npy", collection="alf")
    video  = one.eid2path(eid) / "raw_video_data" / "_iblrig_leftCamera.raw.mp4"
    srt    = video.with_suffix(".srt")
    # video playback time = session time relative to first frame
    # using cam_t directly avoids drift from assuming constant fps
    t_vid  = cam_t - cam_t[0]
    dt     = np.median(np.diff(cam_t[:2000]))   # nominal frame duration

    # load scores if NPZ exists, otherwise show timestamp only
    npz_path = BASE / f"sleep_score_{pid[:8]}.npz"
    signals  = {}
    if npz_path.exists():
        d = dict(np.load(npz_path))
        wt = d["win_times"]
        signals = {
            "sleep" : d["sleep_score"].mean(axis=1),
            "delta" : d["delta_ratio"].mean(axis=1),
            "theta" : d["theta_ratio"].mean(axis=1),
            "SWR/s" : d["ripple_rate"],
            "wheel" : d["wheel_win"],
            "pupil" : d["pupil_win"],
        }

    def srt_ts(sec):
        h, r = divmod(sec, 3600); m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(s % 1 * 1000):03d}"

    with open(srt, "w") as f:
        for i, t in enumerate(cam_t):
            t_start_vid = t_vid[i]
            t_end_vid   = t_vid[i + 1] if i + 1 < len(cam_t) else t_vid[i] + dt
            ts_line = f"{int(t)//60:02d}:{int(t)%60:02d}.{int(t % 1 * 10)}"
            if signals:
                scores = "  ".join(f"{k}:{float(np.interp(t, wt, v)):.2f}"
                                   for k, v in signals.items())
                text = f"{ts_line}  |  {scores}"
            else:
                text = ts_line
            f.write(f"{i+1}\n{srt_ts(t_start_vid)} --> {srt_ts(t_end_vid)}\n{text}\n\n")

    print(f"Saved: {srt}\nOpen {video.name} in VLC — subtitles auto-load.")
    return srt


def make_wheel_srt(pid, one=None, cam="left"):
    """
    Write a per-frame SRT subtitle synced to the side camera, showing wheel speed
    interpolated from raw wheel timestamps — no 2-second binning artefact.

    Output: <video_dir>/_iblrig_{cam}Camera.raw.srt
    Play:   mpv _iblrig_leftCamera.raw.mp4 --sub-file=_iblrig_leftCamera.raw.srt
    VLC auto-loads it if the .srt sits next to the .mp4 with the same stem.
    """
    if one is None:
        one = ONE()
    eid, _ = one.pid2eid(pid)

    cam_t = one.load_dataset(eid, f"_ibl_{cam}Camera.times.npy", collection="alf")
    wheel = one.load_object(eid, "wheel")
    speed = np.abs(np.gradient(wheel["position"], wheel["timestamps"]))
    wheel_at_frame = np.interp(cam_t, wheel["timestamps"], speed)

    video = one.eid2path(eid) / "raw_video_data" / f"_iblrig_{cam}Camera.raw.mp4"
    srt   = video.with_suffix(".srt")
    t_vid = cam_t - cam_t[0]
    dt    = np.median(np.diff(cam_t[:2000]))

    def srt_ts(sec):
        h, r = divmod(sec, 3600); m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(s % 1 * 1000):03d}"

    with open(srt, "w") as f:
        for i, t in enumerate(cam_t):
            t0  = t_vid[i]
            t1  = t_vid[i + 1] if i + 1 < len(cam_t) else t_vid[i] + dt
            ts  = f"{int(t)//60:02d}:{int(t)%60:02d}.{int(t % 1 * 10)}"
            spd = wheel_at_frame[i]
            f.write(f"{i+1}\n{srt_ts(t0)} --> {srt_ts(t1)}\n{ts}  wheel: {spd:.3f} rad/s\n\n")

    print(f"Saved: {srt}")
    return srt


def make_annotated_video(pid, one=None, cam="left", likelihood_thr=0.9):
    """
    Write annotated mp4 (analysis window only) with:
      - DLC body-part markers: filled cv2 circles + black outline, per-point colour
      - DLC point legend on the right side (colour-matched, DLC_labeled_video style)
      - Per-frame signal panel (bottom-left, semi-transparent):
          session time, frame number,
          wheel speed at 1 kHz resolution,
          delta ratio, theta ratio, SWR/s, pupil diameter,
          motion energy left/right/body cameras,
          composite sleep score
    Requires opencv-python and brainbox.
    Output: BASE/annotated_{pid8}.mp4
    """
    import cv2
    import brainbox.behavior.wheel as wh

    if one is None:
        one = ONE()
    eid, _ = one.pid2eid(pid)

    cam_t = one.load_dataset(eid, f"_ibl_{cam}Camera.times.npy", collection="alf")

    # wheel interpolated to 1 kHz — smooth per-frame speed (brainbox style)
    wheel = one.load_object(eid, "wheel")
    try:
        pos_1k, t_1k = wh.interpolate_position(wheel["timestamps"], wheel["position"], freq=1000)
    except Exception:
        pos_1k, t_1k = wh.interpolate_position(wheel["times"], wheel["position"], freq=1000)
    speed_1k = np.abs(np.gradient(pos_1k, t_1k))   # rad/s

    # motion energy — three cameras, raw signal
    def _load_me(c):
        try:
            t  = one.load_dataset(eid, f"_ibl_{c}Camera.times.npy", collection="alf")
            me = one.load_dataset(eid, f"{c}Camera.ROIMotionEnergy.npy", collection="alf")
            return t, (me[:, 0] if me.ndim > 1 else me).astype(np.float64)
        except Exception as e:
            print(f"  {c}Camera ME unavailable: {e}"); return None, None

    t_mel, me_l = _load_me("left")
    t_mer, me_r = _load_me("right")
    t_meb, me_b = _load_me("body")

    # DLC — original implementation: simple colour palette, likelihood checked per frame
    dlc_arrays, bp_colors, parts = {}, {}, []
    try:
        dlc   = one.load_dataset(eid, f"_ibl_{cam}Camera.dlc.pqt", collection="alf")
        parts = sorted({c.rsplit("_", 1)[0] for c in dlc.columns if c.endswith("_x")})
        if cam != "body":
            parts = [p for p in parts if p not in ("tube_top", "tube_bottom")]
        pal   = [(220, 80,  80), (80, 220,  80), ( 80,  80, 220),
                 (220, 220, 80), (220, 80, 220), ( 80, 220, 220),
                 (180, 120, 60), (60, 180, 120), (120,  60, 180), (200, 200, 200)]
        bp_colors = {p: pal[i % len(pal)] for i, p in enumerate(parts)}
        for bp in parts:
            dlc_arrays[bp] = {
                "x":  dlc[f"{bp}_x"].values          if f"{bp}_x"  in dlc.columns else None,
                "y":  dlc[f"{bp}_y"].values          if f"{bp}_y"  in dlc.columns else None,
                "lk": dlc[f"{bp}_likelihood"].values if f"{bp}_likelihood" in dlc.columns else None,
            }
        print(f"DLC body parts ({len(parts)}): {parts}")
    except Exception as e:
        print(f"DLC unavailable: {e}")

    # NPZ signals
    npz_path = BASE / f"sleep_score_{pid[:8]}.npz"
    if not npz_path.exists():
        raise FileNotFoundError("Run explore_session() first to generate NPZ.")
    d = dict(np.load(npz_path))
    win_times = d["win_times"]
    lf_sigs = {
        "sleep" : d["sleep_score"].mean(axis=1),
        "delta" : d["delta_ratio"].mean(axis=1),
        "theta" : d["theta_ratio"].mean(axis=1),
        "SWR/s" : d["ripple_rate"],
        "pupil" : d["pupil_win"],
    }

    fi_all = np.where((cam_t >= win_times[0]) & (cam_t <= win_times[-1]))[0]
    if len(fi_all) == 0:
        raise ValueError("No camera frames found in the analysis window.")

    video_path = one.eid2path(eid) / "raw_video_data" / f"_iblrig_{cam}Camera.raw.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = BASE / f"annotated_{pid[:8]}.mp4"
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # font settings matching DLC_labeled_video.py conventions
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 4 if cam == "left" else 2
    lineType  = 2
    dot_s     = 10 if cam == "left" else 5

    # anchor point (bottom-left), mirroring DLC_labeled_video.py
    a, b = 20, H - 30           # a=left margin, b=bottom y
    lh   = fontScale // 2 * 35  # line height in pixels

    print(f"Writing {len(fi_all)} frames → {out_path.name} ...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi_all[0]))
    cur_pos = int(fi_all[0])

    for step, fi in enumerate(fi_all):
        fi = int(fi)
        if fi != cur_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        cur_pos = fi + 1
        if not ret:
            break

        t = float(cam_t[fi])

        # ── DLC: filled circle + black outline (original implementation) ──────
        for bp, color in bp_colors.items():
            arr = dlc_arrays[bp]
            if arr["x"] is None or fi >= len(arr["x"]):
                continue
            lk = float(arr["lk"][fi]) if arr["lk"] is not None else 1.0
            x, y = float(arr["x"][fi]), float(arr["y"][fi])
            if lk >= likelihood_thr and not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (int(x), int(y)), dot_s // 2, color, -1)
                cv2.circle(frame, (int(x), int(y)), dot_s // 2 + 1, (0, 0, 0), 1)

        # ── DLC legend: right side, coloured labels (DLC_labeled_video.py style)
        for ll, bp in enumerate(parts):
            col = bp_colors[bp]
            cv2.putText(frame, bp,
                        (b, a * 2 * (1 + ll)),   # x=b, y steps down from top
                        font, fontScale / 4, col, lineType)

        # ── per-frame signals ─────────────────────────────────────────────────
        wi          = min(int(np.searchsorted(t_1k, t)), len(t_1k) - 1)
        wheel_speed = float(speed_1k[wi])

        def _me(t_me, me):
            if t_me is None: return float("nan")
            return float(me[min(int(np.searchsorted(t_me, t)), len(t_me) - 1)])

        def _lf(key):
            return float(np.interp(t, win_times, lf_sigs[key]))

        ts = f"{int(t)//60:02d}:{int(t)%60:02d}.{int(t % 1 * 10)}"

        # lines ordered bottom → top (last entry appears at the very bottom)
        signal_lines = [
            ("sleep score", f"{_lf('sleep'):+.3f}"),
            ("ME body    ", f"{_me(t_meb, me_b):.1f}"),
            ("ME right   ", f"{_me(t_mer, me_r):.1f}"),
            ("ME left    ", f"{_me(t_mel, me_l):.1f}"),
            ("pupil dia  ", f"{_lf('pupil'):.1f} px"),
            ("SWR/s      ", f"{_lf('SWR/s'):.3f}"),
            ("theta      ", f"{_lf('theta'):.3f}"),
            ("delta      ", f"{_lf('delta'):.3f}"),
            ("wheel spd  ", f"{wheel_speed:.3f} r/s"),
            ("time", f"{ts}   frame {fi}"),
        ]

        txt_scale = fontScale / 20 * 3   # readable: 10× smaller than original, ×3 back up
        txt_lh    = int(txt_scale * 35) + 4
        n         = len(signal_lines)
        pw        = int(txt_scale * 310) + 60
        py0       = b - n * txt_lh
        overlay   = frame.copy()
        cv2.rectangle(overlay, (0, py0 - txt_lh // 2), (pw, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for li, (label, val) in enumerate(signal_lines):
            y_pos = b - (n - 1 - li) * txt_lh
            cv2.putText(frame, f"{label}: {val}",
                        (a, y_pos), font, txt_scale, (255, 255, 255), 1)

        writer.write(frame)
        if step % 900 == 0:
            print(f"  {step}/{len(fi_all)} ({100*step//len(fi_all)}%)")

    cap.release()
    writer.release()
    print(f"Done → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _pid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PID
    fig, axes = explore_session(_pid)
