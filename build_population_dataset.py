# -*- coding: utf-8 -*-
"""
Build population-based SHM dataset for current folder structure.

Folder structure:
01_models/
    model_0001/
        params.json
        modal_summary.json
        pushover_summary.json
        model_info.pkl

03_simulation_results/
    model_0001/
        healthy_ref/
            acc_ground.txt
            acc_roof.txt
            acc_story_1.txt ...
            run_summary.json
        gm_0001_pga_0p20g/
            eq_summary.json
            acc_ground.txt
            acc_roof.txt
            acc_story_1.txt ...
            posteq_white_noise/
                acc_ground.txt
                acc_roof.txt
                acc_story_1.txt ...
                run_summary.json
"""

from __future__ import annotations

import json
import traceback
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal


# =========================================================
# Basic I/O
# =========================================================
def load_two_col(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"Expect 2-column txt file: {path}")
    return a[:, 0].astype(float), a[:, 1].astype(float)


def align_truncate(*arrs):
    n = min(len(a) for a in arrs)
    return [a[:n] for a in arrs]


def dt_from_time(t: np.ndarray) -> float:
    return float(np.mean(np.diff(t)))


def safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Noise helper
# =========================================================
def add_noise(signal_in: np.ndarray, snr_db: Optional[float], rng: np.random.Generator):
    if snr_db is None:
        return signal_in
    sig_power = np.mean(signal_in ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_linear + 1e-12)
    noise = rng.normal(0.0, np.sqrt(noise_power), len(signal_in))
    return signal_in + noise


# =========================================================
# Load accelerations from txt
# =========================================================
def load_abs_accels_txt(out_dir: Path, n_story: int, snr_db=None, seed=123):
    tg, ag = load_two_col(out_dir / "acc_ground.txt")

    floor_abs = {}
    for k in range(1, n_story + 1):
        fp = out_dir / f"acc_story_{k}.txt"
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        tf, af = load_two_col(fp)
        tg2, tf2 = align_truncate(tg, tf)
        ag2, af2 = align_truncate(ag, af)
        tg, ag = tg2, ag2
        floor_abs[k] = af2 + ag

    dt = dt_from_time(tg)

    if snr_db is None:
        return tg, ag, floor_abs, dt

    master = np.random.default_rng(seed)
    rng_g = np.random.default_rng(master.integers(0, 2**32 - 1))
    ag_noisy = add_noise(ag, snr_db, rng_g)

    floor_abs_noisy = {}
    for k in range(1, n_story + 1):
        rng_k = np.random.default_rng(master.integers(0, 2**32 - 1))
        floor_abs_noisy[k] = add_noise(floor_abs[k], snr_db, rng_k)

    return tg, ag_noisy, floor_abs_noisy, dt



# =========================================================
# Frequency-domain DSFs
# =========================================================
def transmissibility_welch(
    x_in: np.ndarray,
    y_out: np.ndarray,
    dt: float,
    fmin=0.2,
    fmax=20.0,
    df_target=0.05,
    nperseg=2048
):
    fs = 1.0 / dt
    n = len(x_in)

    if nperseg is None:
        nperseg = int(fs / df_target)
        nperseg = min(nperseg, max(n // 4, 256))
        nperseg = max(nperseg, 256)
        nperseg = 2 ** int(np.log2(nperseg))

    noverlap = nperseg // 2

    f, Pxx = signal.welch(
        x_in, fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density"
    )
    f, Pxy = signal.csd(
        x_in, y_out, fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density"
    )

    H = Pxy / (Pxx + 1e-12)
    T = np.abs(H)
    phase = np.angle(H)

    mask = (f >= fmin) & (f <= fmax)
    return f[mask], T[mask], phase[mask]


def band_mask(f, f1, low=None, high=None):
    return (f >= low * f1) & (f <= high * f1)


def apply_band_nan(f, T, f1, low=None, high=None):
    T2 = T.copy()
    mask = band_mask(f, f1, low, high)
    T2[~mask] = np.nan
    return T2


def dsf_t_cosine(Tref, Trun):
    valid = (~np.isnan(Tref)) & (~np.isnan(Trun))
    a = Tref[valid]
    b = Trun[valid]
    if len(a) == 0:
        return np.nan
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    sim = num / den
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - sim


def dsf_t_maccriteria(Tref, Trun):
    valid = (~np.isnan(Tref)) & (~np.isnan(Trun))
    a = Tref[valid]
    b = Trun[valid]
    if len(a) == 0:
        return np.nan
    num = float(np.dot(a, b))
    den = float(np.dot(a, a) * np.dot(b, b) + 1e-12)
    tac = (num * num) / den
    tac = max(min(tac, 1.0), 0.0)
    return 1.0 - tac


def t_centroid(f, T):
    valid = (~np.isnan(T)) & (T > 0)
    if not np.any(valid):
        return np.nan
    fv = f[valid]
    Tv = T[valid]
    return float(np.sum(fv * Tv) / (np.sum(Tv) + 1e-12))


def dsf_t_centroid(f, Tref, Tpost):
    cref = t_centroid(f, Tref)
    cpost = t_centroid(f, Tpost)
    if (not np.isfinite(cref)) or (not np.isfinite(cpost)) or abs(cref) < 1e-12:
        return np.nan
    return float(abs(cpost - cref) / abs(cref))


def peak_f_in_band(f, T, f_target, band_low=0.2, band_high=1.2):
    f = np.asarray(f, float)
    T = np.asarray(T, float).copy()
    mask = (f >= band_low * f_target) & (f <= band_high * f_target)
    T[~mask] = np.nan
    if np.all(np.isnan(T)):
        return np.nan, np.nan
    idx = int(np.nanargmax(T))
    return float(f[idx]), float(T[idx])


def dsf_peak_shift(f, Tref, Tpost, f_target, band_low=0.2, band_high=1.2):
    fpk_ref, _ = peak_f_in_band(f, Tref, f_target, band_low, band_high)
    fpk_post, _ = peak_f_in_band(f, Tpost, f_target, band_low, band_high)
    if (not np.isfinite(fpk_ref)) or (not np.isfinite(fpk_post)) or (fpk_ref <= 1e-12):
        return np.nan, np.nan, np.nan
    return abs(fpk_post - fpk_ref) / fpk_ref, fpk_ref, fpk_post


def dsf_area_change(f, Tref, Tpost, f_target, band_low=0.2, band_high=1.1):
    mask = (f >= band_low * f_target) & (f <= band_high * f_target)
    area_ref = np.trapezoid(Tref[mask], f[mask])
    area_post = np.trapezoid(Tpost[mask], f[mask])
    return abs(area_post - area_ref) / (area_ref + 1e-12)

# =========================================================
# Time-domain DSFs
# =========================================================
def dsf_time_delay(x, y_ref, y_post, dt):
    """
    通过互相关计算响应相对于输入的延迟变化
    x: 地面输入, y: 楼层响应
    """
    # 计算受损前的滞后
    corr_ref = signal.correlate(y_ref, x, mode='full')
    lags = signal.correlation_lags(len(y_ref), len(x))
    delay_ref = lags[np.argmax(np.abs(corr_ref))] * dt
    # 计算受损后的滞后
    corr_post = signal.correlate(y_post, x, mode='full')
    delay_post = lags[np.argmax(np.abs(corr_post))] * dt
    return float(abs(delay_post - delay_ref))

def butter_bandpass_filter(x, fs, f_low, f_high, order=4):
    nyq = 0.5 * fs
    f_low = max(f_low, 1e-6)
    f_high = min(f_high, nyq * 0.999)
    b, a = signal.butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def accel_to_disp_freqdomain(a, dt, f_low, f_high, detrend=True, eps_w=1e-6):
    a = np.asarray(a, float)
    fs = 1.0 / dt
    if detrend:
        a0 = signal.detrend(a, type="linear")
    else:
        a0 = a - np.mean(a)
    af = butter_bandpass_filter(a0, fs, f_low, f_high, order=4)
    n = len(af)
    A = np.fft.rfft(af)
    w = 2 * np.pi * np.fft.rfftfreq(n, d=dt)
    w_safe = np.where(w < eps_w, eps_w, w)
    D = A / (-(w_safe ** 2))
    d = np.fft.irfft(D, n=n)
    return d, af


def kprx_from_acc_only(ag, aout_abs, dt, f1, band_low=0.2, band_high=1.2, eps=1e-9):
    f_low = band_low * f1
    f_high = band_high * f1
    d_g, ag_f = accel_to_disp_freqdomain(ag, dt, f_low, f_high)
    d_o, ao_f = accel_to_disp_freqdomain(aout_abs, dt, f_low, f_high)
    denom = (d_o - d_g)
    k_ts = ao_f / (denom + eps)
    K = float(np.median(np.abs(k_ts)))
    return K, k_ts


def dsf_kprx_rel(K_ref, K_post):
    return (K_ref - K_post) / (K_ref + 1e-12)


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def fit_ar_least_squares(y, order=4):
    y = np.asarray(y, float)
    p = int(order)
    if len(y) <= p + 5:
        return None
    Y = y[p:]
    X = np.column_stack([y[p - i - 1:-(i + 1)] for i in range(p)])
    a, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return a.astype(float)


def compute_time_domain_features(ref_dir: Path, post_dir: Path, n_story: int, snr_db=None):
    _, ag_ref, floor_ref, dt = load_abs_accels_txt(ref_dir, n_story, snr_db=snr_db)
    _, ag_post, floor_post, _ = load_abs_accels_txt(post_dir, n_story, snr_db=snr_db)

    y_ref = floor_ref[n_story]
    y_post = floor_post[n_story]
    x_ref = ag_ref
    x_post = ag_post

    m = min(len(y_ref), len(y_post), len(x_ref), len(x_post))
    y_ref = y_ref[:m]
    y_post = y_post[:m]
    x_ref = x_ref[:m]
    x_post = x_post[:m]

    dsf_delay = dsf_time_delay(x_ref, y_ref, y_post, dt)

    yref_n = y_ref / (rms(x_ref) + 1e-12)
    ypost_n = y_post / (rms(x_post) + 1e-12)

    rms_ref = rms(yref_n)
    rms_post = rms(ypost_n)
    dsf_rms = (rms_ref - rms_post) / (rms_ref + 1e-12)

    a_ref = fit_ar_least_squares(yref_n, order=4)
    a_post = fit_ar_least_squares(ypost_n, order=4)
    if (a_ref is None) or (a_post is None):
        dsf_ar = np.nan
    else:
        dsf_ar = float(np.linalg.norm(a_post - a_ref, ord=2))

    return {
        "DSF_RMSratio": float(dsf_rms),
        "DSF_AR4_L2": float(dsf_ar),
        "DSF_Time_Delay": float(dsf_delay)
    }


def compute_dsfs_for_one_run(
    healthy_ref_dir: Path,
    posteq_wn_dir: Path,
    n_story: int,
    f1: float,
    f2: float,
    snr_db=None,
    fmin=0.2,
    fmax=25.0,
    nperseg=2048,
    band_low=0.2,
    band_high=1.2,
):
    _, ag_ref, floor_ref, dt_ref = load_abs_accels_txt(healthy_ref_dir, n_story, snr_db=snr_db)
    _, ag_post, floor_post, dt_post = load_abs_accels_txt(posteq_wn_dir, n_story, snr_db=snr_db)

    fR, TR, PR = transmissibility_welch(
        ag_ref, floor_ref[n_story], dt_ref, fmin=fmin, fmax=fmax, nperseg=nperseg
    )
    fP, TP, PP = transmissibility_welch(
        ag_post, floor_post[n_story], dt_post, fmin=fmin, fmax=fmax, nperseg=nperseg
    )

    m = min(len(fR), len(fP))
    f = fR[:m]
    TR = TR[:m]
    TP = TP[:m]

    # --- 一阶特征计算 (Mode 1) ---
    TR_b1 = apply_band_nan(f, TR, f1, low=band_low, high=band_high)
    TP_b1 = apply_band_nan(f, TP, f1, low=band_low, high=band_high)
    # dsf_t1_cos = dsf_t_cosine(TR_b1, TP_b1)
    dsf_t1_mac = dsf_t_maccriteria(TR_b1, TP_b1)
    dsf_t1_cent = dsf_t_centroid(f, TR_b1, TP_b1)
    dsf_df1, _, _ = dsf_peak_shift(f, TR, TP, f1, band_low=band_low, band_high=band_high)
    dsf_t1_area = dsf_area_change(f, TR, TP, f1, band_low=band_low, band_high=band_high)


    # --- 二阶特征计算 (Mode 2) ---
    TR_b2 = apply_band_nan(f, TR, f2, low=0.4, high=1.15)
    TP_b2 = apply_band_nan(f, TP, f2, low=0.4, high=1.15)
    # dsf_t2_cos = dsf_t_cosine(TR_b2, TP_b2)
    dsf_t2_mac = dsf_t_maccriteria(TR_b2, TP_b2)
    dsf_t2_cent = dsf_t_centroid(f, TR_b2, TP_b2)
    dsf_df2, _, _ = dsf_peak_shift(f, TR, TP, f2, band_low=0.4, band_high=1.15)
    dsf_t2_area = dsf_area_change(f, TR, TP, f2, band_low=0.4, band_high=1.15)


    K_ref, _ = kprx_from_acc_only(ag_ref, floor_ref[n_story], dt_ref, f1, band_low=0.2, band_high=1.2)
    K_post, _ = kprx_from_acc_only(ag_post, floor_post[n_story], dt_post, f1, band_low=0.2, band_high=1.2)
    dsf_kprx = dsf_kprx_rel(K_ref, K_post)

    time_feats = compute_time_domain_features(healthy_ref_dir, posteq_wn_dir, n_story, snr_db=snr_db)

    out = {
        # "DSF_T1_COS": float(dsf_t1_cos) if np.isfinite(dsf_t1_cos) else np.nan,
        "DSF_T1_CENT": float(dsf_t1_cent) if np.isfinite(dsf_t1_cent) else np.nan,
        "DSF_T1_MAC": float(dsf_t1_mac) if np.isfinite(dsf_t1_mac) else np.nan,
        "DSF_dF1peak": float(dsf_df1) if np.isfinite(dsf_df1) else np.nan,
        "DSF_T1_AREA": float(dsf_t1_area) if np.isfinite(dsf_t1_area) else np.nan,

        # "DSF_T2_COS": float(dsf_t2_cos) if np.isfinite(dsf_t2_cos) else np.nan,
        "DSF_T2_CENT": float(dsf_t2_cent) if np.isfinite(dsf_t2_cent) else np.nan,
        "DSF_T2_MAC": float(dsf_t2_mac) if np.isfinite(dsf_t2_mac) else np.nan,
        "DSF_dF2peak": float(dsf_df2) if np.isfinite(dsf_df2) else np.nan,
        "DSF_T2_AREA": float(dsf_t2_area) if np.isfinite(dsf_t2_area) else np.nan,

        "DSF_KPRX": float(dsf_kprx) if np.isfinite(dsf_kprx) else np.nan,

    }
    out.update(time_feats)
    return out


# =========================================================
# Metadata and labels
# =========================================================
def parse_gm_folder_name(name: str) -> Dict[str, Any]:
    out = {
        "gm_folder": name,
        "gm_id": None,
        "target_pga_g_from_name": np.nan,
    }
    m = re.match(r"^(gm_\d+)_pga_([0-9]+p[0-9]+)g$", name)
    if m:
        out["gm_id"] = m.group(1)
        out["target_pga_g_from_name"] = float(m.group(2).replace("p", "."))
    return out


def read_model_metadata(model_dir: Path) -> Dict[str, Any]:
    params = safe_load_json(model_dir / "params.json")
    msum = safe_load_json(model_dir / "modal_summary.json")
    psum = safe_load_json(model_dir / "pushover_summary.json")

    model_id = model_dir.name

    n_story = int(params["n_story"])
    n_bay = int(params["n_bay"])
    first_story_h = float(params["first_story_h"])
    typical_story_h = float(params["typical_story_h"])
    bay_width = float(params["bay_width"])
    col_b = float(params["col"]["b"])
    col_h = float(params["col"]["h"])
    beam_b = float(params["beam"]["b"])
    beam_h = float(params["beam"]["h"])

    fc = float(params["mat"]["fc"])
    fy = float(params["mat"]["fy"])

    q_dead = float(params["loads"]["q_dead"])
    q_live = float(params["loads"]["q_live"])
    q_tributary_width = float(params["loads"]["tributary_width"])

    damping = float(params["damping"]["zeta"])

    # total_height = first_story_h + (n_story - 1) * typical_story_h
    total_height = typical_story_h + (n_story - 1) * typical_story_h
    total_width = n_bay * bay_width

    T1 = float(msum["T1"])
    f1 = 1.0 / T1
    T2 = float(msum["period"][1])
    f2 = 1.0 / T2

    theta_IO = float(psum["isdr_P1"])
    theta_LS = float(psum["isdr_P2"])
    theta_CP = float(psum["isdr_P3"])

    return {
        "model_id": model_id,
        "n_story": n_story,
        "n_bay": n_bay,
        "first_story_h": first_story_h,
        "typical_story_h": typical_story_h,
        "bay_width": bay_width,
        "total_height": total_height,
        "total_width": total_width,
        "col_b": col_b,
        "col_h": col_h,
        "beam_b": beam_b,
        "beam_h": beam_h,
        "fc": fc,
        "fy": fy,
        "q_dead": q_dead,
        "q_live": q_live,
        "q_tributary_width": q_tributary_width,
        "damping": damping,
        "T1": T1,
        "f1": f1,
        "T2": T2,
        "f2": f2,
        "theta_IO": theta_IO,
        "theta_LS": theta_LS,
        "theta_CP": theta_CP,
    }


def extract_run_labels(run_dir: Path) -> Dict[str, Any]:
    eq_summary = safe_load_json(run_dir / "eq_summary.json")
    if not eq_summary:
        raise FileNotFoundError(f"Missing eq_summary.json in {run_dir}")

    return {
        "gm_id_from_eq": eq_summary.get("gm_id", None),
        "target_pga_g": float(eq_summary.get("target_pga_g", np.nan)),
        "gm_scale": float(eq_summary.get("gm_scale", np.nan)),
        "gm_dt": float(eq_summary.get("gm_dt", np.nan)),
        "zeta": float(eq_summary.get("zeta", np.nan)),

        "MIDR": float(eq_summary.get("edp_isdr_max", np.nan)),
        "roof_drift_max": float(eq_summary.get("edp_roof_drift_max", np.nan)),
        "residual_roof_drift": float(eq_summary.get("edp_residual_roof_drift", np.nan)),
        "damage_state": int(eq_summary.get("damage_state")) if "damage_state" in eq_summary else np.nan,

        "eq_ok": int(eq_summary.get("eq_ok", 0)),
        "decay_ok": int(eq_summary.get("decay_ok", 0)),
        "posteq_wn_ok": int(eq_summary.get("posteq_wn_ok", 0)),
    }


# =========================================================
# Build one sample
# =========================================================
def build_one_sample(
    model_meta: Dict[str, Any],
    run_dir: Path,
    healthy_ref_dir: Path,
    snr_db=None,
) -> Dict[str, Any]:
    gm_info = parse_gm_folder_name(run_dir.name)
    label_info = extract_run_labels(run_dir)

    posteq_wn_dir = run_dir / "posteq_white_noise"
    if not posteq_wn_dir.exists():
        raise FileNotFoundError(f"Missing posteq_white_noise: {posteq_wn_dir}")

    dsfs = compute_dsfs_for_one_run(
        healthy_ref_dir=healthy_ref_dir,
        posteq_wn_dir=posteq_wn_dir,
        n_story=int(model_meta["n_story"]),
        f1=float(model_meta["f1"]),
        f2=float(model_meta["f2"]),
        snr_db=snr_db,
    )

    row = {}
    row.update(model_meta)
    row.update(gm_info)
    row.update(label_info)
    row.update(dsfs)

    row["sample_id"] = f"{model_meta['model_id']}__{run_dir.name}"
    row["run_dir"] = str(run_dir)
    row["healthy_ref_dir"] = str(healthy_ref_dir)
    row["posteq_wn_dir"] = str(posteq_wn_dir)

    return row


# =========================================================
# Main builder
# =========================================================
def build_population_dataset(
    models_root="01_models",
    results_root="03_simulation_results",
    out_csv="population_dataset.csv",
    out_failed_csv="population_failed_samples.csv",
    snr_db=None,
    max_models=None,
    max_runs_per_model=None,
):
    models_root = Path(models_root)
    results_root = Path(results_root)

    rows = []
    failed_rows = []

    model_dirs = sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("model_")])

    if max_models is not None:
        model_dirs = model_dirs[:max_models]

    print(f"[INFO] Found {len(model_dirs)} models.")

    for i, model_dir in enumerate(model_dirs, start=1):
        model_id = model_dir.name
        print(f"\n[INFO] Processing {i}/{len(model_dirs)}: {model_id}")

        try:
            model_meta = read_model_metadata(model_dir)
        except Exception as e:
            failed_rows.append({
                "model_id": model_id,
                "run_name": None,
                "reason": f"read_model_metadata failed: {e}",
                "traceback": traceback.format_exc(),
            })
            continue

        result_model_dir = Path(results_root) / model_id
        healthy_ref_dir = result_model_dir / "healthy_ref"

        if not result_model_dir.exists():
            failed_rows.append({
                "model_id": model_id,
                "run_name": None,
                "reason": f"Missing result folder: {result_model_dir}",
                "traceback": "",
            })
            continue

        if not healthy_ref_dir.exists():
            failed_rows.append({
                "model_id": model_id,
                "run_name": None,
                "reason": f"Missing healthy_ref: {healthy_ref_dir}",
                "traceback": "",
            })
            continue

        run_dirs = sorted([
            p for p in result_model_dir.iterdir()
            if p.is_dir() and p.name.startswith("gm_")
        ])

        if max_runs_per_model is not None:
            run_dirs = run_dirs[:max_runs_per_model]

        print(f"[INFO]   Found {len(run_dirs)} gm runs.")

        for j, run_dir in enumerate(run_dirs, start=1):
            print(f"[INFO]   Run {j}/{len(run_dirs)}: {run_dir.name}")
            try:
                row = build_one_sample(
                    model_meta=model_meta,
                    run_dir=run_dir,
                    healthy_ref_dir=healthy_ref_dir,
                    snr_db=snr_db,
                )
                row["valid_flag"] = 1
                rows.append(row)

            except Exception as e:
                failed_rows.append({
                    "model_id": model_id,
                    "run_name": run_dir.name,
                    "reason": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"[WARN]   Failed: {run_dir.name} -> {e}")

    df = pd.DataFrame(rows)
    df_failed = pd.DataFrame(failed_rows)

    # if len(df) > 0:
    #     preferred_cols = [
    #         "sample_id",
    #         "model_id",
    #         "gm_id",
    #         "gm_id_from_eq",
    #         "gm_folder",
    #         "target_pga_g_from_name",
    #         "target_pga_g",
    #         "gm_scale",
    #         "gm_dt",

    #         "n_story",
    #         "n_bay",
    #         "first_story_h",
    #         "typical_story_h",
    #         "bay_width",
    #         "total_height",
    #         "total_width",
    #         "T1",
    #         "f1",
    #         "theta_IO",
    #         "theta_LS",
    #         "theta_CP",
    #         "zeta",

    #         "DSF_T_COS",
    #         "DSF_T_CENT",
    #         "DSF_T_MAC",
    #         "DSF_dFpeak",
    #         "DSF_KPRX",
    #         "DSF_RMSratio",
    #         "DSF_AR4_L2",
    #         "fpk_ref",
    #         "fpk_post",

    #         "MIDR",
    #         "roof_drift_max",
    #         "residual_roof_drift",
    #         "damage_state",

    #         "eq_ok",
    #         "decay_ok",
    #         "posteq_wn_ok",
    #         "valid_flag",

    #         "run_dir",
    #         "healthy_ref_dir",
    #         "posteq_wn_dir",
    #     ]
    #     remain_cols = [c for c in df.columns if c not in preferred_cols]
    #     df = df[[c for c in preferred_cols if c in df.columns] + remain_cols]

    df.to_csv(out_csv, index=False)
    df_failed.to_csv(out_failed_csv, index=False)

    print("\n[INFO] ====================================")
    print(f"[INFO] Saved dataset: {out_csv}")
    print(f"[INFO] Saved failed log: {out_failed_csv}")
    print(f"[INFO] Valid samples: {len(df)}")
    print(f"[INFO] Failed samples: {len(df_failed)}")

    if len(df) > 0:
        print("\n[INFO] Damage state distribution:")
        print(df["damage_state"].value_counts(dropna=False).sort_index())

    return df, df_failed


if __name__ == "__main__":
    df, df_failed = build_population_dataset(
        models_root="01_models4",
        results_root="03_simulation_results4",
        out_csv="population_dataset_v4.csv",
        out_failed_csv="population_failed_samples4.csv",
        snr_db=None,
        max_models=None,
        max_runs_per_model=None,
    )

    print("\n[INFO] Preview:")
    print(df.head(8))
    
    # =========================================================
    # 2. DS 分布
    # =========================================================
    print("\n===== DS Distribution =====")
    ds_counts = df["damage_state"].value_counts().sort_index()
    print(ds_counts)

    print("\nDS percentage:")
    print((ds_counts / len(df)).round(3))