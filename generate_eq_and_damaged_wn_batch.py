import os
import json
import argparse
import numpy as np
import pandas as pd
from openseespy.opensees import *

from generate_healthy_wn import (
    rebuild_healthy_model_from_params,
    generate_white_noise_accel,
    save_time_history
)

# ---------------------------------------------------------
# 1. 工具函数：读取 json
# ---------------------------------------------------------
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# 2. 读取 model 参数与 pushover 阈值
# ---------------------------------------------------------
def load_model_inputs(model_id, models_base_dir="01_models"):
    model_name = f"model_{model_id:04d}"
    model_dir = os.path.join(models_base_dir, model_name)
    params = load_json(os.path.join(model_dir, "params.json"))
    pushover = load_json(os.path.join(model_dir, "pushover_summary.json"))
    return model_dir, params, pushover


# ---------------------------------------------------------
# 3. Rayleigh 阻尼
# ---------------------------------------------------------
def apply_rayleigh_damping_from_first_two_modes(zeta):
    lam = eigen(2)
    omega1 = np.sqrt(lam[0])
    omega2 = np.sqrt(lam[1])
    a0 = 2.0 * zeta * omega1 * omega2 / (omega1 + omega2)
    a1 = 2.0 * zeta / (omega1 + omega2)
    rayleigh(a0, 0.0, 0.0, a1)
    return omega1, omega2, a0, a1


# ---------------------------------------------------------
# 4. 统一 recorder（EQ / damaged WN 都可用）
# ---------------------------------------------------------
def setup_story_recorders(out_dir, story_master, roof_ctrl_node):
    os.makedirs(out_dir, exist_ok=True)

    for k, nd in story_master.items():
        recorder(
            "Node", "-file", os.path.join(out_dir, f"acc_story_{k}.txt"),
            "-time", "-node", nd, "-dof", 1, "accel"
        )
        recorder(
            "Node", "-file", os.path.join(out_dir, f"disp_story_{k}.txt"),
            "-time", "-node", nd, "-dof", 1, "disp"
        )

    recorder(
        "Node", "-file", os.path.join(out_dir, "acc_roof.txt"),
        "-time", "-node", roof_ctrl_node, "-dof", 1, "accel"
    )
    recorder(
        "Node", "-file", os.path.join(out_dir, "disp_roof.txt"),
        "-time", "-node", roof_ctrl_node, "-dof", 1, "disp"
    )


# ---------------------------------------------------------
# 5. 瞬态分析设置
# ---------------------------------------------------------
def setup_transient_analysis():
    wipeAnalysis()
    system("BandGeneral")
    constraints("Transformation")
    numberer("RCM")
    test("EnergyIncr", 1.0e-7, 150, 0, 2)
    algorithm("NewtonLineSearch", 0.8)
    integrator("Newmark", 0.5, 0.25)
    analysis("Transient")


# ---------------------------------------------------------
# 6. 从 recorder 位移结果计算 EDP
# ---------------------------------------------------------
def compute_edp_from_recorded_displacements(story_master, y_levels, out_dir):
    disp = {}

    for k in story_master.keys():
        filepath = os.path.join(out_dir, f"disp_story_{k}.txt")
        arr = np.loadtxt(filepath)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        disp[k] = arr[:, 1]

    n_story = len(story_master)
    roof_disp = disp[n_story]
    H_total = y_levels[-1]

    roof_drift_max = float(np.max(np.abs(roof_disp)) / H_total)
    residual_roof_drift = float(abs(roof_disp[-1]) / H_total)

    isdr_max = 0.0
    for k in range(1, n_story + 1):
        if k == 1:
            lower = np.zeros_like(disp[k])
        else:
            lower = disp[k - 1]
        upper = disp[k]

        h_story = y_levels[k] - y_levels[k - 1]
        story_isdr = np.abs(upper - lower) / h_story
        isdr_max = max(isdr_max, float(np.max(story_isdr)))

    return {
        "roof_drift_max": roof_drift_max,
        "residual_roof_drift": residual_roof_drift,
        "isdr_max": isdr_max
    }


# ---------------------------------------------------------
# 7. 根据 P1/P2/P3 打 label
# ---------------------------------------------------------
def assign_damage_state(isdr_max, isdr_P1, isdr_P2, isdr_P3):
    if isdr_max < isdr_P1:
        return 0
    elif isdr_max < isdr_P2:
        return 1
    elif isdr_max < isdr_P3:
        return 2
    else:
        return 3


# ---------------------------------------------------------
# 8. 地震输入分析
# ---------------------------------------------------------
def run_earthquake_analysis(
    model_data,
    gm_file,
    gm_dt,
    gm_scale,
    out_dir
):
    os.makedirs(out_dir, exist_ok=True)

    # 清 recorder，避免重复
    remove("recorders")

    story_master = model_data["story_master"]
    roof_ctrl_node = model_data["roof_ctrl_node"]

    # 保存一份输入地震波（原始输入记录）
    gm_raw = np.loadtxt(gm_file)
    t = np.arange(len(gm_raw)) * gm_dt
    eq_input_copy = os.path.join(out_dir, "acc_ground.txt")
    data = np.column_stack((t, gm_raw))
    np.savetxt(eq_input_copy, data)

    # recorder
    setup_story_recorders(out_dir, story_master, roof_ctrl_node)

    # analysis
    ts_tag = 20001
    pat_tag = 20001
    timeSeries("Path", ts_tag, "-dt", gm_dt, "-filePath", gm_file, "-factor", gm_scale)
    pattern("UniformExcitation", pat_tag, 1, "-accel", ts_tag)

    setup_transient_analysis()

    n_steps = len(gm_raw)
    ok = analyze(n_steps, gm_dt)

    edp = compute_edp_from_recorded_displacements(
        story_master=story_master,
        y_levels=model_data["y_levels"],
        out_dir=out_dir
    )

    return {
        "gm_file": gm_file,
        "gm_dt": gm_dt,
        "gm_scale": gm_scale,
        "n_steps": int(n_steps),
        "ok": int(ok),
        "edp": edp
    }


# ---------------------------------------------------------
# 9. 自由震动衰减
# ---------------------------------------------------------
def run_free_decay(gm_dt, decay_time=10.0, verbose=True):
    time_before = getTime()

    if verbose:
        print("\n[Free Decay] Start...")
        print(f"  Current analysis time = {time_before:.4f} s")

    try:
        remove("loadPattern", 20001)
        if verbose:
            print("  Main EQ loadPattern 20001 removed.")
    except Exception:
        if verbose:
            print("  Warning: loadPattern 20001 not found or already removed.")

    try:
        remove("recorders")
        if verbose:
            print("  Recorders removed.")
    except Exception:
        pass

    wipeAnalysis()
    system("BandGeneral")
    constraints("Transformation")
    numberer("RCM")
    test("EnergyIncr", 1.0e-7, 150, 0, 2)
    algorithm("NewtonLineSearch", 0.8)
    integrator("Newmark", 0.5, 0.25)
    analysis("Transient")

    n_decay = int(decay_time / gm_dt)
    ok = analyze(n_decay, gm_dt)

    time_after = getTime()

    if verbose:
        print(f"  Free decay finished | ok = {ok}")
        print(f"  Time after decay = {time_after:.4f} s")

    loadConst("-time", 0.0)
    if verbose:
        print(f"  Time reset to {getTime():.4f} s")

    return {
        "ok": int(ok),
        "time_before": float(time_before),
        "time_after": float(time_after),
        "decay_time": float(decay_time),
    }


# ---------------------------------------------------------
# 10. 地震后白噪声分析
# ---------------------------------------------------------
def run_posteq_white_noise_analysis(
    model_data,
    out_dir,
    dt=0.01,
    duration=60.0,
    rms_g=0.005,
    seed=1
):
    os.makedirs(out_dir, exist_ok=True)

    remove("recorders")

    story_master = model_data["story_master"]
    roof_ctrl_node = model_data["roof_ctrl_node"]

    t, acc = generate_white_noise_accel(
        dt=dt,
        duration=duration,
        rms_g=rms_g,
        seed=seed
    )

    wn_file = os.path.join(out_dir, "posteq_white_noise_acc.txt")
    save_time_history(wn_file, acc)

    ground_acc_file = os.path.join(out_dir, "acc_ground.txt")
    data = np.column_stack((t, acc))
    np.savetxt(ground_acc_file, data)

    setup_story_recorders(out_dir, story_master, roof_ctrl_node)

    ts_tag = 30001
    pat_tag = 30001
    timeSeries("Path", ts_tag, "-dt", dt, "-filePath", wn_file, "-factor", 1.0)
    pattern("UniformExcitation", pat_tag, 1, "-accel", ts_tag)

    setup_transient_analysis()

    n_steps = len(acc)
    ok = analyze(n_steps, dt)

    run_summary = {
        "dt": dt,
        "duration": duration,
        "rms_g": rms_g,
        "seed": seed,
        "n_steps": int(n_steps),
        "ok": int(ok)
    }

    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary


# ---------------------------------------------------------
# 11. 单个样本主流程
# ---------------------------------------------------------
def run_one_eq_sample_with_posteq_wn(
    model_id,
    gm_id,
    gm_file,
    gm_dt,
    gm_scale,
    job_target_pga_g,
    models_base_dir="01_models",
    results_base_dir="03_simulation_results",
    posteq_wn_dt=0.01,
    posteq_wn_duration=60.0,
    posteq_wn_rms_g=0.005,
    posteq_wn_seed=1
):
    model_name = f"model_{model_id:04d}"
    gm_tag = f"{gm_id}_pga_{job_target_pga_g:.2f}g".replace(".", "p")

    sample_dir = os.path.join(results_base_dir, model_name, gm_tag)
    os.makedirs(sample_dir, exist_ok=True)

    # 1) 读模型参数与 pushover 阈值
    model_dir, params, pushover = load_model_inputs(model_id, models_base_dir=models_base_dir)

    isdr_P1 = float(pushover["isdr_P1"])
    isdr_P2 = float(pushover["isdr_P2"])
    isdr_P3 = float(pushover["isdr_P3"])

    # 2) 重建健康模型
    model_data = rebuild_healthy_model_from_params(params)

    # 3) 加阻尼
    zeta = float(params["damping"]["zeta"])
    omega1, omega2, a0, a1 = apply_rayleigh_damping_from_first_two_modes(zeta)

    # 4) 地震分析
    eq_info = run_earthquake_analysis(
        model_data=model_data,
        gm_file=gm_file,
        gm_dt=gm_dt,
        gm_scale=gm_scale,
        out_dir=sample_dir
    )

    edp = eq_info["edp"]
    ds_label = assign_damage_state(
        isdr_max=edp["isdr_max"],
        isdr_P1=isdr_P1,
        isdr_P2=isdr_P2,
        isdr_P3=isdr_P3
    )

    decay_info = run_free_decay(
        gm_dt=gm_dt,
        decay_time=10.0,
        verbose=True
    )

    # 5) 地震后白噪声
    posteq_dir = os.path.join(sample_dir, "posteq_white_noise")
    posteq_info = run_posteq_white_noise_analysis(
        model_data=model_data,
        out_dir=posteq_dir,
        dt=posteq_wn_dt,
        duration=posteq_wn_duration,
        rms_g=posteq_wn_rms_g,
        seed=posteq_wn_seed + model_id
    )

    # 6) 保存汇总（保持原来单个 sample 的输出）
    summary = {
        "model_id": model_id,
        "gm_id": gm_id,
        "gm_file": gm_file,
        "gm_dt": gm_dt,
        "gm_scale": gm_scale,
        "target_pga_g": job_target_pga_g,
        "zeta": zeta,
        "omega1": omega1,
        "omega2": omega2,
        "rayleigh_a0": a0,
        "rayleigh_a1": a1,
        "isdr_P1": isdr_P1,
        "isdr_P2": isdr_P2,
        "isdr_P3": isdr_P3,
        "edp_isdr_max": edp["isdr_max"],
        "edp_roof_drift_max": edp["roof_drift_max"],
        "edp_residual_roof_drift": edp["residual_roof_drift"],
        "damage_state": ds_label,
        "eq_ok": eq_info["ok"],
        "decay_ok": decay_info["ok"],
        "posteq_wn_ok": posteq_info["ok"],
        "sample_dir": sample_dir
    }

    with open(os.path.join(sample_dir, "eq_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[EQ] model_{model_id:04d} | {gm_id} | scale={gm_scale:.4f} "
        f"| ISDRmax={edp['isdr_max']:.4f} | DS={ds_label}"
    )

    wipe()
    return summary


# ---------------------------------------------------------
# 12. 读取地震波 metadata
# ---------------------------------------------------------
def load_gm_metadata(csv_path="02_ground_motions/gm_metadata.csv"):
    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------
# 13. 根据 target PGA 生成 scale 表
# ---------------------------------------------------------
def build_gm_scale_table(
    gm_df,
    target_pga_g=(0.20, 0.40, 0.60, 0.80, 1.0, 1.2, 1.4),
    gm_base_dir="02_ground_motions"
):
    rows = []
    for _, r in gm_df.iterrows():
        gm_id = r["gm_id"]
        gm_dt = float(r["dt"])
        pga_raw = float(r["pga_mps2"])
        gm_relpath = str(r["new_filepath"]).replace("\\", os.sep).replace("/", os.sep)
        gm_file = os.path.join(gm_base_dir, gm_relpath)

        for pga_g in target_pga_g:
            pga_target = pga_g * 9.81
            sf = pga_target / pga_raw

            rows.append({
                "gm_id": gm_id,
                "gm_file": gm_file,
                "gm_dt": gm_dt,
                "pga_raw_mps2": pga_raw,
                "target_pga_g": pga_g,
                "target_pga_mps2": pga_target,
                "scale_factor": sf
            })

    scale_df = pd.DataFrame(rows)
    return scale_df


# ---------------------------------------------------------
# 14. 保存 scale 表
# ---------------------------------------------------------
def save_gm_scale_table(scale_df, out_csv="02_ground_motions/gm_scale_table.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    scale_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


# ---------------------------------------------------------
# 15. 组合模型 × 地震波 × scale 的 jobs
# ---------------------------------------------------------
def build_jobs_from_models_and_gms(model_ids, scale_df):
    jobs = []
    for model_id in model_ids:
        for _, r in scale_df.iterrows():
            jobs.append({
                "model_id": int(model_id),
                "gm_id": r["gm_id"],
                "gm_file": r["gm_file"],
                "gm_dt": float(r["gm_dt"]),
                "gm_scale": float(r["scale_factor"]),
                "target_pga_g": float(r["target_pga_g"]),
            })
    return jobs


# ---------------------------------------------------------
# 16. 批量运行（仅返回，不写总汇总文件）
# ---------------------------------------------------------
def batch_run_eq_samples(
    jobs,
    models_base_dir="01_models",
    results_base_dir="03_simulation_results"
):
    summaries = []

    for job in jobs:
        try:
            summary = run_one_eq_sample_with_posteq_wn(
                model_id=job["model_id"],
                gm_id=job["gm_id"],
                gm_file=job["gm_file"],
                gm_dt=job["gm_dt"],
                gm_scale=job["gm_scale"],
                job_target_pga_g=job["target_pga_g"],
                models_base_dir=models_base_dir,
                results_base_dir=results_base_dir
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[EQ] failed for job={job}: {e}")
            wipe()

    return summaries


# ---------------------------------------------------------
# 17. main：每个任务只跑一个模型
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int, required=True, help="Model ID to run")
    parser.add_argument("--models-base-dir", type=str, default="01_models")
    parser.add_argument("--results-base-dir", type=str, default="03_simulation_results")
    parser.add_argument(
        "--target-pga-g",
        type=float,
        nargs="+",
        default=[0.20, 0.40, 0.60, 0.80, 1.0, 1.2, 1.4],
        help="Target PGA levels in g"
    )
    args = parser.parse_args()

    model_id = args.model_id

    # 1. 读取 metadata
    gm_df = load_gm_metadata("02_ground_motions/gm_metadata.csv")

    # 2. 生成当前任务需要的 scale 表
    scale_df = build_gm_scale_table(
        gm_df,
        target_pga_g=tuple(args.target_pga_g),
        gm_base_dir="02_ground_motions"
    )

    # 3. 只让一个任务写 scale 表，避免并行冲突
    if model_id == 1:
        save_gm_scale_table(scale_df, "02_ground_motions/gm_scale_table.csv")

    # 4. 当前任务只跑一个 model_id
    model_ids = [model_id]

    # 5. 生成 jobs
    jobs = build_jobs_from_models_and_gms(model_ids, scale_df)

    print(f"Running model_id = {model_id}")
    print(f"Total jobs for this model = {len(jobs)}")
    print(scale_df.head())

    # 6. 批量运行，但不写总汇总文件
    summaries = batch_run_eq_samples(
        jobs=jobs,
        models_base_dir=args.models_base_dir,
        results_base_dir=args.results_base_dir
    )

    # 7. 不做总汇总；如需调试，可保留一个单模型汇总
    #    如果你完全不想要额外文件，可以把下面这几行也删掉
    if len(summaries) > 0:
        model_out_dir = args.results_base_dir
        os.makedirs(model_out_dir, exist_ok=True)
        out_csv = os.path.join(model_out_dir, f"summary_model_{model_id:04d}.csv")
        pd.DataFrame(summaries).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")