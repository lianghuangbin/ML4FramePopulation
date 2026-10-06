import os
import json
import numpy as np
from openseespy.opensees import *
from frame2d import build_2d_frame_nodes_elements, build_opensees_2d_model, run_gravity_analysis  #建模函数

# ---------------------------------------------------------
# 1. 读取 params.json
# ---------------------------------------------------------
def load_model_params(model_dir):
    params_file = os.path.join(model_dir, "params.json")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"params.json not found in {model_dir}")

    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params


# ---------------------------------------------------------
# 2. 根据 params 重建健康模型
# ---------------------------------------------------------
def rebuild_healthy_model_from_params(params):
    node_list, element_list, story_nodes, story_master, story_mass, roof_ctrl_node, y_levels = build_2d_frame_nodes_elements(params)

    build_opensees_2d_model(node_list, element_list, story_nodes, story_mass)
    run_gravity_analysis(story_nodes, story_mass)

    model_data = {
        "params": params,
        "node_list": node_list,
        "element_list": element_list,
        "story_nodes": story_nodes,
        "story_master": story_master,
        "story_mass": story_mass,
        "roof_ctrl_node": roof_ctrl_node,
        "y_levels": y_levels,
    }
    return model_data



# ---------------------------------------------------------
# 3. 生成白噪声
# ---------------------------------------------------------
def generate_white_noise_accel(dt, duration, rms_g=0.005, seed=1, g=9.81):
    """
    生成白噪声加速度时程（单位 m/s^2）
    rms_g = 0.01 表示 RMS=0.01g
    """
    rng = np.random.default_rng(seed)
    n = int(duration / dt)

    acc = rng.standard_normal(n)
    acc = acc / np.sqrt(np.mean(acc**2))   # normalize to RMS=1
    acc = acc * rms_g * g                  # target RMS

    t = np.arange(n) * dt
    return t, acc


def save_time_history(filepath, data):
    np.savetxt(filepath, data, fmt="%.8e")


# ---------------------------------------------------------
# 4. 设置记录器
# ---------------------------------------------------------
def setup_healthy_recorders(out_dir, story_master, roof_ctrl_node):
    os.makedirs(out_dir, exist_ok=True)

    # 每层 master 节点：加速度与位移
    for k, nd in story_master.items():
        recorder("Node", "-file", os.path.join(out_dir, f"acc_story_{k}.txt"),
                 "-time", "-node", nd, "-dof", 1, "accel")

        recorder("Node", "-file", os.path.join(out_dir, f"disp_story_{k}.txt"),
                 "-time", "-node", nd, "-dof", 1, "disp")
    # roof节点：加速度与位移
    recorder("Node", "-file", os.path.join(out_dir, "acc_roof.txt"),
             "-time", "-node", roof_ctrl_node, "-dof", 1, "accel")
    recorder("Node", "-file", os.path.join(out_dir, "disp_roof.txt"),
             "-time", "-node", roof_ctrl_node, "-dof", 1, "disp")


# ---------------------------------------------------------
# 5. 瞬态分析设置
# ---------------------------------------------------------
def setup_transient_analysis():
    wipeAnalysis()
    system('BandGeneral')
    constraints('Transformation')
    numberer('RCM')
    test('EnergyIncr', 1.0e-7, 150, 0, 2)
    algorithm('NewtonLineSearch', 0.8)
    integrator('Newmark', 0.5, 0.25)
    analysis('Transient')


# ---------------------------------------------------------
# 6. 对单个模型跑 healthy white noise
# ---------------------------------------------------------
def run_healthy_white_noise_for_one_model(
    model_id,
    models_base_dir="01_models",
    results_base_dir="03_simulation_results",
    dt=0.01,
    duration=60.0,
    rms_g=0.005,
    seed_wn=1
):
    model_name = f"model_{model_id:04d}"
    model_dir = os.path.join(models_base_dir, model_name)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # 结果目录
    out_dir = os.path.join(results_base_dir, model_name, "healthy_ref")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 读取 params
    params = load_model_params(model_dir)

    # 2) 重建健康模型
    model_data = rebuild_healthy_model_from_params(params)

    # 3) 施加 Rayleigh damping
    lambda_vals = eigen(2)
    omega1 = np.sqrt(lambda_vals[0])
    omega2 = np.sqrt(lambda_vals[1])
    zeta = params["damping"]["zeta"]
    a0 = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
    a1 = 2 * zeta / (omega1 + omega2)
    rayleigh(a0, 0.0, 0.0, a1)


    # 4) 生成白噪声
    t, acc = generate_white_noise_accel(
        dt=dt,
        duration=duration,
        rms_g=rms_g,
        seed=seed_wn + model_id
    )

    wn_file = os.path.join(out_dir, "white_noise_acc.txt")
    save_time_history(wn_file, acc)

    ground_acc_file = os.path.join(out_dir, "acc_ground.txt")
    data = np.column_stack((t, acc))  # 第一列时间，第二列加速度
    save_time_history(ground_acc_file, data)

    # 5) 设置地震输入（UniformExcitation）
    ts_tag = 10001
    pat_tag = 10001
    timeSeries("Path", ts_tag, "-dt", dt, "-filePath", wn_file, "-factor", 1.0)
    pattern("UniformExcitation", pat_tag, 1, "-accel", ts_tag)

    # 6) 设置 recorder
    setup_healthy_recorders(
        out_dir=out_dir,
        story_master=model_data["story_master"],
        roof_ctrl_node=model_data["roof_ctrl_node"]
    )

    # 7) 瞬态分析
    setup_transient_analysis()

    n_steps = len(acc)
    ok = analyze(n_steps, dt)

    # 8) 保存 summary
    run_summary = {
        "model_id": model_id,
        "dt": dt,
        "duration": duration,
        "n_steps": n_steps,
        "rms_g": rms_g,
        "seed_wn": seed_wn + model_id,
        "ok": int(ok),
        "out_dir": out_dir
    }

    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"[Healthy WN] model_{model_id:04d} finished | ok = {ok}")

    wipe()
    return run_summary


# ---------------------------------------------------------
# 7. 批量跑全部模型
# ---------------------------------------------------------
def batch_generate_healthy_white_noise(
    models_base_dir="01_models",
    results_base_dir="03_simulation_results",
    dt=0.01,
    duration=60.0,
    rms_g=0.005,
    seed_wn=1
):
    if not os.path.exists(models_base_dir):
        raise FileNotFoundError(f"models_base_dir not found: {models_base_dir}")

    model_folders = []
    for name in sorted(os.listdir(models_base_dir)):
        if name.startswith("model_"):
            full_path = os.path.join(models_base_dir, name)
            if os.path.isdir(full_path):
                model_folders.append(name)

    summaries = []

    for name in model_folders:
        model_id = int(name.split("_")[1])
        try:
            summary = run_healthy_white_noise_for_one_model(
                model_id=model_id,
                models_base_dir=models_base_dir,
                results_base_dir=results_base_dir,
                dt=dt,
                duration=duration,
                rms_g=rms_g,
                seed_wn=seed_wn
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[Healthy WN] model_{model_id:04d} failed: {e}")

    # 总表
    os.makedirs(results_base_dir, exist_ok=True)
    with open(os.path.join(results_base_dir, "healthy_ref_batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nBatch finished. Success = {len(summaries)}")
    return summaries

# ---------------------------------------------------------
# 8. main
# ---------------------------------------------------------
# #先只跑 1 个模型测试
# if __name__ == "__main__":
#     summary = run_healthy_white_noise_for_one_model(
#         model_id=1,
#         models_base_dir="01_models",
#         results_base_dir="03_simulation_results",
#         dt=0.01,
#         duration=60.0,
#         rms_g=0.005,
#         seed_wn=1
#     )

if __name__ == "__main__":
    summaries = batch_generate_healthy_white_noise(
        models_base_dir="01_models",
        results_base_dir="03_simulation_results",
        dt=0.01,
        duration=60.0,
        rms_g=0.005,
        seed_wn=1
    )

