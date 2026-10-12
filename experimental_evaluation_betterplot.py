"""
experimental_evaluation.py
===========================
用跨种子训练好的模型，在实验测试集上评估性能。
复用现有的可视化函数（metric_comparison / confusion_matrix）。
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import mode as scipy_mode
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix)

# ============================================================
# 配置
# ============================================================
ROOT_DIR      = Path("/cluster/work/ibk_chatzi/huliang/population_root/04_results")
SEEDS         = [123, 456, 789, 1024, 2026]
N_CLASSES     = 4
DS_LABELS     = [f"DS{i}" for i in range(N_CLASSES)]

# 实验测试集路径
EXP_CSV       = Path("/cluster/work/ibk_chatzi/huliang/population_root/experimental_dataset_noDS1_extended.csv")



# 和训练时完全一致的特征列顺序
FEAT_NAME     = "DSF_TOPO_SHM"
# FEAT_NAME     = "DSF_TOPO_NDE"
# FEAT_NAME     = "DSF_TOPO"
# FEAT_NAME     = "DSF12"

FEATURE_COLS  = [
    "DSF_T1_CENT","DSF_T1_MAC","DSF_dF1peak","DSF_T1_AREA",
    "DSF_T2_CENT","DSF_T2_MAC","DSF_dF2peak","DSF_T2_AREA",
    "DSF_KPRX","DSF_RMSratio","DSF_AR4_L2","DSF_Time_Delay",
    "n_story","n_bay","typical_story_h","bay_width",
    "total_height","total_width","col_b","col_h",
    "beam_b","beam_h","fc","fy",
    "q_dead","q_live","q_tributary_width","damping","T1","T2"]
# FEATURE_COLS  = [
#     "DSF_T1_CENT","DSF_T1_MAC","DSF_dF1peak","DSF_T1_AREA",
#     "DSF_T2_CENT","DSF_T2_MAC","DSF_dF2peak","DSF_T2_AREA",
#     "DSF_KPRX","DSF_RMSratio","DSF_AR4_L2","DSF_Time_Delay",
#     "n_story","n_bay","typical_story_h","bay_width",
#     "total_height","total_width","col_b","col_h",
#     "beam_b","beam_h","fc","fy",
#     "q_dead","q_live","q_tributary_width"]
# FEATURE_COLS  = [
#     "DSF_T1_CENT","DSF_T1_MAC","DSF_dF1peak","DSF_T1_AREA",
#     "DSF_T2_CENT","DSF_T2_MAC","DSF_dF2peak","DSF_T2_AREA",
#     "DSF_KPRX","DSF_RMSratio","DSF_AR4_L2","DSF_Time_Delay",
#     "n_story","n_bay","typical_story_h","bay_width",
#     "total_height","total_width","col_b","col_h",
#     "beam_b","beam_h","q_tributary_width"]
# FEATURE_COLS  = [
#     "DSF_T1_CENT","DSF_T1_MAC","DSF_dF1peak","DSF_T1_AREA",
#     "DSF_T2_CENT","DSF_T2_MAC","DSF_dF2peak","DSF_T2_AREA",
#     "DSF_KPRX","DSF_RMSratio","DSF_AR4_L2","DSF_Time_Delay"]

LABEL_COL     = "damage_state"


# TRUE_DS_MAP   = {1: 0, 17: 0, 23: 2, 29: 3}   # case_num -> 真实DS
# CASE_ORDER    = [1, 17, 23, 29]

TRUE_DS_MAP   = {17: 0, 23: 2, 29: 3}   # case_num -> 真实DS
CASE_ORDER    = [17, 23, 29]

DS_COLORS     = ['#C0C0C0', '#92D050', '#FFC000', '#C0504D']
DS_HATCHES    = ['', '///', '...', 'xxx']

NATURE_COLORS = ['#E64B35','#4DBBD5','#00A087','#3C5488',
                 '#F39B7F','#8491B4','#91D1C2','#DC0000']


# ============================================================
# Step 1: 加载实验测试集
# ============================================================

def load_experimental_data(csv_path: Path):
    df = pd.read_csv(csv_path)

    # 过滤无效行
    df = df[df["valid_flag"] == 1].copy()

    # 检查是否有 NaN（时域DSF可能有）
    missing = df[FEATURE_COLS].isnull().sum()
    if missing.any():
        print("[WARN] 以下特征存在 NaN，将用列中位数填充：")
        print(missing[missing > 0])
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    print(f"\n✅ 实验测试集加载完毕: {len(df)} 条样本")
    print("   类别分布:")
    for ds in range(N_CLASSES):
        n = (y == ds).sum()
        print(f"     DS{ds}: {n} ({100*n/len(y):.1f}%)")

    return X, y, df


# ============================================================
# Step 2: 加载各 seed 的模型并推理
# ============================================================

def evaluate_all_seeds(root_dir: Path, feat_name: str,
                       seeds: list, X: np.ndarray, y: np.ndarray) -> dict:
    seed_data = {}
    EXCLUDE_MODELS = {'LGBM', 'LightGBM'}  # ← 加这一行

    for seed in seeds:
        folder     = root_dir / f"{feat_name}_seed{seed}"
        pipes_path = folder / "trained_pipes.pkl"   # ← 直接找 trained_pipes.pkl

        if not pipes_path.exists():
            print(f"⚠️  找不到: {pipes_path}，跳过")
            continue

        trained_pipes = joblib.load(pipes_path)   # dict: {model_name: pipeline}
        print(f"   Seed {seed}: 加载了 {list(trained_pipes.keys())}")

        for model_name, pipe in trained_pipes.items():
            if model_name in EXCLUDE_MODELS:
                continue
            y_pred = pipe.predict(X)              # pipeline 内含 scaler，直接用原始 X

            if hasattr(pipe, "predict_proba"):
                y_prob = pipe.predict_proba(X)
                try:
                    auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
                except Exception:
                    auc = float("nan")
            else:
                auc = float("nan")

            result = {
                "seed"    : seed,
                "acc"     : accuracy_score(y, y_pred),
                "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
                "auc_ovr" : auc,
                "y_pred"  : y_pred.tolist(),
                "y_true"  : y.tolist(),
            }

            seed_data.setdefault(model_name, []).append(result)

    print(f"\n✅ 实验集评估完成，共 {len(seed_data)} 个模型")
    return seed_data


# ============================================================
# Step 3: 收集逐窗口 predict_proba（所有 seed 平均）
# ============================================================

def collect_window_probas(root_dir: Path, feat_name: str,
                          seeds: list, X: np.ndarray,
                          df_exp: pd.DataFrame) -> dict:
    """
    返回:
        proba_dict[model_name][case_num] = mean_proba  shape [n_windows, 4]
        mean 是对所有 seed 取平均
    """
    # {model_name: {case_num: list of [n_win, 4]}}
    accum = {}
    EXCLUDE_MODELS = {'LGBM', 'LightGBM'}

    # 自动找排序列
    sort_col = next((c for c in ["win_start_idx", "win_start_sec", "sample_id"]
                     if c in df_exp.columns), None)
    if sort_col is None:
        sort_col = df_exp.columns[0]
        print(f"[WARN] 找不到时间列，用 {sort_col} 排序")


    for seed in seeds:
        pipes_path = root_dir / f"{feat_name}_seed{seed}" / "trained_pipes.pkl"
        if not pipes_path.exists():
            continue
        trained_pipes = joblib.load(pipes_path)

        for model_name, pipe in trained_pipes.items():
            if model_name in EXCLUDE_MODELS:
                continue
            if not hasattr(pipe, "predict_proba"):
                continue

            for case_num in CASE_ORDER:
                sub = df_exp[df_exp["case_num"] == case_num] \
                             .sort_values(sort_col)
                X_sub = sub[FEATURE_COLS].values
                proba = pipe.predict_proba(X_sub)   # [n_win, 4]
                accum \
                    .setdefault(model_name, {}) \
                    .setdefault(case_num, []) \
                    .append(proba)

    # 对每个 (model, case) 取 seed 均值
    proba_dict = {}
    for model_name, case_dict in accum.items():
        proba_dict[model_name] = {}
        for case_num, proba_list in case_dict.items():
            proba_dict[model_name][case_num] = np.mean(proba_list, axis=0)

    return proba_dict


# ============================================================
# Step 4: 窗口级概率分布图（堆叠条形）
# ============================================================

def plot_window_prob_distribution(proba_dict: dict,
                                  df_exp: pd.DataFrame,
                                  model_name: str,
                                  save_dir: Path):
    """
    每个工况一个子图，横轴=窗口序号，纵轴=各DS预测概率（堆叠）。
    底部状态带显示跨种子平均概率的 argmax；标题显示工况级决策。
    """
    if model_name not in proba_dict:
        print(f"⚠️  {model_name} 没有概率数据，跳过")
        return
    # case_num -> 显示名称
    CASE_LABEL = {1: "TS1", 17: "TS4", 23: "TS6", 29: "TS8"}

    fig = plt.figure(figsize=(11, 4.0 * len(CASE_ORDER)))
    case_grid = fig.add_gridspec(len(CASE_ORDER), 1, hspace=0.50,
                                left=0.11, right=0.99, bottom=0.07, top=0.94)
    axes, prediction_axes = [], []
    for index in range(len(CASE_ORDER)):
        panel = case_grid[index].subgridspec(2, 1, height_ratios=[8, 1], hspace=0.08)
        ax = fig.add_subplot(panel[0])
        prediction_ax = fig.add_subplot(panel[1], sharex=ax)
        axes.append(ax)
        prediction_axes.append(prediction_ax)
    # fig.suptitle(
    #     f"Window-level predicted probability — {model_name}\n"
    #     f"(mean over {len(SEEDS)} seeds)",
    #     fontsize=12, fontweight="bold"
    # )

    for ax, prediction_ax, case_num in zip(axes, prediction_axes, CASE_ORDER):

        mean_prob = proba_dict[model_name][case_num]   # [n_win, 4]
        n_win     = mean_prob.shape[0]
        x         = np.arange(n_win)
        true_ds   = TRUE_DS_MAP[case_num]

        ax.tick_params(axis='both', which='major', labelsize=15)

        # 堆叠条形
        bottom = np.zeros(n_win)
        for di in range(N_CLASSES):
            ax.bar(x, mean_prob[:, di], bottom=bottom,
                   color=DS_COLORS[di], label=DS_LABELS[di],
                   hatch=DS_HATCHES[di], edgecolor='black',
                   width=0.85, alpha=0.88, linewidth=0.4)
            bottom += mean_prob[:, di]

        # 0.5 基准线
        ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.35)

        # 工况级融合：概率平均后 argmax
        avg_prob  = mean_prob.mean(axis=0)
        voted_ds  = int(np.argmax(avg_prob))
        correct   = (voted_ds == true_ds)
        tick_sym  = "✓" if correct else "✗"

        ax.set_ylim(0, 1.08)
        ax.set_xlim(-0.6, n_win - 0.4)
        ax.set_ylabel("Probability", fontsize=15)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        window_ds = np.argmax(mean_prob, axis=1)
        label_size = 10 if n_win <= 20 else 8
        for window, ds in enumerate(window_ds):
            prediction_ax.bar(window, 1.0, width=0.85,
                              color=DS_COLORS[ds], hatch=DS_HATCHES[ds],
                              edgecolor='black', linewidth=0.4, alpha=0.88)
            prediction_ax.text(window, 0.5, DS_LABELS[ds],
                               ha='center', va='center', fontsize=label_size,
                               bbox=dict(facecolor='white', edgecolor='none', pad=0.1),
                               zorder=5)
        prediction_ax.set_ylim(0, 1)
        prediction_ax.set_yticks([])
        prediction_ax.set_ylabel("Pred DS", rotation=0, fontsize=11,
                                 ha='right', va='center', labelpad=8)
        tick_step = max(1, (n_win + 19) // 20)
        prediction_ax.set_xticks(x[::tick_step])
        prediction_ax.set_xticklabels(
            [f"W{i+1}" for i in x[::tick_step]], fontsize=12)
        prediction_ax.tick_params(axis='x', length=0, pad=5)
        for spine in prediction_ax.spines.values():
            spine.set_visible(False)


        # 标题：真实 / 投票 / 正确与否
        # color_title = '#2ca02c' if correct else '#d62728'
        case_label  = CASE_LABEL[case_num]
        ax.set_title(
            f"{case_label}  |  Experimental label: DS{true_ds}  |  "
            f"Case-level Vote: DS{voted_ds}  {tick_sym}",
            fontsize=15, color='k', fontweight="bold", pad=8
        )

        # 平均概率文字注释（右下角）
        avg_txt = "  ".join([f"{DS_LABELS[i]}: {avg_prob[i]:.1%}"
                              for i in range(N_CLASSES)])
        ax.text(0.99, 0.04, avg_txt,
                transform=ax.transAxes, fontsize=14,
                ha='right', va='bottom',
                color='k',
                bbox=dict(facecolor='white', edgecolor='0.6',
                          linewidth=0.6, boxstyle='square,pad=0.3', alpha=1.0),
                zorder=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 统一图例（只在第一个子图显示）
    patches = [mpatches.Patch(facecolor=DS_COLORS[i], edgecolor='black',
                             hatch=DS_HATCHES[i], label=DS_LABELS[i])
               for i in range(N_CLASSES)]
    # axes[0].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.0, 1.05),
    #                fontsize=14, ncol=N_CLASSES, framealpha=0.6)
    patches = [mpatches.Patch(facecolor=DS_COLORS[i], edgecolor='black',
                             hatch=DS_HATCHES[i], label=DS_LABELS[i])
           for i in range(N_CLASSES)]
    fig.legend(handles=patches, loc='upper center', 
            bbox_to_anchor=(0.5, 1.01),        # ← 图形顶部中央，略微超出
            fontsize=13, ncol=N_CLASSES, 
            framealpha=0.6)
    prediction_axes[-1].set_xlabel("Window index", fontsize=15)
    save_path = save_dir / f"window_prob_dist_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 窗口级概率分布图已保存: {save_path}")

# ============================================================
# Step 5: 工况级融合决策汇总（所有模型）
# ============================================================
def plot_case_level_summary(proba_dict: dict,
                             summary_df: pd.DataFrame,
                             save_dir: Path):
    """
    横轴=工况，纵轴=各DS平均预测概率（分组条形图）。
    每个模型一行子图，按 F1 降序排列。
    """
    model_order = summary_df.index.tolist()
    # 只保留有概率数据的模型
    model_order = [m for m in model_order if m in proba_dict]

    n_models = len(model_order)
    fig, axes = plt.subplots(n_models, 1,
                              figsize=(10, 3.5 * n_models))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Case-level averaged probability (all models, F1-ranked)",
                 fontsize=12, fontweight="bold")

    bar_w   = 0.18
    x_case  = np.arange(len(CASE_ORDER))

    for ax, model_name in zip(axes, model_order):
        for di in range(N_CLASSES):
            offsets = (di - 1.5) * bar_w
            heights = [proba_dict[model_name][c].mean(axis=0)[di]
                       for c in CASE_ORDER]
            ax.bar(x_case + offsets, heights,
                   width=bar_w, color=DS_COLORS[di],
                   hatch=DS_HATCHES[di], edgecolor='black',
                   label=DS_LABELS[di], alpha=0.85, linewidth=0.4)

        # 真实标签竖线标注
        for xi, case_num in enumerate(CASE_ORDER):
            true_di = TRUE_DS_MAP[case_num]
            ax.axvline(xi + (true_di - 1.5) * bar_w,
                       color='k', lw=1.5, ls='--', alpha=0.6)

        f1_m = summary_df.loc[model_name, "F1_mean"]
        f1_s = summary_df.loc[model_name, "F1_std"]
        ax.set_title(f"{model_name}   F1={f1_m:.3f}±{f1_s:.3f}",
                     fontsize=10)
        ax.set_xticks(x_case)
        ax.set_xticklabels(
            [f"Case {c}\n(True DS{TRUE_DS_MAP[c]})" for c in CASE_ORDER],
            fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Avg probability", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(facecolor=DS_COLORS[i], edgecolor='black',
                             hatch=DS_HATCHES[i], label=DS_LABELS[i])
               for i in range(N_CLASSES)]
    axes[0].legend(handles=patches, loc='upper right',
                   fontsize=9, ncol=N_CLASSES, framealpha=0.6,
                   title="Predicted DS  (-- = true DS bar)")
    axes[-1].set_xlabel("Experimental case", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = save_dir / "case_level_summary_all_models.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 工况级汇总图已保存: {save_path}")




# ============================================================
# Step 3: 保存每个窗口的预测结果（方便后续分析）
# ============================================================
def save_per_window_predictions(root_dir: Path, feat_name: str,
                                seeds: list, X: np.ndarray,
                                df_exp: pd.DataFrame, save_dir: Path):
    """
    对每个模型，把 5 个 seed 的预测结果（投票/平均）保存为 CSV，
    方便按 case_num / win_start_sec 可视化预测时序。
    """
    all_preds = {}   # {model_name: [pred_seed1, pred_seed2, ...]}
    EXCLUDE_MODELS = {'LGBM', 'LightGBM'}

    for seed in seeds:
        pipes_path = root_dir / f"{feat_name}_seed{seed}" / "trained_pipes.pkl"
        if not pipes_path.exists():
            continue
        trained_pipes = joblib.load(pipes_path)
        for model_name, pipe in trained_pipes.items():
            y_pred = pipe.predict(X)
            all_preds.setdefault(model_name, []).append(y_pred)

    out_df = df_exp[["sample_id","case_num","win_start_sec",
                      "damage_state"]].copy()

    for model_name, preds in all_preds.items():
        if model_name in EXCLUDE_MODELS:
            continue
        preds_arr = np.array(preds)   # [n_seeds x n_samples]
        # 多数投票
        from scipy.stats import mode as scipy_mode
        voted, _ = scipy_mode(preds_arr, axis=0, keepdims=False)
        out_df[f"pred_{model_name}"] = voted.astype(int)

    out_path = save_dir / "per_window_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ 逐窗口预测结果已保存: {out_path}")
    return out_df


# ============================================================
# Step 4: 时序预测可视化（实验集专属）
# ============================================================

def plot_prediction_timeline(pred_df: pd.DataFrame,
                              model_name: str, save_dir: Path):
    """
    按 case_num 分子图，横轴=时间，纵轴=预测DS，
    与真实标签对比。
    """
    pred_col = f"pred_{model_name}"
    if pred_col not in pred_df.columns:
        print(f"⚠️  {pred_col} 不在 DataFrame 中")
        return

    cases = sorted(pred_df["case_num"].unique())
    fig, axes = plt.subplots(len(cases), 1,
                              figsize=(14, 3 * len(cases)), sharex=False)
    if len(cases) == 1:
        axes = [axes]

    colors_ds = ['#00A087','#4DBBD5','#F39B7F','#E64B35']

    for ax, case in zip(axes, cases):
        sub = pred_df[pred_df["case_num"] == case].sort_values("win_start_sec")
        t   = sub["win_start_sec"].values
        y_t = sub["damage_state"].values
        y_p = sub[pred_col].values

        ax.step(t, y_t, where="post", color="black",
                linewidth=1.5, label="True DS", linestyle="--")
        ax.step(t, y_p, where="post",
                color=colors_ds[int(y_t[0])],
                linewidth=1.5, label=f"Pred ({model_name})", alpha=0.8)

        # 背景色标注真实DS区间
        ax.fill_between(t, y_t - 0.05, y_t + 0.05,
                         alpha=0.12, step="post",
                         color=colors_ds[int(y_t[0])])

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(DS_LABELS)
        ax.set_ylabel("Damage State")
        ax.set_title(f"Case {case}  (True DS={y_t[0]})", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Per-Window Predictions: {model_name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = save_dir / f"timeline_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 时序预测图已保存: {save_path}")


# ============================================================
# 主流程（直接复用你现有的可视化函数）
# ============================================================

def main():
    save_dir = ROOT_DIR / f"{FEAT_NAME}_experimental_eval_window"
    save_dir.mkdir(exist_ok=True)

    # ---------- 加载实验数据 ----------
    X, y, df_exp = load_experimental_data(EXP_CSV)
    print("[DEBUG] df_exp columns:", df_exp.columns.tolist())
    # ---------- 推理 ----------
    seed_data = evaluate_all_seeds(ROOT_DIR, FEAT_NAME, SEEDS, X, y)
    if not seed_data:
        print("❌ 没有可用模型，请检查路径")
        return

    # ---------- 保存原始推理结果 ----------
    json_path = save_dir / "experimental_all_results.json"
    # 转成可序列化格式
    serializable = {
        m: [{"seed": r["seed"], "acc": r["acc"],
             "f1_macro": r["f1_macro"], "auc_ovr": r["auc_ovr"],
             "y_pred": r["y_pred"], "y_true": r["y_true"]}
            for r in rs]
        for m, rs in seed_data.items()
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ 推理结果已保存: {json_path}")

    # ---------- 复用你现有的统计 + 可视化函数 ----------
    # （直接从你的原脚本 import，或者把函数粘贴过来）
    from generate_aggregated_results_betterplot import (compute_summary_stats,
                                    compute_mean_confusion_matrices,
                                    plot_metric_comparison,
                                    plot_confusion_matrices,
                                    plot_best_model_cm_detail,
                                    save_summary_table)

    summary_df = compute_summary_stats(seed_data)
    mean_cms   = compute_mean_confusion_matrices(seed_data, N_CLASSES)

    plot_metric_comparison(summary_df, save_dir)
    plot_confusion_matrices(mean_cms, summary_df, DS_LABELS, save_dir)
    plot_best_model_cm_detail(mean_cms, seed_data, summary_df, DS_LABELS, save_dir)
    save_summary_table(summary_df, save_dir)

    # 5. 收集逐窗口概率（新增）
    print("\n[INFO] 收集逐窗口 predict_proba ...")
    proba_dict = collect_window_probas(ROOT_DIR, FEAT_NAME, SEEDS, X, df_exp)

    # ── 新增：计算6个模型的集成平均 ──────────────────────────────
    ensemble_proba = {}
    for case_num in CASE_ORDER:
        all_model_probas = np.array([
            proba_dict[m][case_num]
            for m in proba_dict
            if case_num in proba_dict[m]
        ])  # [n_models, n_windows, 4]
        ensemble_proba[case_num] = all_model_probas.mean(axis=0)  # [n_windows, 4]

    proba_dict["Ensemble"] = ensemble_proba
    # ─────────────────────────────────────────────────────────────

    # 6. 窗口级概率分布图 —— 每个模型 + 集成
    for model_name in list(summary_df.index) + ["Ensemble"]:
        plot_window_prob_distribution(proba_dict, df_exp, model_name, save_dir)

    # # 6. 窗口级概率分布图
    # for model_name in summary_df.index:
    #     plot_window_prob_distribution(proba_dict, df_exp, model_name, save_dir)

    # 7. 工况级汇总图（所有模型）
    plot_case_level_summary(proba_dict, summary_df, save_dir)
    

    # ---------- 实验集专属：逐窗口时序图 ----------
    pred_df    = save_per_window_predictions(
                     ROOT_DIR, FEAT_NAME, SEEDS, X, df_exp, save_dir)
    best_model = summary_df["F1_mean"].idxmax() 
    plot_prediction_timeline(pred_df, best_model, save_dir)

    print(f"\n🎉 完成！结果保存在: {save_dir.resolve()}")
    return summary_df, mean_cms


if __name__ == "__main__":
    summary_df, mean_cms = main()
