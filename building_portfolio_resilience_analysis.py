#%%
import numpy as np
from scipy.stats import norm
import os
out_dir = "./04_resilience_plots_04"
os.makedirs(out_dir, exist_ok=True)
import matplotlib.pyplot as plt

# 1. 基础参数设置
np.random.seed(42)  # 固定随机种子以便复现

# =========================================================
# 基础参数
# =========================================================
IM_true       = 0.4
sigma_gmpe    = 0.4
medians = [0.57, 0.97, 2.00]
betas   = [0.56, 0.58, 0.79]

N_total       = 80
N_hospital    = 8
w_hospital    = 5.0
w_residential = 1.0

TIME_HORIZON  = 200   # 评估时间窗口（天）
N_TIME_STEPS  = 500
N_INSPECT_CREWS = 3    # 巡检队伍数量（并行资源约束）
N_REPAIR_CREWS = 5    # 修复队伍数量（并行资源约束）

# SSHM 安装配置：哪些建筑装了传感器
# 可自由修改，True=已安装，False=未安装
sshm_installed = np.zeros(N_total, dtype=bool)
sshm_installed[:] = True
# sshm_installed[:N_hospital] = True                         # 全部医院安装
# sshm_installed[N_hospital::2] = True                       # 住宅每隔一栋安装

q_map_hospital    = {0: 1.0, 1: 0.7, 2: 0.3, 3: 0.0}
q_map_residential = {0: 1.0, 1: 0.8, 2: 0.4, 3: 0.0}

# 巡检时间（天）- 按建筑类型和真实DS
insp_time_map = {
    ("hospital",    0): 0.20,
    ("hospital",    1): 0.30,
    ("hospital",    2): 0.40,
    ("hospital",    3): 0.50,
    ("residential", 0): 0.20,
    ("residential", 1): 0.25,
    ("residential", 2): 0.30,
    ("residential", 3): 0.35,
}

# 修复时间对数正态参数 (median_days, beta) - 按真实DS
#   DS=0: 无需修复; DS=1: 轻微; DS=2: 中等; DS=3: 严重
repair_lognorm = {
    0: (0.0,  0.0),   # DS0: 无损坏，修复时间=0
    1: (5.0,  0.4),   # DS1: 中值5天，beta=0.4
    2: (15.0, 0.5),   # DS2: 中值20天，beta=0.5
    3: (35.0, 0.6),   # DS3: 中值60天，beta=0.6
}

confusion_INSPECTION = np.array([
    [0.80, 0.20, 0.00, 0.00], # 真实为DS0时，被预测为0,1,2,3的概率
    [0.15, 0.65, 0.17, 0.03], # 真实为DS1时，被预测为0,1,2,3的概率
    [0.03, 0.17, 0.65, 0.15], # 真实为DS2时，被预测为0,1,2,3的概率
    [0.00, 0.05, 0.15, 0.80]  # 真实为DS3时，被预测为0,1,2,3的概率
])
# 每一行的概率和为 1
confusion_SSHM = np.array([
    [0.90, 0.10, 0.00, 0.00], # 真实为DS0时，被预测为0,1,2,3的概率
    [0.13, 0.66, 0.20, 0.01], # 真实为DS1时，被预测为0,1,2,3的概率
    [0.01, 0.18, 0.71, 0.10], # 真实为DS2时，被预测为0,1,2,3的概率
    [0.00, 0.01, 0.17, 0.82]  # 真实为DS3时，被预测为0,1,2,3的概率
])


# =========================================================
# 基础工具函数
# =========================================================
def get_fragility_probs(im, medians, betas):
    p_exceed = [norm.cdf(np.log(im / m) / b) for m, b in zip(medians, betas)]
    probs = np.array([
        1.0 - p_exceed[0],
        p_exceed[0] - p_exceed[1],
        p_exceed[1] - p_exceed[2],
        p_exceed[2],
    ])
    probs = np.clip(probs, 0, 1)
    return probs / probs.sum()


def get_q(idx, ds):
    """建筑 idx 在 DS=ds 时的功能值 q"""
    return q_map_hospital[ds] if idx < N_hospital else q_map_residential[ds]


def get_weight(idx):
    return w_hospital if idx < N_hospital else w_residential


W_TOTAL = N_hospital * w_hospital + (N_total - N_hospital) * w_residential

def compute_community_functionality(DS_vec):
    """给定所有建筑当前DS，计算社区加权功能Q"""
    DS = np.asarray(DS_vec, dtype=int)
    q  = np.array([get_q(i, DS[i]) for i in range(N_total)])
    w  = np.array([get_weight(i)   for i in range(N_total)])
    return float(np.dot(w, q) / W_TOTAL)


def base_insp_time(idx, ds_true):
    btype = "hospital" if idx < N_hospital else "residential"
    return insp_time_map[(btype, int(ds_true))]


# =========================================================
# 模块 1：修复时间抽样
# =========================================================
def sample_repair_times(DS_true_vec, rng):
    """
    对每栋建筑按其真实DS，从对数正态分布抽样修复时间（天）。
    DS=0 → 修复时间 = 0（无损坏）
    DS>0 → LogNormal(median, beta) 抽样
    返回: repair_times (ndarray, shape=N_total)
    """
    repair_times = np.zeros(N_total)
    for i in range(N_total):
        ds = int(DS_true_vec[i])
        median, beta = repair_lognorm[ds]
        # if ds == 0 or median == 0:
        #     repair_times[i] = 0.0
        # else:
        #     # LogNormal: ln(T) ~ N(ln(median), beta)
        #     ln_t = np.log(median) + beta * rng.standard_normal()
        #     repair_times[i] = float(np.exp(ln_t))
        repair_times[i] = median
    return repair_times


# =========================================================
# 模块 2：修复优先级排序
# =========================================================
def compute_repair_priority_order(DS_hat_post, repair_times_sampled):
    """
    基于巡检后得到的 DS_hat_post 和抽样修复时间，
    计算每栋建筑的修复优先级并排序。

    优先级指标（功能恢复速率）:
        rate_i = ΔQ_i / T_repair_i
    其中：
        ΔQ_i = (w_i / W_total) * (q(DS=0) - q(DS_hat_i))
               即修复后对社区Q的贡献增量
        T_repair_i = 基于 DS_hat_post 对应的修复时间期望值
                    （用 median 作为期望估计，而非抽样值；
                      因为排序时决策者只知道 DS_hat，不知道真实修复时间）

    DS_hat=0 的建筑认为完好，不纳入修复队列。
    返回: ordered_indices（修复顺序，优先级高 → 低）
    """
    priorities = []
    for i in range(N_total):
        ds_hat = int(DS_hat_post[i])
        if ds_hat == 0:
            continue  # 认为完好，跳过

        # 功能增益（基于 DS_hat 估计）
        q_current = get_q(i, ds_hat)
        q_restored = get_q(i, 0)          # 修复至 DS=0
        delta_q = (get_weight(i) / W_TOTAL) * (q_restored - q_current)

        # 预期修复时间（用 median 作为决策依据）
        median_repair, _ = repair_lognorm[ds_hat]
        if median_repair <= 0:
            median_repair = 0.01           # 极短，优先级极高

        rate = delta_q / median_repair     # 功能恢复速率
        priorities.append((i, rate, delta_q, median_repair))

    # 按 rate 降序排列
    priorities.sort(key=lambda x: -x[1])

    ordered_indices = [p[0] for p in priorities]
    return ordered_indices

# %%
# =========================================================
# 模块 3：基于修复队伍资源的社区功能 Q(t) 动态演化
# =========================================================
SSHM_QUICK_REVIEW_SHRINK = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
MISMATCH_PENALTY  = 1.4  # DS高/低估时的修复时间惩罚系数
MISDETECT_PENALTY = 2.0  # DS_hat=0 但 DS_true>0（漏判）的修复时间惩罚系数
MOBILITY_TIME     = 0.5  # 队伍空跑损耗时间（天）：DS_true=0 但 DS_hat>0 时触发

def simulate_Q_dynamics(repair_order, repair_times_true, DS_true_vec,
                        DS_hat_post, t_insp_end, n_crews=N_REPAIR_CREWS):
    """
    模拟修复过程中社区功能 Q(t) 的动态变化。
    修复队列构成：
      ① 主队列 repair_order：DS_hat > 0 的建筑，按功能恢复速率优先级排序
      ② 漏判补充队列：DS_hat == 0 但 DS_true > 0 的建筑
         → 被错误评估为完好，直到主队列修完后才被发现并补入修复
         → 修复时间额外乘以 MISMATCH_PENALTY（发现晚、调度代价）
    修复完成时刻计算规则：
      DS_hat == DS_true          →  rt = repair_times_true[i]
      DS_hat != DS_true（含漏判）  →  rt = repair_times_true[i] × MISMATCH_PENALTY
    返回: (times, Q_curve, resilience, t_repair_done, n_missed, n_mismatch)
      n_missed   : DS_hat=0 但 DS_true>0 的漏判建筑数
      n_mismatch : 所有DS估计偏差的建筑数（含漏判）
    """
    # --- 初始化 ---
    t_repair_done = np.full(N_total, np.inf)
    for i in range(N_total):
        if DS_true_vec[i] == 0:
            t_repair_done[i] = 0.0   # 真实完好，无需修复

    # --- 识别漏判建筑（DS_hat=0 但 DS_true>0）---
    missed_buildings = [
        i for i in range(N_total)
        if int(DS_hat_post[i]) == 0 and int(DS_true_vec[i]) > 0
    ]
    n_missed = len(missed_buildings)

    # --- 并行队伍调度：先执行主队列，再补入漏判队列 ---
    crew_free_at = np.full(n_crews, float(t_insp_end))
    n_mismatch   = 0

    # 完整修复顺序 = 主队列 + 漏判补充队列（附加到末尾）
    full_repair_sequence = list(repair_order) + missed_buildings

    for i in full_repair_sequence:
        ds_hat  = int(DS_hat_post[i])
        ds_true = int(DS_true_vec[i])

        # ── 情形A：DS_true=0，但 DS_hat>0（高估损伤，误判为需修复）──
        # 建筑本身无需修复（t_repair_done 已初始化为0），
        # 但队伍白跑一趟，消耗 MOBILITY_TIME 占用队伍资源
        if ds_true == 0:
            if ds_hat > 0:
                k = int(np.argmin(crew_free_at))
                crew_free_at[k] += MOBILITY_TIME   # 队伍空跑，推迟其空闲时刻
                n_mismatch += 1
            # ds_true=0 且 ds_hat=0：完全正确，无需任何操作
            continue

        # ── 情形B：DS_true>0，正常修复（含误判惩罚）──
        rt_base = repair_times_true[i]   # 由模块1抽样的真实修复时间（>0）

        if ds_hat == 0 and ds_true > 0:
            # 漏判：被误认为完好，排在队列末尾才被发现
            # 惩罚最重：调配严重滞后 + 重新组织资源
            rt_actual   = rt_base * MISDETECT_PENALTY
            n_mismatch += 1
        elif ds_hat != ds_true:
            # 普通误判（高估或低估）：修复时间乘以较轻惩罚系数
            rt_actual   = rt_base * MISMATCH_PENALTY
            n_mismatch += 1
        else:
            rt_actual = rt_base

        # 分配给最早空闲的队伍
        k       = int(np.argmin(crew_free_at))
        t_start = crew_free_at[k]
        t_end   = t_start + rt_actual
        crew_free_at[k]  = t_end
        t_repair_done[i] = t_end

    # --- 构造 Q(t) 曲线 ---
    times   = np.linspace(0, TIME_HORIZON, N_TIME_STEPS)
    Q_curve = np.zeros(N_TIME_STEPS)

    for k, t in enumerate(times):
        DS_t = np.where(t >= t_repair_done, 0, DS_true_vec).astype(int)
        Q_curve[k] = compute_community_functionality(DS_t)

    # resilience = float(np.trapezoid(Q_curve, times) / TIME_HORIZON)
    resilience = float(np.trapezoid(1 - Q_curve, times))

    return times, Q_curve, resilience, t_repair_done, n_missed, n_mismatch


# =========================================================
# 单次完整仿真
# =========================================================
def one_simulation(rng):
    # --- 真实 DS ---
    p_true      = get_fragility_probs(IM_true, medians, betas)
    DS_true_vec = rng.choice([0, 1, 2, 3], size=N_total, p=p_true)
    Q_true      = compute_community_functionality(DS_true_vec)

    # === 模块1：抽样每栋建筑的真实修复时间 ===
    repair_times_true = sample_repair_times(DS_true_vec, rng)

    # --- 初始 DS 估计 ---
    # S1: 含噪声 IM → argmax → 全部建筑同一DS
    im_s1      = float(np.exp(np.log(IM_true) + rng.normal(0.0, sigma_gmpe)))
    ds_s1_init = int(np.argmax(get_fragility_probs(im_s1, medians, betas)))
    DS_s1_init = np.full(N_total, ds_s1_init, dtype=int)

    # S2: 真实 IM → argmax → 全部建筑同一DS（但比S1准）
    ds_s2_init = int(np.argmax(get_fragility_probs(IM_true, medians, betas)))
    DS_s2_init = np.full(N_total, ds_s2_init, dtype=int)

    # S3: 部分 SSHM 感知
    #   已装 SSHM → 从 confusion_SSHM 逐栋抽样
    #   未装 SSHM → 与 S2 相同，用真实 IM argmax 作为初始估计
    DS_s3_init = np.where(
        sshm_installed,
        np.array([rng.choice([0,1,2,3], p=confusion_SSHM[int(ds),:]) for ds in DS_true_vec], dtype=int),
        ds_s2_init   # 未安装 SSHM 的建筑沿用 IM-based 初始估计
    ).astype(int)

    Q_s1_init = compute_community_functionality(DS_s1_init)
    Q_s2_init = compute_community_functionality(DS_s2_init)
    Q_s3_init = compute_community_functionality(DS_s3_init)

    # 预先生成一份“上帝视角”的巡检判定表
    inspection_results_lookup = np.array([rng.choice([0, 1, 2, 3], p=confusion_INSPECTION[int(ds), :]) for ds in DS_true_vec], dtype=int)


    def do_inspection(DS_init, use_sshm_for=None):
        # order       = np.argsort(-DS_init)
        order = np.arange(len(DS_init))  # 生成 [0, 1, 2, ..., N-1]
        DS_hat_post = DS_init.copy()
        T_insp      = 0.0
        for i in order:
            ds_true = int(DS_true_vec[i])
            t_base  = base_insp_time(i, ds_true)

            has_sshm = (use_sshm_for is not None) and bool(use_sshm_for[i])

            if has_sshm: # 已装 SSHM  → 快速复核，保留 SSHM 初始结果
                shrink = SSHM_QUICK_REVIEW_SHRINK[ds_true]
                T_insp += shrink * t_base      # 快速复核，不更新DS
            else: # 需要人工巡检 → 从预生成查找表取结果（三方案共享）
                DS_hat_post[i] = inspection_results_lookup[i]
                T_insp += t_base
        return DS_hat_post, T_insp/N_INSPECT_CREWS

    DS_s1_post, T_insp_s1 = do_inspection(DS_s1_init)
    DS_s2_post, T_insp_s2 = do_inspection(DS_s2_init)
    DS_s3_post, T_insp_s3 = do_inspection(DS_s3_init, use_sshm_for=sshm_installed)

    Q_s1_post = compute_community_functionality(DS_s1_post)
    Q_s2_post = compute_community_functionality(DS_s2_post)
    Q_s3_post = compute_community_functionality(DS_s3_post)

    # === 模块2：修复优先级排序（基于巡检后DS_hat） ===
    order_s1 = compute_repair_priority_order(DS_s1_post, repair_times_true)
    order_s2 = compute_repair_priority_order(DS_s2_post, repair_times_true)
    order_s3 = compute_repair_priority_order(DS_s3_post, repair_times_true)

    # === 模块3：Q(t) 动态演化 ===
    times, Qc_s1, R_s1, td_s1, nmiss_s1, nm_s1 = simulate_Q_dynamics(
        order_s1, repair_times_true, DS_true_vec, DS_s1_post, T_insp_s1)

    _,     Qc_s2, R_s2, td_s2, nmiss_s2, nm_s2 = simulate_Q_dynamics(
        order_s2, repair_times_true, DS_true_vec, DS_s2_post, T_insp_s2)

    _,     Qc_s3, R_s3, td_s3, nmiss_s3, nm_s3 = simulate_Q_dynamics(
        order_s3, repair_times_true, DS_true_vec, DS_s3_post, T_insp_s3)

    return dict(
        Q_true=Q_true,
        Q_s1_init=Q_s1_init, Q_s2_init=Q_s2_init, Q_s3_init=Q_s3_init,
        Q_s1_post=Q_s1_post, Q_s2_post=Q_s2_post, Q_s3_post=Q_s3_post,
        T_insp_s1=T_insp_s1, T_insp_s2=T_insp_s2, T_insp_s3=T_insp_s3,
        R_s1=R_s1, R_s2=R_s2, R_s3=R_s3,
        nm_s1=nm_s1, nm_s2=nm_s2, nm_s3=nm_s3,
        nmiss_s1=nmiss_s1, nmiss_s2=nmiss_s2, nmiss_s3=nmiss_s3,
        times=times, Qc_s1=Qc_s1, Qc_s2=Qc_s2, Qc_s3=Qc_s3,
        DS_true_vec=DS_true_vec,
        repair_times_true=repair_times_true,
    )

# %%
# =========================================================
# Monte Carlo
# =========================================================
n_mc   = 6000
rng_mc = np.random.default_rng(2026)

print(f"Running {n_mc} MC simulations...")
results = [one_simulation(rng_mc) for _ in range(n_mc)]
print("Done.")

def extract(key):
    return np.array([r[key] for r in results])

Q_true_arr = extract("Q_true")
Q_s1_init  = extract("Q_s1_init"); Q_s2_init = extract("Q_s2_init"); Q_s3_init = extract("Q_s3_init")
Q_s1_post  = extract("Q_s1_post"); Q_s2_post = extract("Q_s2_post"); Q_s3_post = extract("Q_s3_post")
T1 = extract("T_insp_s1");  T2 = extract("T_insp_s2");  T3 = extract("T_insp_s3")
R1 = extract("R_s1");       R2 = extract("R_s2");       R3 = extract("R_s3")
NM1 = extract("nm_s1");    NM2 = extract("nm_s2");    NM3 = extract("nm_s3")
NMISS1 = extract("nmiss_s1"); NMISS2 = extract("nmiss_s2"); NMISS3 = extract("nmiss_s3")

# =========================================================
# 汇总统计
# =========================================================
def summary(arr, name):
    print(f"  {name:32s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
          f"  p25={np.percentile(arr,25):.4f}  p75={np.percentile(arr,75):.4f}")

print(f"\n{'='*72}")
print(f"  MC Summary  (n={n_mc},  Horizon={TIME_HORIZON}d,  Crews={N_REPAIR_CREWS})")
print(f"{'='*72}")
summary(Q_true_arr,  "Q_true (post-earthquake)")
print(f"  {'-'*68}")
summary(Q_s1_init, "Q_init  S1 (noisy IM → argmax)")
summary(Q_s2_init, "Q_init  S2 (true  IM → argmax)")
summary(Q_s3_init, "Q_init  S3 (SSHM per-building)")
print(f"  {'-'*68}")
summary(Q_s1_post, "Q_post  S1 (after inspection)")
summary(Q_s2_post, "Q_post  S2 (after inspection)")
summary(Q_s3_post, "Q_post  S3 (after inspection)")
print(f"  {'-'*68}")
summary(T1, "T_inspect  S1 (days)")
summary(T2, "T_inspect  S2 (days)")
summary(T3, "T_inspect  S3 (days,  SSHM skip DS<2)")
print(f"  {'-'*68}")
summary(R1, "Resilience  S1")
summary(R2, "Resilience  S2")
summary(R3, "Resilience  S3")
print(f"  {'-'*68}")
summary(NM1, "DS mismatch count  S1  (penalized bldgs)")
summary(NM2, "DS mismatch count  S2  (penalized bldgs)")
summary(NM3, "DS mismatch count  S3  (penalized bldgs)")
print(f"  {'-'*68}")
summary(NMISS1, "Missed bldgs (DS_hat=0,DS_true>0)  S1")
summary(NMISS2, "Missed bldgs (DS_hat=0,DS_true>0)  S2")
summary(NMISS3, "Missed bldgs (DS_hat=0,DS_true>0)  S3")
print(f"{'='*72}\n")

# %%
# =========================================================
# 绘图
# =========================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})
COLORS = {"S1": "#E74C3C", "S2": "#F39C12", "S3": "#2ECC71"}
LABELS = {"S1": "noSHM", "S2": "S2: Measured IM", "S3": "SHM"}
ALPHA  = 0.60



# ---- (A) 单次示例 Q(t) 曲线（取一个中等损伤的仿真）----
# med_idx = int(np.argsort(np.abs(Q_true_arr - np.median(Q_true_arr)))[0])
# ex = results[med_idx]
ex = results[2]
lor_s1 = ex["R_s1"]
lor_s3 = ex["R_s3"]
print(f"Single simulation (index=2) LoR:")
print(f"  S1 (noSHM): {lor_s1:.4f}")
print(f"  S3 (SHM):   {lor_s3:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.fill_between(ex["times"], ex["Qc_s1"], 1.0, alpha=0.15, color=COLORS["S1"])
# ax.fill_between(ex["times"], ex["Qc_s2"], 1.0, alpha=0.15, color=COLORS["S2"])
ax.fill_between(ex["times"], ex["Qc_s3"], 1.0, alpha=0.15, color=COLORS["S3"])
for s in ["S1","S3"]:
    ax.plot(ex["times"], ex[f"Qc_{s.lower()}"],
             color=COLORS[s], lw=2.2, label=LABELS[s])
# ax.axhline(ex["Q_true"], color="k", ls="--", lw=1.4,
#             label=f"Q_true = {ex['Q_true']:.3f}")
ax.axhline(1.0,          color="gray", ls=":", lw=1.0)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Community Functionality  $Q(t)$")
# ax.set_title("(A)  Example Q(t) Recovery Curve — Single Simulation "
#               f"[Q_true={ex['Q_true']:.3f},  Crews={N_REPAIR_CREWS}]",
#               fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(-2, 202); ax.set_ylim(0.48, 1.05)
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "A_example_Qt.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# %%

# ---- (B) 平均 Q(t) 曲线 + 95% 置信区间（S1 vs S3）----
times = results[0]["times"]
# 收集所有仿真的 Q(t) 曲线矩阵，shape = (n_mc, N_TIME_STEPS)
Qc_s1_all = np.array([r["Qc_s1"] for r in results])
Qc_s3_all = np.array([r["Qc_s3"] for r in results])

fig, ax = plt.subplots(figsize=(6.5, 5))
for s, mat in [("S1", Qc_s1_all), ("S3", Qc_s3_all)]:
    mean   = mat.mean(axis=0)
    ci_low = np.percentile(mat,  2.5, axis=0)
    ci_high= np.percentile(mat, 97.5, axis=0)

    ax.fill_between(times, ci_low, ci_high,
                    alpha=0.20, color=COLORS[s], label=f"{LABELS[s]} 95% CI")
    ax.plot(times, mean,
            color=COLORS[s], lw=2.2, label=f"{LABELS[s]} mean")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Community Functionality  $Q(t)$")
# ax.set_title("Mean Q(t) Recovery Curve with 95% Confidence Interval")
ax.set_xlim(-2, 202)
ax.set_ylim(0.48, 1.05)
ax.legend(loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "B_mean_Qt_CI.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

#%%
# ---- (C) 韧性分布 ----
fig, ax = plt.subplots(figsize=(6.5, 5))
bins = np.linspace(min(R1.min(),R3.min())*0.97, max(R1.max(),R3.max())*1.0, 45)
for s, arr in [("S1",R1),("S3",R3)]:
    ax.hist(arr, bins=bins, density=True, alpha=ALPHA,
             color=COLORS[s], label=f"{LABELS[s]} (mean={arr.mean():.2f}, std={arr.std():.2f})")
    ax.axvline(arr.mean(), color=COLORS[s], lw=2.0, ls="--")
ax.set_xlabel("LoR")
ax.set_ylabel("Density")
ax.legend(loc="lower right"); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "C_resilience_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
#%%
# ---- (D) 巡检时间分布 ----
fig, ax = plt.subplots(figsize=(6.5, 5))
bins_t = np.linspace(0, max(T1.max(),T3.max())*1.05, 45)
for s, arr in [("S1",T1),("S3",T3)]:
    ax.hist(arr, bins=bins_t, density=True, alpha=ALPHA,
             color=COLORS[s], label=f"{LABELS[s]}\n(mean={arr.mean():.2f}, std={arr.std():.2f})")
    ax.axvline(arr.mean(), color=COLORS[s], lw=2.0, ls="--")
ax.set_xlabel("Inspection Time (days)")
ax.set_ylabel("Density")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "D_inspection_time_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ---- (E) Q_init 误差（相对 Q_true） ----
# fig, ax = plt.subplots(figsize=(6, 5))
# err1 = Q_s1_init - Q_true_arr
# err2 = Q_s2_init - Q_true_arr
# err3 = Q_s3_init - Q_true_arr
# bins_e = np.linspace(-0.8, 0.8, 50)
# for s, arr in [("S1",err1),("S3",err3)]:
#     ax.hist(arr, bins=bins_e, density=True, alpha=ALPHA,
#              color=COLORS[s], label=f"{LABELS[s]}\nμ={arr.mean():.2f}")
#     ax.axvline(arr.mean(), color=COLORS[s], lw=2.0, ls="--")
# ax.axvline(0, color="k", lw=1.2, ls=":")
# ax.set_xlabel("Q_init − Q_true (error)")
# ax.set_ylabel("Density")
# # ax.set_title("(D)  Initial Estimation Error", fontweight="bold")
# ax.legend(loc="upper right"); ax.grid(alpha=0.25)
# plt.tight_layout()
# fig.savefig(os.path.join(out_dir, "E_initial_Q_error_distribution.png"), dpi=150, bbox_inches="tight")
# plt.close(fig)


# ---- (F) 均值汇总条形图 ----
# metrics_labels = ["Q_init", "Q_post", "Resilience R"]
# s1_vals = [Q_s1_init.mean(), Q_s1_post.mean(), R1.mean()]
# s2_vals = [Q_s2_init.mean(), Q_s2_post.mean(), R2.mean()]
# s3_vals = [Q_s3_init.mean(), Q_s3_post.mean(), R3.mean()]
# x      = np.arange(len(metrics_labels))
# width  = 0.22
# fig, ax = plt.subplots(figsize=(6, 5))
# for k, (s, vals) in enumerate([("S1",s1_vals),("S2",s2_vals),("S3",s3_vals)]):
#     bars = ax.bar(x + (k-1)*width, vals, width,
#                    color=COLORS[s], alpha=0.80, label=LABELS[s],
#                    edgecolor="white", linewidth=0.8)
#     for bar, v in zip(bars, vals):
#         ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
#                  f"{v:.3f}", ha="center", va="bottom")
# ax.axhline(Q_true_arr.mean(), color="k", ls=":", lw=1.4,
#             label=f"Q_true={Q_true_arr.mean():.3f}")
# ax.set_xticks(x); ax.set_xticklabels(metrics_labels)
# ax.set_ylabel("Value")
# ax.set_ylim(0, 1.10)
# # ax.set_title("(E)  Mean Metrics", fontweight="bold")
# ax.legend()
# ax.grid(alpha=0.25, axis="y")
# plt.tight_layout()
# fig.savefig(os.path.join(out_dir, "F_mean_metrics.png"), dpi=150, bbox_inches="tight")
# plt.close(fig)

#%%
# ---- (G) DS误判数 & 漏判数对比 ----
fig, ax = plt.subplots(figsize=(6.5, 5))
scen_data = [("S1", NM1, NMISS1),
             # ("S2", NM2, NMISS2),
             ("S3", NM3, NMISS3)]
x_f   = np.arange(2)
w_f   = 0.30
bars_mm = ax.bar(x_f - w_f/2,
                  [NM1.mean(), NM3.mean()],
                  w_f, label="Mismatch ",
                             # "(×1.2 penalty)",
                  color=[COLORS[s] for s in ["S1","S3"]],
                  alpha=0.85, edgecolor="white")
bars_ms = ax.bar(x_f + w_f/2,
                  [NMISS1.mean(), NMISS3.mean()],
                  w_f, label="Mis-detect",
                              # "\n→ appended to end of queue"),
                  color=[COLORS[s] for s in ["S1","S3"]],
                  alpha=0.45, edgecolor="gray", linewidth=0.8, hatch="//")
for bars in [bars_mm, bars_ms]:
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, v+0.04,
                 f"{v:.1f}", ha="center", va="bottom")
ax.set_xticks(x_f)
ax.set_xticklabels([LABELS[s] for s in ["S1","S3"]])
ax.set_ylabel("Mean number of buildings")
# ax.set_title(f"(F)  Repair Errors\n(penalty ×{MISMATCH_PENALTY}  |  missed → tail of queue)", fontweight="bold")
ax.set_ylim(0, 25)
# ax.legend(loc="center right"); ax.grid(alpha=0.25, axis="y")
from matplotlib.patches import Patch

legend_elements = [
    # Patch(facecolor=COLORS["S1"], alpha=0.85, edgecolor="white", label=LABELS["S1"]),
    # Patch(facecolor=COLORS["S3"], alpha=0.85, edgecolor="white", label=LABELS["S3"]),
    Patch(facecolor="lightgray", alpha=0.85, edgecolor="white", label="Mismatch"),
    Patch(facecolor="lightgray", alpha=0.45, edgecolor="gray", hatch="//", label="Mis-detect"),
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=True,
    ncol=2
)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "G_repair_errors.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

#%%
# ---- (H) 韧性均值随MC收敛曲线 ----
fig, ax = plt.subplots(figsize=(6.5, 5))
ns = np.arange(100, n_mc+1, 50)
for s, arr in [("S1",R1),("S3",R3)]:
    means = [arr[:n].mean() for n in ns]
    ax.plot(ns, means, color=COLORS[s], lw=1.8, label=LABELS[s])
ax.set_xlabel("Number of MC runs")
ax.set_ylabel("Running mean  Resilience")
# ax.set_title("(G)  MC Convergence of Resilience", fontweight="bold")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "H_MC_convergence.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# fig.suptitle(
#     f"Community Resilience — Three Sensing & Inspection Strategies\n"
#     f"(IM={IM_true}g,  N_buildings={N_total},  Repair crews={N_REPAIR_CREWS},  "
#     f"Horizon={TIME_HORIZON}d,  n_MC={n_mc})",
#     fontweight="bold", y=1.005
# )









# %%
import numpy as np
import matplotlib.pyplot as plt
COLORS = {"S1": "#E74C3C", "S2": "#F39C12", "S3": "#2ECC71"}
# =========================
# 1. Replace with your data
# =========================
pga = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

lor_no_shm_mean = np.array([7.07, 13.01, 20.69, 29.51, 38.43, 46.73])
lor_no_shm_std = np.array([2.55, 3.78, 5.15, 6.27, 6.72, 6.82])

lor_shm_mean = np.array([6.04, 11.43, 18.49, 26.70, 35.21, 43.28])
lor_shm_std = np.array([2.28, 3.44, 4.76, 5.89, 6.47, 6.59])

# ΔLoR (%) = relative reduction compared with no-SHM case
delta_lor_percent = (lor_no_shm_mean - lor_shm_mean) / lor_no_shm_mean * 100

# Or replace this line if you already have ΔLoR (%) data:
# delta_lor_percent = np.array([8.6, 8.7, 9.5, 9.2, 8.1])


# =========================
# 2. Plot
# =========================
fig, ax1 = plt.subplots(figsize=(6.2, 4.2))

# Left y-axis: Mean LoR with std
line1 = ax1.errorbar(
    pga, lor_no_shm_mean,
    yerr=lor_no_shm_std,
    marker='o',
    linestyle='-',
    linewidth=2,
    capsize=8,
    color=COLORS["S1"],
    label='No SHM'
)

line2 = ax1.errorbar(
    pga, lor_shm_mean,
    yerr=lor_shm_std,
    marker='s',
    linestyle='-',
    linewidth=2,
    capsize=8,
    color=COLORS["S3"],
    label='SHM-informed'
)

ax1.set_xlabel('PGA (g)', fontsize=12)
ax1.set_ylabel('Mean LoR', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.35)

# Right y-axis: ΔLoR (%)
ax2 = ax1.twinx()

line3 = ax2.plot(
    pga, delta_lor_percent,
    marker='^',
    markersize=8,
    linestyle='--',
    linewidth=2,
    label=r'$\Delta LoR (\%)$'
)

ax2.set_ylabel(r'LoR reduction (%)', fontsize=13)

# =========================
# 3. Legends
# =========================
lines = [line1, line2, line3[0]]
labels = [l.get_label() for l in lines]

ax1.legend(
    lines, labels,
    loc='upper center',
    frameon=True,
    fontsize=12
)

# =========================
# 4. Axis formatting
# =========================
ax1.set_xticks(pga)
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

# Optional: adjust y limits manually
# ax1.set_ylim(0, 50)
# ax2.set_ylim(0, 20)

plt.tight_layout()
fig.savefig(os.path.join("04_resilience_plots", "Figure13.jpg"), dpi=150, bbox_inches="tight")
plt.show()
# %%
