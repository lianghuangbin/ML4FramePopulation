from __future__ import annotations
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from openseespy.opensees import *


# =========================================================
# 0. 随机参数生成
# =========================================================
def generate_random_2d_frame_params(seed=None):
    """
    随机生成一个 2D RC frame 的参数字典
    单位：m, N, Pa, kg
    """
    rng = np.random.default_rng(seed)

    n_story = int(rng.integers(3, 7))   # 3~6层
    # n_bay   = int(rng.integers(2, 5))   # 2~4跨
    n_bay   = int(rng.integers(1, 4))   # 1~3跨      

    first_story_h = float(rng.uniform(3.3, 4.2))
    typical_story_h = float(rng.uniform(3.0, 3.6))
    bay_width = float(rng.uniform(4.5, 7.0))             # 5.0，7.0

    # 梁柱截面（先做全楼统一；后面你可以改成分层变化）
    col_b = float(rng.uniform(0.45, 0.70))              # 0.45, 0.60
    col_h = float(rng.uniform(0.45, 0.70))              # 0.45, 0.60

    beam_b = float(rng.uniform(0.25, 0.40))             # 0.25, 0.35
    beam_h = float(rng.uniform(0.45, 0.75))             # 0.45, 0.60

    # 材料
    fc = -float(rng.uniform(25e6, 40e6))     # 30~35 MPa, 注意 OpenSees 里混凝土强度是负值
    fy = float(rng.uniform(300e6, 400e6))    # 300~350 MPa
    Ec = float(4700.0 * math.sqrt(abs(fc) / 1e6) * 1e6)  # Pa
    Es = 200e9

    # 保护层
    cover = 0.04

    # 配筋（先采用简单统一布置）
    col_rebar = {
        "top":   {"n": 4, "d": 0.018},
        "bot":   {"n": 4, "d": 0.018},
        "left":  {"n": 2, "d": 0.016},
        "right": {"n": 2, "d": 0.016},
    }

    beam_rebar = {
        "top":   {"n": 4, "d": 0.016},
        "bot":   {"n": 4, "d": 0.016},
        "left":  {"n": 2, "d": 0.012},
        "right": {"n": 2, "d": 0.012},
    }

    # 荷载（面荷载转线荷载时，你后面可再细化）
    q_dead = float(rng.uniform(3.0e3, 6.0e3))   # 4.0~5.0 kN/m^2
    q_live = float(rng.uniform(1.0e3, 3.0e3))   # 1.5~2.5 kN/m^2
    # tributary_width = float(rng.uniform(4.0, 6.0))  # 2D框架对应的等效楼板分担宽度（m）
    tributary_width = bay_width  # 2D框架对应的等效楼板分担宽度（m）

    # 阻尼
    zeta = float(rng.uniform(0.045, 0.055))     # 0.05

    # 非线性积分/纤维离散
    nl = {
        "nIP_beam": 5,
        "nIP_col":  5,
        "nfCoreY": 10,
        "nfCoreZ": 10,
        "nfCoverY": 4,
        "nfCoverZ": 4,
    }

    params = {
        "n_story": n_story,
        "n_bay": n_bay,
        "first_story_h": first_story_h,
        "typical_story_h": typical_story_h,
        "bay_width": bay_width,
        "col": {
            "b": col_b,
            "h": col_h,
            "cover": cover,
            "rebar": col_rebar,
        },
        "beam": {
            "b": beam_b,
            "h": beam_h,
            "cover": cover,
            "rebar": beam_rebar,
        },
        "mat": {
            "fc": fc,
            "Ec": Ec,
            "epsc0": -0.002,
            "fcu": 0.2 * fc,        # 例如残余强度取 20% fc（仍为负）
            "epsU": -0.006,
            "fy": fy,
            "Es": Es,
            "b_kin": 0.02,
            "epsU_steel": 0.12,
        },
        "loads": {
            "q_dead": q_dead,
            "q_live": q_live,
            "tributary_width": tributary_width,
        },
        "damping": {
            "zeta": zeta
        },
        "nl": nl
    }
    return params


# =========================================================
# 1. 生成 2D 框架节点和单元
# =========================================================
def build_2d_frame_nodes_elements(params):
    """
    自动生成 2D 框架节点、梁柱单元、楼层节点集、屋顶控制节点、质量信息
    坐标:
      x: 水平
      y: 竖向
    """
    n_story = params["n_story"]
    n_bay = params["n_bay"]
    bay_width = params["bay_width"]
    h1 = params["typical_story_h"]
    # h1 = params["first_story_h"]
    ht = params["typical_story_h"]

    # 楼层标高
    y_levels = [0.0]
    for i in range(1, n_story + 1):
        if i == 1:
            y_levels.append(h1)
        else:
            y_levels.append(h1 + (i - 1) * ht)

    # 网格线
    x_grids = [i * bay_width for i in range(n_bay + 1)]

    node_list = {}        # nodeTag -> (x, y)
    story_nodes = {}      # story index -> [node tags]
    story_master = {}     # story index -> master node tag (2D里可取中间或屋顶中点最近节点)
    element_list = {}     # eleTag -> dict

    # ---------- 建节点 ----------
    # 节点编号: 1000*story + grid
    for k in range(n_story + 1):
        story_nodes[k] = []
        y = y_levels[k]
        for i, x in enumerate(x_grids):
            nd = k * 1000 + i + 1
            node_list[nd] = (x, y)
            story_nodes[k].append(nd)

    # 每层 master 节点：取几何中间最接近的那个节点
    x_mid = 0.5 * (x_grids[0] + x_grids[-1])
    for k in range(1, n_story + 1):
        nodes_k = story_nodes[k]
        master = min(nodes_k, key=lambda nd: abs(node_list[nd][0] - x_mid))
        story_master[k] = master

    # ---------- 建单元 ----------
    eleTag = 1
    # 柱
    for k in range(n_story):
        for i in range(n_bay + 1):
            ni = story_nodes[k][i]
            nj = story_nodes[k + 1][i]
            element_list[eleTag] = {
                "type": "column",
                "i": ni,
                "j": nj,
                "sec_cfg": copy.deepcopy(params["col"]),
                "mat_cfg": copy.deepcopy(params["mat"]),
                "nl_cfg": copy.deepcopy(params["nl"]),
            }
            eleTag += 1

    # 梁
    for k in range(1, n_story + 1):
        for i in range(n_bay):
            ni = story_nodes[k][i]
            nj = story_nodes[k][i + 1]
            element_list[eleTag] = {
                "type": "beam",
                "i": ni,
                "j": nj,
                "sec_cfg": copy.deepcopy(params["beam"]),
                "mat_cfg": copy.deepcopy(params["mat"]),
                "nl_cfg": copy.deepcopy(params["nl"]),
            }
            eleTag += 1

    # ---------- 质量 ----------
    # 用等效楼面荷载 * 楼面面积 / g -> 质量
    q_dead = params["loads"]["q_dead"]
    q_live = params["loads"]["q_live"]
    tributary_width = params["loads"]["tributary_width"]

    g = 9.81
    total_floor_length = n_bay * bay_width
    floor_area_2d_equiv = total_floor_length * tributary_width

    # 每层总质量
    story_mass = {}
    for k in range(1, n_story + 1):
        Wk = (q_dead + 0.6 * q_live) * floor_area_2d_equiv  # N
        Mk = Wk / g  # kg
        story_mass[k] = Mk

    roof_ctrl_node = story_master[n_story]

    return node_list, element_list, story_nodes, story_master, story_mass, roof_ctrl_node, y_levels


# =========================================================
# 2. 材料定义
# =========================================================
# def define_uniaxial_materials(mat_cfg, start_tag=1):
#     """
#     定义混凝土和钢筋材料，返回材料tag
#     """
#     fc = mat_cfg["fc"]
#     Ec = mat_cfg["Ec"]
#     epsc0 = mat_cfg["epsc0"]
#     fcu = mat_cfg["fcu"]
#     epsU = mat_cfg["epsU"]
#     fy = mat_cfg["fy"]
#     Es = mat_cfg["Es"]
#     b_kin = mat_cfg["b_kin"]
#
#     # 非受约束混凝土
#     conc_cover_tag = start_tag
#     uniaxialMaterial("Concrete01", conc_cover_tag, fc, epsc0, fcu, epsU)
#
#     # 简单受约束混凝土（先用增强版本；你后面可替换 Mander 模型）
#     conc_core_tag = start_tag + 1
#     fc_core = 1.2 * fc
#     epsc0_core = 1.2 * epsc0
#     fcu_core = 1.2 * fcu
#     epsU_core = 1.5 * epsU
#     uniaxialMaterial("Concrete01", conc_core_tag, fc_core, epsc0_core, fcu_core, epsU_core)
#
#     # 钢筋
#     steel_tag = start_tag + 2
#     uniaxialMaterial("Steel02", steel_tag, fy, Es, b_kin)
#
#     tags = {
#         "conc_cover": conc_cover_tag,
#         "conc_core": conc_core_tag,
#         "steel": steel_tag
#     }
#     return tags

def define_uniaxial_materials(mat_cfg, start_tag=1):
    fc = mat_cfg["fc"]          # negative
    Ec = mat_cfg["Ec"]
    epsc0 = mat_cfg["epsc0"]
    fcu = mat_cfg["fcu"]        # negative
    epsU = mat_cfg["epsU"]
    fy = mat_cfg["fy"]
    Es = mat_cfg["Es"]
    b_kin = mat_cfg["b_kin"]

    # 建议新增参数
    lam = mat_cfg.get("lam", 0.1)
    ft = mat_cfg.get("ft", 0.01 * abs(fc))
    Ets = mat_cfg.get("Ets", 0.01 * Ec)     # 一个初始近似
    epsU_steel = mat_cfg.get("epsU_steel", 0.08)

    conc_cover_tag = start_tag
    uniaxialMaterial("Concrete02", conc_cover_tag, fc * 0.8, epsc0 * 0.8, fcu * 0.8, epsU * 0.8, lam, ft * 0.8, Ets * 0.8)

    conc_core_tag = start_tag + 1
    uniaxialMaterial("Concrete02", conc_core_tag, fc, epsc0, fcu, epsU, lam, ft, Ets)

    steel_core_tag  = start_tag + 2
    uniaxialMaterial("Steel02", steel_core_tag, fy, Es, b_kin, 18.0, 0.925, 0.15)

    steel_tag = start_tag + 3
    uniaxialMaterial("MinMax", steel_tag, steel_core_tag, "-min", -epsU_steel, "-max", epsU_steel)

    tags = {
        "conc_cover": conc_cover_tag,
        "conc_core": conc_core_tag,
        "steel": steel_tag
    }
    return tags

# =========================================================
# 3. RC fiber section
# =========================================================
def bar_area(d):
    return math.pi * d**2 / 4.0


def make_rc_fiber_section_2d(secTag, sec_cfg, mat_tags, nfCoreY=10, nfCoreZ=10, nfCoverY=4, nfCoverZ=4):
    """
    生成矩形 RC 截面 fiber section
    注意：
      OpenSees fiber section 的坐标是局部(y,z)平面
      对2D frame，弯曲关于 z 轴，section 仍按 y-z 离散
    """
    b = float(sec_cfg["b"])
    h = float(sec_cfg["h"])
    cover = float(sec_cfg["cover"])
    reb = sec_cfg["rebar"]

    conc_cover = mat_tags["conc_cover"]
    conc_core = mat_tags["conc_core"]
    steel = mat_tags["steel"]

    y1 = -h / 2.0
    y2 =  h / 2.0
    z1 = -b / 2.0
    z2 =  b / 2.0

    yc1 = y1 + cover
    yc2 = y2 - cover
    zc1 = z1 + cover
    zc2 = z2 - cover

    section("Fiber", secTag)

    # ----- core -----
    patch("rect", conc_core, nfCoreY, nfCoreZ, yc1, zc1, yc2, zc2)

    # ----- cover -----
    # bottom cover
    patch("rect", conc_cover, nfCoverY, nfCoreZ, y1, zc1, yc1, zc2)
    # top cover
    patch("rect", conc_cover, nfCoverY, nfCoreZ, yc2, zc1, y2, zc2)
    # left cover
    patch("rect", conc_cover, nfCoreY, nfCoverZ, yc1, z1, yc2, zc1)
    # right cover
    patch("rect", conc_cover, nfCoreY, nfCoverZ, yc1, zc2, yc2, z2)

    # ----- rebars -----
    # top bars
    A_top = bar_area(reb["top"]["d"])
    layer("straight", steel, int(reb["top"]["n"]), A_top, yc2, zc1, yc2, zc2)

    # bottom bars
    A_bot = bar_area(reb["bot"]["d"])
    layer("straight", steel, int(reb["bot"]["n"]), A_bot, yc1, zc1, yc1, zc2)

    # left bars
    A_left = bar_area(reb["left"]["d"])
    layer("straight", steel, int(reb["left"]["n"]), A_left, yc1, zc1, yc2, zc1)

    # right bars
    A_right = bar_area(reb["right"]["d"])
    layer("straight", steel, int(reb["right"]["n"]), A_right, yc1, zc2, yc2, zc2)


# =========================================================
# 4. 建立 2D OpenSees 模型
# =========================================================
def build_opensees_2d_model(node_list, element_list, story_nodes, story_mass):
    wipe()
    model("basic", "-ndm", 2, "-ndf", 3)

    # ---------- 节点 ----------
    for nd, (x, y) in node_list.items():
        node(nd, x, y)

    # ---------- 基底约束 ----------
    for nd in story_nodes[0]:
        fix(nd, 1, 1, 1)

    # ---------- 质量 ----------
    # 每层质量均分到该层节点，仅在 X/Y 向赋质量，转动质量给极小值
    for k, nodes_k in story_nodes.items():
        if k == 0:
            continue
        mk = float(story_mass[k]) / len(nodes_k)
        for nd in nodes_k:
            mass(nd, mk, mk, 1e-9)

    # ---------- 几何变换 ----------
    transfTag_col = 1
    transfTag_beam = 2
    geomTransf("PDelta", transfTag_col)
    geomTransf("Linear", transfTag_beam)

    # ---------- 材料/截面/单元 ----------
    nextMatTag = 1
    nextSecTag = 1
    nextIntTag = 1

    for eid, e in element_list.items():
        mat_tags = define_uniaxial_materials(e["mat_cfg"], start_tag=nextMatTag)
        nextMatTag += 10

        secTag = nextSecTag
        nextSecTag += 1

        nl_cfg = e["nl_cfg"]
        make_rc_fiber_section_2d(
            secTag=secTag,
            sec_cfg=e["sec_cfg"],
            mat_tags=mat_tags,
            nfCoreY=nl_cfg["nfCoreY"],
            nfCoreZ=nl_cfg["nfCoreZ"],
            nfCoverY=nl_cfg["nfCoverY"],
            nfCoverZ=nl_cfg["nfCoverZ"],
        )

        intTag = nextIntTag
        nextIntTag += 1
        nIP = int(nl_cfg["nIP_col"] if e["type"] == "column" else nl_cfg["nIP_beam"])
        beamIntegration("Lobatto", intTag, secTag, nIP)

        transfTag = transfTag_col if e["type"] == "column" else transfTag_beam
        element("dispBeamColumn", eid, e["i"], e["j"], transfTag, intTag)


# =========================================================
# 5. 重力分析
# =========================================================
def run_gravity_analysis(story_nodes, story_mass):
    timeSeries("Linear", 100)
    pattern("Plain", 100, 100)

    g = 9.81
    for k, nodes_k in story_nodes.items():
        if k == 0:
            continue
        Wk = float(story_mass[k]) * g
        fk = -Wk / len(nodes_k)
        for nd in nodes_k:
            load(nd, 0.0, fk, 0.0)

    system("BandGeneral")
    constraints("Transformation")
    numberer("RCM")
    test("NormDispIncr", 1.0e-6, 50)
    algorithm("Newton")
    integrator("LoadControl", 0.1)
    analysis("Static")

    ok = analyze(10)
    if ok != 0:
        raise RuntimeError("Gravity analysis failed.")

    loadConst("-time", 0.0)


# =========================================================
# 6. 模态分析
# =========================================================
def run_modal_analysis(nModes=5):
    lambdaVals = eigen(nModes)

    omega = []
    freq = []
    period = []

    print(f"\n===== 前 {nModes} 阶自振频率和周期 =====")
    for i, lam in enumerate(lambdaVals):
        w = math.sqrt(lam)
        f = w / (2.0 * math.pi)
        T = 1.0 / f
        omega.append(w)
        freq.append(f)
        period.append(T)
        print(f"Mode {i+1}:  ω = {w:.3f} rad/s,  f = {f:.3f} Hz,  T = {T:.3f} s")

    return np.array(omega), np.array(freq), np.array(period)


# =========================================================
# 7. 基底剪力工具函数
# =========================================================
def get_base_shear(base_nodes, dof=1):
    reactions()
    V = 0.0
    for nd in base_nodes:
        V += nodeReaction(nd, dof)
    return -V


def get_max_interstory_drift(node_list, story_nodes, y_levels, push_dir_dof=1):
    """
    计算当前分析步的最大层间位移角 ISDR_max
    2D 情况下 push_dir_dof=1 表示 X 向侧移
    story_nodes: {0:[...], 1:[...], 2:[...], ...}
    node_list: {nodeTag: (x, y)}   # 2D
    """
    isdr_list = []

    story_keys = sorted(story_nodes.keys())
    for k in story_keys:
        if k == 0:
            continue

        lower_nodes = story_nodes[k - 1]
        upper_nodes = story_nodes[k]

        if len(lower_nodes) == 0 or len(upper_nodes) == 0:
            continue

        # 该层与下层平均水平位移
        u_lower = np.mean([nodeDisp(nd, push_dir_dof) for nd in lower_nodes])
        u_upper = np.mean([nodeDisp(nd, push_dir_dof) for nd in upper_nodes])

        h_story = y_levels[k] - y_levels[k - 1]

        if h_story > 0:
            isdr = abs(u_upper - u_lower) / h_story
            isdr_list.append(isdr)

    if len(isdr_list) == 0:
        return 0.0

    return max(isdr_list)

# =========================================================
# 8. Pushover 分析
# =========================================================
def run_pushover_analysis(node_list, story_master, story_nodes, roof_ctrl_node, y_levels,
                          push_dir_dof=1, dU=5e-4, drift_target=0.03):
    """
    2D pushover
    在原有基础上增加：
    - 每一步记录 ISDR_max
    - 在 P1/P2/P3 处提取对应的 ISDR_max, roof drift, base shear
    """
    base_nodes = list(story_nodes[0])
    H_total = max(y_levels)
    target_roof_disp = drift_target * H_total

    # lateral load pattern
    timeSeries("Linear", 200)
    pattern("Plain", 200, 200)

    weights = {}
    wsum = 0.0
    for k, master in story_master.items():
        yk = node_list[master][1]
        wk = max(yk, 0.0)
        weights[k] = wk
        wsum += wk

    F_total = 1.0
    for k, master in story_master.items():
        fk = F_total * weights[k] / wsum
        if push_dir_dof == 1:
            load(master, fk, 0.0, 0.0)
        else:
            load(master, 0.0, fk, 0.0)

    # analysis settings
    system("BandGeneral")
    constraints("Transformation")
    numberer("RCM")
    test("NormDispIncr", 1e-7, 60)
    algorithm("Newton")
    integrator("DisplacementControl", roof_ctrl_node, push_dir_dof, dU)
    analysis("Static")

    U_hist = []
    V_hist = []
    theta_hist = []
    isdr_max_hist = []

    # 初始点
    U0 = nodeDisp(roof_ctrl_node, push_dir_dof)
    V0 = get_base_shear(base_nodes, dof=push_dir_dof)
    theta0 = abs(U0) / H_total
    isdr0 = get_max_interstory_drift(node_list, story_nodes, y_levels, push_dir_dof)

    U_hist.append(U0)
    V_hist.append(V0)
    theta_hist.append(theta0)
    isdr_max_hist.append(isdr0)

    max_steps = int(target_roof_disp / abs(dU)) + 50

    print(f"[Pushover] H_total = {H_total:.4f} m")
    print(f"[Pushover] target roof disp = {target_roof_disp:.4f} m")
    print(f"[Pushover] max_steps = {max_steps}")

    for step in range(1, max_steps + 1):
        ok = analyze(1)

        if ok != 0:
            print(f"[Pushover] step {step} failed, trying fallback...")
            test("NormDispIncr", 1e-6, 200)
            algorithm("KrylovNewton")
            ok2 = analyze(1)

            if ok2 != 0:
                algorithm("ModifiedNewton")
                ok3 = analyze(1)
                if ok3 != 0:
                    print(f"[Pushover] terminated at step {step}")
                    break

            test("NormDispIncr", 1e-7, 60)
            algorithm("Newton")

        U = nodeDisp(roof_ctrl_node, push_dir_dof)
        V = get_base_shear(base_nodes, dof=push_dir_dof)
        theta = abs(U) / H_total
        isdr_max = get_max_interstory_drift(node_list, story_nodes, y_levels, push_dir_dof)

        U_hist.append(U)
        V_hist.append(V)
        theta_hist.append(theta)
        isdr_max_hist.append(isdr_max)

        if abs(U) >= target_roof_disp:
            print(f"[Pushover] reached target displacement at step {step}")
            break

        if len(V_hist) > 20:
            Vmax = max(np.abs(V_hist))
            if Vmax > 1e-6 and abs(V) < 0.6 * Vmax and abs(U) > 0.2 * target_roof_disp:
                print(f"[Pushover] strength dropped below 60% of peak, stop.")
                break

    U_hist = np.array(U_hist, dtype=float)
    V_hist = np.array(V_hist, dtype=float)
    theta_hist = np.array(theta_hist, dtype=float)
    isdr_max_hist = np.array(isdr_max_hist, dtype=float)

    # ---------- 双折线 ----------
    Vabs = np.abs(V_hist)
    Vmax = Vabs.max()
    V_target = 0.5 * Vmax
    idx = np.where(Vabs >= V_target)[0]
    i = idx[0]
    U06 = abs(U_hist[i])
    Ke0 = V_target / U06

    Du = float(U_hist[-1])
    Vu = float(V_hist[-1])
    E_actual = float(np.trapezoid(V_hist, U_hist))

    def bilinear_energy(Uy):
        Vy = Ke0 * Uy
        E_bi = 0.5 * Vy * Uy + Vy * (Du - Uy)
        return E_bi, Vy

    Uy_lo = 1e-6
    Uy_hi = 0.9 * Du

    for _ in range(60):
        Uy_mid = 0.5 * (Uy_lo + Uy_hi)
        E_mid, _ = bilinear_energy(Uy_mid)
        if E_mid > E_actual:
            Uy_hi = Uy_mid
        else:
            Uy_lo = Uy_mid

    Uy = 0.5 * (Uy_lo + Uy_hi)
    _, Vy = bilinear_energy(Uy)

    # performance points
    U_p1 = 0.7 * Uy
    U_p2 = 1.2 * Uy
    U_p3 = 2.0 * Uy

    def _closest_idx(arr, target):
        return int(np.argmin(np.abs(arr - target)))

    idx_P1 = _closest_idx(U_hist, U_p1)
    idx_P2 = _closest_idx(U_hist, U_p2)
    idx_P3 = _closest_idx(U_hist, U_p3)

    # 对应 roof drift
    theta_P1 = float(theta_hist[idx_P1])
    theta_P2 = float(theta_hist[idx_P2])
    theta_P3 = float(theta_hist[idx_P3])

    # 对应 ISDR_max
    isdr_P1 = float(isdr_max_hist[idx_P1])
    isdr_P2 = float(isdr_max_hist[idx_P2])
    isdr_P3 = float(isdr_max_hist[idx_P3])

    # 对应 base shear
    V_P1 = float(V_hist[idx_P1])
    V_P2 = float(V_hist[idx_P2])
    V_P3 = float(V_hist[idx_P3])

    # 对应 roof displacement
    U_P1 = float(U_hist[idx_P1])
    U_P2 = float(U_hist[idx_P2])
    U_P3 = float(U_hist[idx_P3])

    print("\n===== Performance points =====")
    print(f"P1 = 0.7Uy: U={U_P1:.5e} m, V={V_P1/1000:.3f} kN, roof drift={theta_P1:.5f}, ISDR_max={isdr_P1:.5f}")
    print(f"P2 = 1.2Uy: U={U_P2:.5e} m, V={V_P2/1000:.3f} kN, roof drift={theta_P2:.5f}, ISDR_max={isdr_P2:.5f}")
    print(f"P3 = 2.0Uy: U={U_P3:.5e} m, V={V_P3/1000:.3f} kN, roof drift={theta_P3:.5f}, ISDR_max={isdr_P3:.5f}")

    results = {
        "U_hist": U_hist,
        "V_hist": V_hist,
        "theta_hist": theta_hist,
        "isdr_max_hist": isdr_max_hist,
        "Uy": Uy,
        "Vy": Vy,
        "H_total": H_total,

        # performance point index
        "idx_P1": idx_P1,
        "idx_P2": idx_P2,
        "idx_P3": idx_P3,

        # performance point roof drift
        "theta_P1": theta_P1,
        "theta_P2": theta_P2,
        "theta_P3": theta_P3,

        # performance point ISDR max
        "isdr_P1": isdr_P1,
        "isdr_P2": isdr_P2,
        "isdr_P3": isdr_P3,

        # performance point base shear
        "V_P1": V_P1,
        "V_P2": V_P2,
        "V_P3": V_P3,

        # performance point roof displacement
        "U_P1": U_P1,
        "U_P2": U_P2,
        "U_P3": U_P3,
    }
    return results


# =========================================================
# 9. 画 pushover 图
# =========================================================
def plot_pushover(results):
    U_hist = results["U_hist"]
    V_hist = results["V_hist"]
    Uy = results["Uy"]
    Vy = results["Vy"]

    Du = U_hist[-1]

    U_mm = U_hist * 1000.0
    V_kN = V_hist / 1000.0

    plt.figure(figsize=(7, 5))
    plt.plot(U_mm, V_kN, lw=2.0, label="Pushover")

    U_bi = np.array([0.0, Uy, Du])
    V_bi = np.array([0.0, Vy, Vy])
    plt.plot(U_bi * 1000.0, V_bi / 1000.0, "--", lw=2.0, label="Bilinear")

    plt.scatter([Uy * 1000.0], [Vy / 1000.0], s=80, label="Yield")

    plt.scatter([results["U_P1"] * 1000.0], [results["V_P1"] / 1000.0],
                s=80, color="green",  marker="o", label="P1")
    plt.scatter([results["U_P2"] * 1000.0], [results["V_P2"] / 1000.0],
                s=80, color="orange", marker="o", label="P2")
    plt.scatter([results["U_P3"] * 1000.0], [results["V_P3"] / 1000.0],
                s=80, color="red",    marker="o", label="P3")

    plt.xlabel("Roof displacement Δ (mm)")
    plt.ylabel("Base shear V (kN)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# =========================================================
# 10. 总控函数
# =========================================================
def build_model_and_get_T1_2d(seed=1, nModes=5, do_plot=True):
    params = generate_random_2d_frame_params(seed=seed)

    node_list, element_list, story_nodes, story_master, story_mass, roof_ctrl_node, y_levels = build_2d_frame_nodes_elements(params)

    build_opensees_2d_model(node_list, element_list, story_nodes, story_mass)

    run_gravity_analysis(story_nodes, story_mass)

    omega, freq, period = run_modal_analysis(nModes=nModes)
    T1 = float(period[0])

    pushover_results = run_pushover_analysis(
        node_list=node_list,
        story_master=story_master,
        story_nodes=story_nodes,
        roof_ctrl_node=roof_ctrl_node,
        y_levels=y_levels,
        push_dir_dof=1,
        dU=5e-4,
        drift_target=0.03
    )

    if do_plot:
        plot_pushover(pushover_results)

    # ===== 用 ISDR_max 作为 damage label 阈值 =====
    isdr_P1 = round(pushover_results["isdr_P1"], 4)
    isdr_P2 = round(pushover_results["isdr_P2"], 4)
    isdr_P3 = round(pushover_results["isdr_P3"], 4)

    model_info = {
        "params": params,
        "node_list": node_list,
        "element_list": element_list,
        "story_nodes": story_nodes,
        "story_master": story_master,
        "story_mass": story_mass,
        "roof_ctrl_node": roof_ctrl_node,
        "y_levels": y_levels,

        # 顺手存下来，后面更方便
        "T1": T1,
        "period": period,
        "pushover_results": pushover_results,
        "isdr_P1": isdr_P1,
        "isdr_P2": isdr_P2,
        "isdr_P3": isdr_P3,
    }

    return T1, period, model_info, isdr_P1, isdr_P2, isdr_P3



import os
import json
import pickle

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def save_model_library(model_id, model_info, base_dir="01_models"):
    model_dir = os.path.join(base_dir, f"model_{model_id:04d}")
    os.makedirs(model_dir, exist_ok=True)

    # 1. 原始随机参数
    with open(os.path.join(model_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(model_info["params"]), f, indent=2)

    # 2. 模态结果
    modal_summary = {
        "T1": model_info["T1"],
        "period": model_info["period"]
    }
    with open(os.path.join(model_dir, "modal_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(modal_summary), f, indent=2)

    # 3. pushover 阈值
    pushover_summary = {
        "isdr_P1": model_info["isdr_P1"],
        "isdr_P2": model_info["isdr_P2"],
        "isdr_P3": model_info["isdr_P3"],
        "pushover_results": {
            "Uy": model_info["pushover_results"]["Uy"],
            "Vy": model_info["pushover_results"]["Vy"],
            "U_P1": model_info["pushover_results"]["U_P1"],
            "U_P2": model_info["pushover_results"]["U_P2"],
            "U_P3": model_info["pushover_results"]["U_P3"],
            "V_P1": model_info["pushover_results"]["V_P1"],
            "V_P2": model_info["pushover_results"]["V_P2"],
            "V_P3": model_info["pushover_results"]["V_P3"]
        }
    }
    with open(os.path.join(model_dir, "pushover_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(pushover_summary), f, indent=2)

    # 4. 完整对象（以后重建或调试方便）
    with open(os.path.join(model_dir, "model_info.pkl"), "wb") as f:
        pickle.dump(model_info, f)



# =========================================================
# 11. 示例运行
# =========================================================
if __name__ == "__main__":
    for model_id in range(1, 101):
        seed = model_id
        T1, period, model_info, isdr_P1, isdr_P2, isdr_P3 = build_model_and_get_T1_2d(
            seed=seed,
            nModes=5,
            do_plot=False
        )

        save_model_library(model_id, model_info, base_dir="01_models")

        print(f"Saved model_{model_id:04d} | T1={T1:.3f} s | DS thresholds = {isdr_P1}, {isdr_P2}, {isdr_P3}")

