# %%
"""
DSF → Damage State (DS) 判别器 Pipeline
数据: 100 models × 12 GMs × 7 PGA levels = 8400 samples
特征: 12 DSF特征 → 标签: damage_state (0/1/2/3)
"""
import os
import json

import numpy as np                          # 数值和数组运算
import pandas as pd                         # 读csv 处理表格数据
import matplotlib.pyplot as plt             # 画图
import seaborn as sns                       # 更美观的统计图
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder              # 特征缩放和标签编码（没用到）
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score   # StratifiedGroupKFold分层分组交叉验证 cross_val_score 做交叉验证并返回评分
from sklearn.pipeline import Pipeline                                       # 构建机器学习管道
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, ExtraTreesClassifier            
)                                                                           # 各种集成树模型
from sklearn.svm import SVC                                                 # 支持向量机分类器
from sklearn.neural_network import MLPClassifier                            # 多层感知机分类器（神经网络）
from sklearn.linear_model import LogisticRegression                         # 逻辑回归分类器（元模型）
from sklearn.neighbors import KNeighborsClassifier                          # K近邻分类器（基线模型）
from sklearn.naive_bayes import GaussianNB                                  # 高斯朴素贝叶斯分类器（基线模型）
from sklearn.metrics import (classification_report, confusion_matrix,       
                             accuracy_score, f1_score, roc_auc_score)       # 各种评估指标和报告函数
import xgboost as xgb
import lightgbm as lgb              # XGBoost和LightGBM是两种流行的梯度提升树算法库

import optuna                       # 贝叶斯优化库，用于超参数调优
import shap                         # SHAP库，用于模型解释性分析
import joblib                       # 模型持久化（保存和加载模型）

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1. 数据加载与预处理
# ============================================================
DSF_FEATURES = [
    'DSF_T1_CENT', 'DSF_T1_MAC', 'DSF_dF1peak', 'DSF_T1_AREA',
    'DSF_T2_CENT', 'DSF_T2_MAC', 'DSF_dF2peak', 'DSF_T2_AREA',
    'DSF_KPRX',    'DSF_RMSratio','DSF_AR4_L2', 'DSF_Time_Delay'
]
TARGET      = 'damage_state'
GROUP_COL   = 'model_id'
N_OPTUNA_TRIALS = 50
N_CV_SPLITS     = 5
RANDOM_STATE    = 42

def load_and_preprocess(csv_path: str,
                        dsf_features = DSF_FEATURES,
                        target = TARGET,
                        group_col = GROUP_COL) -> tuple:
    df = pd.read_csv(csv_path)
    print(f"原始数据行数: {len(df)}")    
    # 目标变量分布
    print("\n损伤状态分布:\n", df[TARGET].value_counts().sort_index())
    print("\n损伤状态比例:\n", df[TARGET].value_counts(normalize=True).sort_index().round(3))

    X = df[dsf_features].values
    y = df[target].values.astype(int)
    groups = df[group_col].values      # model_0001, model_0002, ...
    
    return X, y, groups, df

# X, y, groups, df = load_and_preprocess('population_dataset.csv')


# ============================================================
# Step 2: 特征探索性分析（EDA）
# ============================================================
def feature_analysis(df: pd.DataFrame, 
                     dsf_features=DSF_FEATURES,
                     target=TARGET,
                     save_path="feature_analysis.png"):
    """特征相关性与分布分析
      1. 各特征与损伤状态的皮尔逊相关系数
      2. 特征之间的相关矩阵（热力图）
      3. 相关性最强的4个特征在各DS下的箱线图
      4. DS样本数量分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 特征-标签相关性 Pearson 相关系数
    corr = df[dsf_features + [target]].corr()[target].drop(target).sort_values(key=abs, ascending=False)
    corr.plot(kind='barh', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('Pearson correlation of DSFs with Damage State')
    axes[0,0].axvline(0, color='k', lw=0.5)
    
    # 2. 特征相关矩阵
    corr_matrix = df[dsf_features].corr()
    sns.heatmap(corr_matrix, ax=axes[0,1], cmap='RdBu_r', center=0,
                annot=False, fmt='.2f', square=True)
    axes[0,1].set_title('Correlation Matrix of DSFs')
    
    # 3. 各DS下的特征箱线图（取最重要的4个）
    top4 = corr.abs().nlargest(4).index.tolist()
    df_melt = df[top4 + [target]].melt(id_vars=target, var_name='feature', value_name='value')
    sns.boxplot(data=df_melt, x='feature', y='value', hue=target, ax=axes[1,0])
    axes[1,0].set_title('DS distribution for the Top 4 DSFs')
    axes[1,0].tick_params(axis='x', rotation=30)
    
    # 4. DS分布
    df[target].value_counts().sort_index().plot(kind='bar', ax=axes[1,1], color='coral', edgecolor='k')
    axes[1,1].set_title('DS distribution')
    axes[1,1].set_xlabel('Damage State')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print("特征分析图已保存")

# feature_analysis(df, save_path="feature_analysis.png")


# ============================================================
# Step 3: 数据划分（按模型 ID 分组，防数据泄露）
# ============================================================

def group_train_val_test_split(X, y, groups, test_ratio=0.2, random_state=RANDOM_STATE):
    """
    关键：按 model_id 分组划分，确保同一结构的数据不跨越训练/测试集
    """
    unique_models = np.unique(groups)
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_models)
    
    n      = len(unique_models)
    n_test = int(n * test_ratio)
    
    test_models  = unique_models[:n_test]
    train_models = unique_models[n_test:]
    
    def mask(models_list):
        return np.isin(groups, models_list)
    
    X_train, y_train = X[mask(train_models)], y[mask(train_models)]
    X_test,  y_test  = X[mask(test_models)],  y[mask(test_models)]
    groups_train     = groups[mask(train_models)]
    
    print(f"\n数据划分 (按模型ID分组):")
    print(f"  训练集: {len(train_models)} 模型, {len(X_train)} 样本")
    print("\n 训练集 标签分布:")
    print(pd.Series(y_train).value_counts(normalize=False).sort_index())

    print(f"  测试集: {len(test_models)} 模型, {len(X_test)} 样本")
    print("\n 测试集 标签分布:")
    print(pd.Series(y_test).value_counts(normalize=False).sort_index())
    
    return (X_train, y_train, groups_train, X_test, y_test)

# X_train,y_train,groups_train,X_test,y_test = group_train_val_test_split(X, y, groups)


# ============================================================
# Step 4: 超参数优化
# ============================================================
def _run_cv(pipeline, X, y, groups, n_cv_splits=N_CV_SPLITS):
    """统一的交叉验证入口，返回 F1-macro 均值"""
    scores = cross_val_score(
        pipeline, X, y,
        groups=groups,
        cv=StratifiedGroupKFold(n_splits=n_cv_splits),
        scoring='f1_macro',
        n_jobs=-1
    )
    return scores.mean()

# ---------- XGBoost ----------
def optimize_xgboost(X_train, y_train, groups_train, 
                     n_trials=N_OPTUNA_TRIALS, n_cv_splits=N_CV_SPLITS, random_state=RANDOM_STATE):
    def objective(trial):
        model = xgb.XGBClassifier(
            n_estimators     = trial.suggest_int('n_estimators', 100, 500),    # 树的数量
            max_depth        = trial.suggest_int('max_depth', 3, 10),          # 树的最大深度
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),     # 0.6-1.0之间的连续值，表示每棵树随机采样的训练样本比例
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),# 0.6-1.0之间的连续值，表示每棵树随机采样的特征比例
            min_child_weight = trial.suggest_int('min_child_weight', 1, 15),   # 叶子节点最小样本权重和，控制过拟合
            reg_alpha        = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),# L1正则化项权重，控制过拟合
            reg_lambda       = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),# L2正则化项权重，控制过拟合
            random_state = random_state, eval_metric = 'mlogloss', verbosity = 0
        ) # 使用建议的超参数创建XGBoost分类器
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return _run_cv(pipe, X_train, y_train, groups_train, n_cv_splits=n_cv_splits) # Pipeline：Scaler 和模型打包，CV 时不会泄露数据，返回平均 F1-macro 评分
    
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state)) # 创建一个Optuna研究对象，指定优化方向为最大化，并使用TPE采样器，设置随机种子以确保结果可复现
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nXGBoost最优F1-macro: {study.best_value:.4f}")
    print(f"最优超参数: {study.best_params}")
    return study.best_params

# ---------- LightGBM ----------
def optimize_lightgbm(X_train, y_train, groups_train, 
                      n_trials=N_OPTUNA_TRIALS, n_cv_splits=N_CV_SPLITS, random_state=RANDOM_STATE):  
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 3, 10)  # 树的最大深度
        # 确保 num_leaves < 2^max_depth，官方推荐约 0.6 倍
        upper_limit = max(8, int(0.6 * (2 ** max_depth)))
        num_leaves = trial.suggest_int('num_leaves', 7, upper_limit)

        model = lgb.LGBMClassifier(
            n_estimators      = trial.suggest_int('n_estimators', 100, 600), # 树的数量
            max_depth         = max_depth,
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            num_leaves        = num_leaves, # 叶子节点数，控制模型复杂度，过大容易过拟合
            min_child_samples = trial.suggest_int('min_child_samples', 5, 50), # 叶子节点最小样本数，控制过拟合
            subsample         = trial.suggest_float('subsample', 0.6, 1.0),     # 0.6-1.0之间的连续值，表示每棵树随机采样的训练样本比例
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),# 0.5-1.0之间的连续值，表示每棵树随机采样的特征比例
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),# L1正则化项权重，控制过拟合
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),# L2正则化项权重，控制过拟合
            random_state= random_state, verbose= -1, n_jobs= -1
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return _run_cv(pipe, X_train, y_train, groups_train, n_cv_splits=n_cv_splits)
    
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nLightGBM最优F1-macro: {study.best_value:.4f}")
    print(f"最优超参数: {study.best_params}")
    return study.best_params

# ---------- RandomForest ----------
def optimize_rf(X_train, y_train, groups_train, 
                n_trials=N_OPTUNA_TRIALS, n_cv_splits=N_CV_SPLITS, random_state=RANDOM_STATE):
    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=   trial.suggest_int('n_estimators', 100, 500),
            max_depth=      trial.suggest_int('max_depth', 3, 20),
            min_samples_split= trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=  trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=   trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            random_state= random_state, n_jobs= -1
        ) # 使用建议的超参数创建随机森林分类器
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return _run_cv(pipe, X_train, y_train, groups_train, n_cv_splits=n_cv_splits)    
    
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nRandomForest最优F1-macro: {study.best_value:.4f}")
    print(f"最优超参数: {study.best_params}")
    return study.best_params

# ---------- SVM ----------
def optimize_svm(X_train, y_train, groups_train, 
                 n_trials=N_OPTUNA_TRIALS, n_cv_splits=N_CV_SPLITS, random_state=RANDOM_STATE):
    """
    SVM 对特征尺度非常敏感，必须做标准化，这里通过 Pipeline 实现无泄露标准化。
    C：惩罚系数，越大越容易过拟合；gamma：RBF 核的带宽。
    """
    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        # degree 只对 poly 核有意义
        degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
        model = SVC(
            C           = trial.suggest_float('C', 0.01, 100.0, log=True), # 惩罚系数，越大越容易过拟合
            gamma       = trial.suggest_float('gamma', 1e-4, 10.0, log=True), # RBF核的带宽，越小越容易过拟合
            kernel      = kernel,
            degree      = degree,   # 仅对 poly 有效
            probability = True, 
            random_state=random_state
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return _run_cv(pipe, X_train, y_train, groups_train, n_cv_splits=n_cv_splits)
 
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nSVM 最优 F1-macro: {study.best_value:.4f}")
    print(f"最优超参数: {study.best_params}")
    return study.best_params

# ---------- KNN ----------
def optimize_knn(X_train, y_train, groups_train, 
                 n_trials=N_OPTUNA_TRIALS, n_cv_splits=N_CV_SPLITS, random_state=RANDOM_STATE):
    """KNN 对特征尺度极其敏感，必须标准化。"""
    def objective(trial):
        model = KNeighborsClassifier(
            n_neighbors = trial.suggest_int('n_neighbors', 3, 30), # 邻居数量，越小越容易过拟合
            weights     = trial.suggest_categorical('weights', ['uniform', 'distance']), # 权重类型，uniform表示所有邻居权重相同，distance表示距离近的邻居权重更大
            p           = trial.suggest_int('p', 1, 2),  # 1=曼哈顿, 2=欧氏
            n_jobs      = -1
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return _run_cv(pipe, X_train, y_train, groups_train, n_cv_splits=n_cv_splits)
 
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nKNN 最优 F1-macro: {study.best_value:.4f}")
    print(f"最优超参数: {study.best_params}")
    return study.best_params

# ---------- MLP ----------
def optimize_mlp(X_train, y_train, groups_train,
                 n_trials=N_OPTUNA_TRIALS, n_cv_splits=N_CV_SPLITS, random_state=RANDOM_STATE):
    """
    MLP 对学习率和网络结构很敏感，也需要标准化。
    """
    def objective(trial):
        # 动态决定隐藏层数和每层节点数
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layer_sizes = tuple(
            trial.suggest_int(f'n_units_l{i}', 16, 256)
            for i in range(n_layers)
        )
        model = MLPClassifier(
            hidden_layer_sizes = layer_sizes,
            activation         = trial.suggest_categorical('activation', ['relu', 'tanh']),
            learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 0.1, log=True),
            alpha              = trial.suggest_float('alpha', 1e-5, 0.1, log=True),  # L2正则
            max_iter           = 500,
            early_stopping     = True,
            random_state       = random_state
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return _run_cv(pipe, X_train, y_train, groups_train, n_cv_splits=n_cv_splits)
 
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nMLP 最优 F1-macro: {study.best_value:.4f}")
    print(f"最优超参数: {study.best_params}")
    return study.best_params

# ---------- Logistic Regression----------
def optimize_lr(X_train, y_train, groups_train, 
                n_trials=N_OPTUNA_TRIALS, n_cv_splits=N_CV_SPLITS, random_state=RANDOM_STATE):
    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        model = LogisticRegression(
            C       = trial.suggest_float('C', 1e-3, 100.0, log=True),
            penalty = penalty,
            solver  = 'saga',
            max_iter= 1000,
            random_state=random_state
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        return _run_cv(pipe, X_train, y_train, groups_train, n_cv_splits=n_cv_splits)
 
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nLogisticRegression 最优 F1-macro: {study.best_value:.4f}")
    print(f"最优超参数: {study.best_params}")
    return study.best_params

# 超参数优化
# best_params = {}
# print("\n[Optuna] 优化 XGBoost ...")
# best_params['XGBoost'] = optimize_xgboost(X_train, y_train, groups_train, n_trials=1)
# print("\n[Optuna] 优化 LightGBM ...")
# best_params['LightGBM'] = optimize_lightgbm(X_train, y_train, groups_train, n_trials=1)
# print("\n[Optuna] 优化 RandomForest ...")
# best_params['RandomForest'] = optimize_rf(X_train, y_train, groups_train, n_trials=1)
# print("\n[Optuna] 优化 SVM ...")
# best_params['SVM'] = optimize_svm(X_train, y_train, groups_train, n_trials=1)
# print("\n[Optuna] 优化 KNN ...")
# best_params['KNN'] = optimize_knn(X_train, y_train, groups_train, n_trials=1)
# print("\n[Optuna] 优化 MLP ...")
# best_params['MLP'] = optimize_mlp(X_train, y_train, groups_train, n_trials=1)
# print("\n[Optuna] 优化 LogisticRegression ...")
# best_params['LogisticRegression'] = optimize_lr(X_train, y_train, groups_train, n_trials=1)


# mlp_raw_params = best_params.get('MLP', {}).copy()
# n_layers = mlp_raw_params.pop('n_layers')
# # 根据 n_layers 拼接出元组，并从字典中删掉旧的键
# hidden_layer_sizes = tuple(mlp_raw_params.pop(f'n_units_l{i}') for i in range(n_layers))
# # 把转换后的元组塞回去
# mlp_raw_params['hidden_layer_sizes'] = hidden_layer_sizes



# ============================================================
# 5. 用最优参数 训练模型 与 评估模型
# ============================================================
def build_pipeline(model) -> Pipeline:
    """
    把 StandardScaler 和任意模型组合成 Pipeline。
    这样在预测时只需调用 pipe.predict(X_raw)，
    内部自动完成 transform → predict，无需手动 scaler.transform。
    """
    return Pipeline([('scaler', StandardScaler()), ('clf', model)])

def evaluate_pipeline(name: str, pipe: Pipeline,
                       X_train, y_train,
                       X_test,  y_test) -> dict:
    """
    测试集结果就是最终结果。
    """
    pipe.fit(X_train, y_train)
 
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test) if hasattr(pipe, 'predict_proba') else None
 
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    auc = (roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
           if y_prob is not None else np.nan)
 
    print(f"  [test] Acc={acc:.3f}  F1-macro={f1:.3f}  AUC-OVR={auc:.3f}")
 
    return {'acc': acc, 'f1_macro': f1, 'auc_ovr': auc, 'y_pred': y_pred, 'y_true': y_test}



def run_all_models(X_train, y_train, groups_train,
                   X_test, y_test,
                   best_params: dict,
                   mlp_raw_params: dict): # 训练所有模型并评估性能，返回结果字典和训练好的管道
 
    models = {
        'RF': RandomForestClassifier(
            **{
                **{'n_estimators': 300, 'random_state': RANDOM_STATE, 'n_jobs': -1},
                **best_params.get('RandomForest', {})}),
 
        'XGBoost': xgb.XGBClassifier(
            **{
                **{'n_estimators':300,'random_state':RANDOM_STATE,'eval_metric':'mlogloss','verbosity':0},
                **best_params.get('XGBoost', {})}),
 
        # 'LGBM': lgb.LGBMClassifier(
        #     **{
        #         **{'n_estimators': 300, 'random_state': RANDOM_STATE,'verbose': -1, 'n_jobs': -1},
        #         **best_params.get('LightGBM', {})}),
 
        'SVM': SVC(
            probability=True, random_state=RANDOM_STATE,
            **best_params.get('SVM', {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'})),
 
        'MLP': MLPClassifier(
            max_iter=500, early_stopping=True, random_state=RANDOM_STATE,
            **mlp_raw_params), # MLP 的特殊处理，已在前面转换好 hidden_layer_sizes
 
        'KNN': KNeighborsClassifier(
            n_jobs=-1,
            **best_params.get('KNN', {'n_neighbors': 10, 'weights': 'distance'})),
 
        'LR': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver='saga',
            **best_params.get('LogisticRegression', {'C': 1.0, 'penalty': 'l2', 'solver': 'saga'})),
    }
 
    # Stacking 集成
    # stacking_base = [
    #         ('rf',
    #         RandomForestClassifier(
    #             **{
    #                 **{'random_state': RANDOM_STATE, 'n_jobs': -1},
    #                 **best_params['RandomForest']})),
    #         ('xgb',
    #         xgb.XGBClassifier(
    #             **{
    #                 **{'random_state': RANDOM_STATE,'eval_metric': 'mlogloss', 'verbosity': 0},
    #                 **best_params['XGBoost']})),
    #         ('lgb',
    #         lgb.LGBMClassifier(
    #             **{
    #                 **{'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1},
    #                 **best_params['LightGBM']}))]
    
    # models['Stacking'] = StackingClassifier(
    #     estimators=stacking_base,
    #     final_estimator=LogisticRegression(
    #         **{
    #             **{'max_iter': 1000,'random_state': RANDOM_STATE},
    #             **best_params['LogisticRegression']
    #         }),     
    #     cv=5,
    #     n_jobs=-1,
    #     passthrough=True
    # )
 
    all_results   = {}
    trained_pipes = {}
    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"训练模型: {name}")
        pipe   = build_pipeline(model)
        result = evaluate_pipeline(name, pipe, X_train, y_train, X_test, y_test)
        all_results[name]   = result
        trained_pipes[name] = pipe
 
    return all_results, trained_pipes


# Step 5: 训练所有模型
# all_results, trained_pipes = run_all_models(
#     X_train, y_train, groups_train, X_test, y_test, best_params)


# ============================================================
# 6. 可视化对比
# ============================================================
def plot_comparison(all_results: dict, 
                    model_comparison_text_save_path="model_comparison.csv", 
                    model_comparison_figure_save_path="model_comparison.png", 
                    confusion_save_path="confusion_matrix.png"):
    """多模型性能柱状图 + 混淆矩阵"""
    # 汇总表
    # ── 0. 过滤掉 LGBM ──────────────────────────────────────────────
    results = {k: v for k, v in all_results.items() if k not in ('LGBM', 'LightGBM')}

    n_classes = 4
    ds_labels = [f'DS{i}' for i in range(n_classes)]

    # ── 1. 计算六模型均值与标准差 ────────────────────────────────────
    metric_keys = ['acc', 'f1_macro', 'auc_ovr']
    metric_cols  = ['Accuracy', 'F1-macro', 'AUC-OVR']
    
    mean_vals = {col: np.mean([results[m][k] for m in results]) for col, k in zip(metric_cols, metric_keys)}
    std_vals  = {col: np.std ([results[m][k] for m in results], ddof=1) for col, k in zip(metric_cols, metric_keys)}

    # ── 2. 汇总表（含 Ensemble_Mean 行，不排序）─────────────────────
    rows = []
    for name, r in results.items():
        rows.append({'Model': name,
                     'Accuracy': r['acc'], 'F1-macro': r['f1_macro'], 'AUC-OVR': r['auc_ovr'],
                     'Acc_std': np.nan, 'F1_std': np.nan, 'AUC_std': np.nan})

    rows.append({'Model': 'Ensemble_Mean',
                 'Accuracy': mean_vals['Accuracy'], 'F1-macro': mean_vals['F1-macro'], 'AUC-OVR': mean_vals['AUC-OVR'],
                 'Acc_std': std_vals['Accuracy'],   'F1_std': std_vals['F1-macro'],   'AUC_std': std_vals['AUC-OVR']})

    summary_df = pd.DataFrame(rows).set_index('Model')
    # ❌ 删除 sort_values，保持原始顺序
    # rows = []
    # for name, r in all_results.items():
    #     rows.append({'Model'   : name,
    #                  'Accuracy': r['acc'],
    #                  'F1-macro': r['f1_macro'],
    #                  'AUC-OVR' : r['auc_ovr']})
    # summary_df = pd.DataFrame(rows).set_index('Model').sort_values('F1-macro', ascending=False)

    print("\n===== 测试集性能汇总 =====")
    print(summary_df.round(4).to_string())
    summary_df.to_csv(model_comparison_text_save_path)
    
    # 最优模型（排除 Ensemble_Mean）
    best_model_name = summary_df.loc[summary_df.index != 'Ensemble_Mean', 'F1-macro'].idxmax()

    # ── 3. 柱状图一：三指标并排 ──────────────────────────────────────
    metrics   = ['Accuracy', 'F1-macro', 'AUC-OVR']
    std_cols  = ['Acc_std',  'F1_std',   'AUC_std']
    model_names = list(summary_df.index)
    n_models  = len(model_names)
    n_metrics = len(metrics)
    x         = np.arange(n_models)
    width     = 0.22
    colors    = ['#4C72B0', '#DD8452', '#55A868']

    fig1, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, std_col, color) in enumerate(zip(metrics, std_cols, colors)):
        vals = summary_df[metric].values
        stds = summary_df[std_col].values
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=metric, color=color,
                      edgecolor='k', linewidth=0.5,
                      yerr=np.where(np.isnan(stds), 0, stds),
                      capsize=3, error_kw={'linewidth': 0.8})
        for bar, v, s in zip(bars, vals, stds):
            label_txt = f'{v:.3f}' if np.isnan(s) else f'{v:.3f}\n±{s:.3f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    label_txt, ha='center', va='bottom', fontsize=7.5)

    # 最优模型加竖线标注
    best_idx = list(summary_df.index).index(best_model_name)
    ax.axvline(x=best_idx, color='gold', linewidth=2, linestyle='--', alpha=0.8, label=f'Best: {best_model_name}')

    ax.set_xticks(x)
    # ★ 最优模型在 x 轴标签上加星
    xticklabels = [f'★ {n}' if n == best_model_name else n for n in model_names]
    ax.set_xticklabels(xticklabels, rotation=30, ha='right')
    ax.set_ylim(0, 1.18)
    ax.set_title('Model Comparison (LGBM excluded)')
    ax.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(model_comparison_figure_save_path, dpi=150, bbox_inches='tight')
    plt.show()


    # ── 4. 柱状图二：每指标单独子图（Nature 配色，不排序）────────────
    nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']
    ensemble_color = '#555555'

    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Model Comparison by Metric (LGBM excluded)', fontsize=13, fontweight='bold', y=1.02)

    for ax_i, (metric, std_col) in zip(axes, zip(metrics, std_cols)):
        # ❌ 不再按当前指标排序，直接用 summary_df 原始顺序
        names = list(summary_df.index)
        vals  = summary_df[metric].values
        stds  = summary_df[std_col].values

        bar_colors = [ensemble_color if n == 'Ensemble_Mean' else nature_colors[j % len(nature_colors)]
                      for j, n in enumerate(names)]
        # 最优模型柱子加金色边框
        edge_colors   = ['gold' if n == best_model_name else 'k' for n in names]
        edge_linewidth = [2.0  if n == best_model_name else 0.5 for n in names]

        bars = ax_i.bar(names, vals,
                        color=bar_colors,
                        edgecolor=edge_colors,
                        linewidth=edge_linewidth,
                        width=0.6,
                        yerr=np.where(np.isnan(stds), 0, stds),
                        capsize=4, error_kw={'linewidth': 0.9})

        for bar, v, s, name in zip(bars, vals, stds, names):
            label_txt = f'{v:.3f}' if np.isnan(s) else f'{v:.3f}\n±{s:.3f}'
            ax_i.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.005,
                      label_txt, ha='center', va='bottom', fontsize=8)

        ymin = max(0, np.nanmin(vals) - 0.05)
        ymax = min(1.0, np.nanmax(vals) + 0.10)
        ax_i.set_ylim(ymin, ymax)
        ax_i.set_title(f'Test {metric}', fontsize=11, fontweight='bold')
        # ★ x 轴标签标注最优模型
        ax_i.set_xticklabels([f'★ {n}' if n == best_model_name else n for n in names],
                              rotation=30, ha='right', fontsize=9)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)

    plt.tight_layout()
    sep_path = model_comparison_figure_save_path.replace('.png', '_separate.png')
    plt.savefig(sep_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ── 5. 最优模型混淆矩阵 ─────────────────────────────────────────
    # n_classes = 4
    # ds_labels = [f'DS{i}' for i in range(n_classes)]

    # cm = confusion_matrix(results[best_model_name]['y_true'],
    #                       results[best_model_name]['y_pred'])
    # cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # fig3, ax = plt.subplots(figsize=(6, 5))
    # sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues', ax=ax,
    #             xticklabels=ds_labels, yticklabels=ds_labels)
    # ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    # ax.set_title(f'Confusion Matrix — ★ {best_model_name} (Best Model)')
    # plt.tight_layout()
    # plt.savefig(confusion_save_path, dpi=150, bbox_inches='tight')
    # plt.show()

    # ── 6. 六模型均值 & 标准差混淆矩阵 ──────────────────────────────
    single_names = [m for m in summary_df.index if m != 'Ensemble_Mean']
    cm_pcts = []
    for m in single_names:
        cm_i = confusion_matrix(results[m]['y_true'], results[m]['y_pred'], labels=list(range(n_classes)))
        cm_pcts.append(cm_i.astype(float) / cm_i.sum(axis=1, keepdims=True))

    cm_pct_mean = np.mean(cm_pcts, axis=0)
    cm_pct_std  = np.std (cm_pcts, axis=0, ddof=1)

    # annot_std = np.array([[f'{cm_pct_mean[r,c]:.1%}\n±{cm_pct_std[r,c]:.1%}'
    annot_std = np.array([[f'±{cm_pct_std[r,c]:.1%}'
                           for c in range(n_classes)]
                          for r in range(n_classes)])

    # 6a. 均值混淆矩阵
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_pct_mean, annot=True, fmt='.1%', cmap='Blues', ax=ax4,
                xticklabels=ds_labels, yticklabels=ds_labels, vmin=0, vmax=1)
    ax4.set_xlabel('Predicted', fontsize=12, labelpad=10)
    ax4.set_ylabel('True', fontsize=12, labelpad=10)
    # ax4.set_title('Mean Confusion Matrix — All 6 Models (row-normalized %)')
    plt.tight_layout()
    mean_cm_path = confusion_save_path.replace('.png', '_ensemble_mean.png')
    plt.savefig(mean_cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 均值混淆矩阵已保存为 {mean_cm_path}")

    # 6b. 标准差混淆矩阵
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_pct_std, annot=annot_std, fmt='', cmap='Oranges', ax=ax5,
                xticklabels=ds_labels, yticklabels=ds_labels,
                linewidths=0.5, linecolor='white')
    ax5.set_xlabel('Predicted', fontsize=12, labelpad=10)
    ax5.set_ylabel('True', fontsize=12, labelpad=10)
    # ax5.set_title('Std Dev Confusion Matrix — All 6 Models\n(color = std, annotation = mean ± std)')
    plt.tight_layout()
    std_cm_path = confusion_save_path.replace('.png', '_ensemble_std.png')
    plt.savefig(std_cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 标准差混淆矩阵已保存为 {std_cm_path}")

    # ── 7. 所有模型混淆矩阵子图（原始顺序，末尾追加 Mean & Std）──────
    plot_items = single_names + ['[Mean]', '[Std]']
    n_items    = len(plot_items)
    n_cols     = 4
    n_rows     = int(np.ceil(n_items / n_cols))

    fig6, axes6 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows))
    axes6 = np.array(axes6).flatten()

    for idx, item in enumerate(plot_items):
        ax_i = axes6[idx]

        # 默认设置
        v_min, v_max = 0, 1

        if item == '[Mean]':
            data  = cm_pct_mean
            annot = np.array([[f'{v:.1%}' for v in row] for row in cm_pct_mean])
            cmap  = 'Blues'
            # 计算 Mean 的整体 Accuracy (这里取单模型 Acc 的均值作为展示)
            avg_acc = summary_df.loc[single_names, 'Accuracy'].mean()
            title = f'Ensemble Mean\nAvg Acc: {avg_acc:.3f} , F1: {summary_df.loc["Ensemble_Mean", "F1-macro"]:.3f}'
            # fw    = 'bold'
        elif item == '[Std]':
            data  = cm_pct_std
            annot = annot_std
            cmap  = 'Oranges'
            v_max  = data.max() if data.max() > 0 else 1 
            avg_std_acc = summary_df.loc[single_names, 'Accuracy'].std()
            title = f'Ensemble Std\nAvg Acc: {avg_std_acc:.3f} , F1: {summary_df.loc["Ensemble_Mean", "F1_std"]:.3f}'
            # fw    = 'bold'
        else:
            cm_i  = confusion_matrix(results[item]['y_true'], results[item]['y_pred'],
                                     labels=list(range(n_classes)))
            data  = cm_i.astype(float) / cm_i.sum(axis=1, keepdims=True)
            annot = np.array([[f'{cm_i[r,c]}\n{data[r,c]:.1%}' for c in range(n_classes)]
                              for r in range(n_classes)])
            cmap  = 'Greys'
            f1_val = summary_df.loc[item, 'F1-macro']
            acc_val = summary_df.loc[item, 'Accuracy']
            # ★ 只标注最优，不排序
            prefix = '★ ' if item == best_model_name else ''
            title  = f'{prefix}{item}\n Acc={acc_val:.3f} , F1={f1_val:.3f}'
            fw     = 'bold' if item == best_model_name else 'normal'

        sns.heatmap(data, annot=annot, fmt='', cmap=cmap, ax=ax_i,
                    xticklabels=ds_labels, yticklabels=ds_labels,
                    vmin=v_min, vmax=v_max, linewidths=0.5, linecolor='white', cbar=False, annot_kws={"size": 9})
        ax_i.set_title(title, fontsize=11, fontweight=fw)
        ax_i.set_xlabel('Predicted', fontsize=11)
        ax_i.set_ylabel('True', fontsize=11)

    for idx in range(n_items, len(axes6)):
        axes6[idx].set_visible(False)

    # fig6.suptitle('Confusion Matrices — All Models (LGBM excluded, row-normalized %)',
    #               fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(h_pad=3, w_pad=2)
    all_cm_path = confusion_save_path.replace('.png', '_separate.png')
    plt.savefig(all_cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 所有模型混淆矩阵已保存为 {all_cm_path}")

    return summary_df

# summary_df = plot_comparison(all_results)


# ============================================================
# Step 7: SHAP 可解释性
# ============================================================
def plot_shap(trained_pipes: dict, X_test, best_model_name: str, 
              dsf_features=DSF_FEATURES, save_path="shap_summary.png"):
    pipe  = trained_pipes[best_model_name]
    model = pipe.named_steps['clf']
    X_sc  = pipe.named_steps['scaler'].transform(X_test)
    X_sc_df = pd.DataFrame(X_sc, columns=dsf_features)  # 带列名方便显示

    if any(kw in best_model_name for kw in ['LightGBM', 'LGBM', 'XGBoost', 'RF', 'RandomForest']):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sc_df)
    else:
        explainer   = shap.KernelExplainer(model.predict_proba, shap.sample(X_sc, 100))
        shap_values = explainer.shap_values(X_sc[:])
        X_sc_df     = X_sc_df.iloc[:]

    # 多分类：shap_values 是 list，每个元素对应一个类
    if isinstance(shap_values, list):
        print(f"shap_values 类型: {type(shap_values)}")
        print(f"shap_values 长度: {len(shap_values)}")
        print(f"每个元素 shape: {shap_values[0].shape}")

        # 取所有类的绝对值均值，得到整体特征重要性
        shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        
        # 画法1：整体重要性（推荐，最直观）
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_mean, X_sc_df, feature_names=dsf_features,
                          plot_type='bar', show=False)
        plt.title(f'SHAP Feature Importance — {best_model_name}')
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 画法2：每个类单独一张蜂群图
        class_names = [f'DS{i}' for i in range(len(shap_values))]
        for i, (sv, cname) in enumerate(zip(shap_values, class_names)):
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X_sc_df, feature_names=dsf_features,
                              plot_type='dot', show=False)
            plt.title(f'SHAP — {best_model_name} | Class {cname}')
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', f'_class{i}.png'), dpi=300, bbox_inches='tight')
            plt.show()
    else:
        # KernelExplainer 返回 (n_samples, n_features, n_classes)
        # 转成 list，每个元素是 (n_samples, n_features)，对应一个类
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        
        # 画法1：整体重要性柱状图
        shap_mean = np.mean([np.abs(sv) for sv in shap_values_list], axis=0)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_mean, X_sc_df, feature_names=dsf_features,
                          plot_type='bar', show=False)
        plt.title(f'SHAP Feature Importance — {best_model_name}')
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_bar.png'), dpi=150, bbox_inches='tight')
        plt.show()

        # 画法2：每个类单独蜂群图
        class_names = [f'DS{i}' for i in range(len(shap_values_list))]
        for i, (sv, cname) in enumerate(zip(shap_values_list, class_names)):
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X_sc_df, feature_names=dsf_features,
                              plot_type='dot', show=False)
            plt.title(f'SHAP — {best_model_name} | Class {cname}')
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', f'_class{i}.png'), dpi=150, bbox_inches='tight')
            plt.show()


# best_model_name = summary_df['F1-macro'].idxmax()
# print(f"\n最优模型: {best_model_name}")
# plot_shap(trained_pipes, X_test, best_model_name)

# ============================================================
# Step 8: 主流程
# ============================================================

def main(csv_path='population_dataset.csv',
         results_dir='saved_results',
         run_optuna=True,
         n_trials=N_OPTUNA_TRIALS,
         n_cv_splits=N_CV_SPLITS,
         random_state=RANDOM_STATE,
         test_ratio=0.2,
         dsf_features=None,
         target=TARGET,           
         group_col=GROUP_COL):
    
    if dsf_features is None:
        dsf_features = DSF_FEATURES

    os.makedirs(results_dir, exist_ok=True)
    params_path   = os.path.join(results_dir, 'best_params.json')
    results_path  = os.path.join(results_dir, 'all_results.json')
    pipes_path    = os.path.join(results_dir, 'trained_pipes.pkl')
    summary_path  = os.path.join(results_dir, 'summary_df.csv')

    # Step 1: 加载预处理
    X, y, groups, df = load_and_preprocess(csv_path, 
                                            dsf_features=dsf_features,
                                            target=target, 
                                            group_col=group_col)
    
    # Step 2: 特征分析
    feature_analysis(df, dsf_features=dsf_features, target=target,
                     save_path=os.path.join(results_dir, 'feature_analysis.png'))
    
    # Step 3: 数据划分（按模型ID分组）
    (X_train, y_train, groups_train, X_test,  y_test) = group_train_val_test_split(
                                X, y, groups, test_ratio=test_ratio, random_state=random_state)
    
    # Step 4: 超参数优化
    if os.path.exists(params_path):
        print(f"[Optuna] 发现缓存，直接加载: {params_path}")
        with open(params_path, 'r') as f:
            best_params = json.load(f)
            
    elif run_optuna:
        best_params = {}
        optuna_tasks = {
            'XGBoost':            optimize_xgboost,
            'RandomForest':       optimize_rf,
            'SVM':                optimize_svm,
            'MLP':                optimize_mlp,
            'KNN':                optimize_knn,
            'LogisticRegression': optimize_lr,
        }
        for model_name, opt_fn in optuna_tasks.items():
            print(f"\n[Optuna] 正在优化 {model_name} ...")
            best_params[model_name] = opt_fn(X_train, y_train, groups_train,
                                            n_trials=n_trials,
                                            n_cv_splits=n_cv_splits,
                                            random_state=random_state)
                # 保存超参数
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"[Optuna] 超参数已保存至 {params_path}")
    else:
        best_params = {}
    
    # MLP 参数预处理（必须在 run_all_models 之前）
    mlp_raw_params = best_params.get('MLP', {}).copy()
    if mlp_raw_params and 'n_layers' in mlp_raw_params:
        n_layers = mlp_raw_params.pop('n_layers')
        hidden_layer_sizes = tuple(mlp_raw_params.pop(f'n_units_l{i}') for i in range(n_layers))
        mlp_raw_params['hidden_layer_sizes'] = hidden_layer_sizes
    else:
        mlp_raw_params = {'hidden_layer_sizes': (128, 64)}  # 默认值


    # Step 5: 训练所有模型
    if os.path.exists(pipes_path) and os.path.exists(results_path):
        print(f"[模型训练] 发现缓存，直接加载: {pipes_path} 和 {results_path}")
        trained_pipes = joblib.load(pipes_path)
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results, trained_pipes  = run_all_models(
            X_train, y_train, groups_train, X_test, y_test, best_params, mlp_raw_params)
        # 保存管道 & 结果
        joblib.dump(trained_pipes, pipes_path)
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        with open(results_path, 'w') as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
        print(f"[Train] 模型和结果已保存至 {results_dir}/")

    # Step 6: 可视化对比
    summary_df = plot_comparison(all_results,
                    model_comparison_text_save_path=os.path.join(results_dir, 'model_comparison.csv'),
                    model_comparison_figure_save_path=os.path.join(results_dir, 'model_comparison.png'),
                    confusion_save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    summary_df.to_csv(summary_path)
    
    # # Step 7: SHAP可解释性
    # EXCLUDE_SHAP = {'LGBM', 'LightGBM'}
    # for model_name in trained_pipes:
    #     if model_name in EXCLUDE_SHAP:
    #         continue
    #     print(f"\n[SHAP] 正在处理: {model_name} ...")
    #     plot_shap(
    #         trained_pipes, X_test, model_name,
    #         dsf_features=dsf_features, save_path=os.path.join(results_dir, f'shap_summary_{model_name}.png')
    #     )

    # best_model_name = summary_df['F1-macro'].idxmax()
    # print(f"\n最优模型: {best_model_name}")
    # plot_shap(trained_pipes , X_test, best_model_name, 
    #           dsf_features=dsf_features, save_path=os.path.join(results_dir, 'shap_summary.png'))
    
    # Step 8: 保存最优模型
    # best_path = os.path.join(results_dir, 'best_ds_classifier.pkl')
    # joblib.dump({'pipeline': trained_pipes[best_model_name],
    #              'features':  dsf_features,
    #              'best_name': best_model_name},
    #             best_path)
    # print(f"\n✅ 最优模型已保存至 {best_path}")
    
    # 推理示例
    # print("\n📋 推理示例（直接输入原始特征，无需手动标准化）:")
    # pipe = trained_pipes[best_model_name]
    # for i, (pred, prob) in enumerate(
    #         zip(pipe.predict(X_test[:3]),
    #             pipe.predict_proba(X_test[:3]))):
    #     print(f"  样本{i+1}: 预测 DS={pred}，各类概率={np.round(prob, 3)}")
    
    return trained_pipes, all_results, summary_df

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',     type=str,   default='population_dataset.csv')
    parser.add_argument('--results_dir',  type=str,   default='saved_results')
    parser.add_argument('--run_optuna',   type=int,   default=1)   # 1=True, 0=False
    parser.add_argument('--n_trials',     type=int,   default=100)
    parser.add_argument('--n_cv_splits',  type=int,   default=5)
    parser.add_argument('--random_state', type=int,   default=123)
    parser.add_argument('--test_ratio',   type=float, default=0.2)
    parser.add_argument('--target',       type=str,   default='damage_state')
    parser.add_argument('--group_col',    type=str,   default='model_id')
    parser.add_argument('--dsf_features', type=str,   default=None,
                        help='逗号分隔的特征名，如 DSF_T1_AREA,DSF_dF1peak')
    args = parser.parse_args()

    # 解析特征列表
    if args.dsf_features is not None:
        dsf_features = args.dsf_features.split(',')
    else:
        dsf_features = None  # 用默认值

    trained_pipes, results, summary = main(
        csv_path=args.csv_path,
        results_dir=args.results_dir,
        run_optuna=args.run_optuna,
        n_trials=args.n_trials,
        n_cv_splits=args.n_cv_splits,
        random_state=args.random_state,
        test_ratio=args.test_ratio,
        target=args.target,
        group_col=args.group_col,
        dsf_features=dsf_features
    )
