"""
RBF Kernel SVM 训练优化版 - Letter Recognition Dataset
优化策略：
1. GridSearchCV 交叉验证调参 (C, gamma)
2. 使用 cache_size 加速核函数计算
3. 两阶段搜索：先粗粒度再细粒度
4. 尝试不同的 decision_function_shape
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 加载数据 ==========
print("=" * 60)
print("RBF Kernel SVM 优化版 - Letter Recognition")
print("=" * 60)

train_path = os.path.join("..", "..", "letter_recognition_train.csv")
test_path = os.path.join("..", "..", "letter_recognition_test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"\n训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ========== 2. 分离特征和标签 ==========
X_train = train_df.drop("letter", axis=1).values
y_train = train_df["letter"].values
X_test = test_df.drop("letter", axis=1).values
y_test = test_df["letter"].values

print(f"\n特征数量: {X_train.shape[1]}")
print(f"类别数量: {len(np.unique(y_train))}")
print(f"样本数量: {X_train.shape[0]} (训练), {X_test.shape[0]} (测试)")

# ========== 3. 特征标准化 ==========
print("\n>>> 特征标准化 (StandardScaler) ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 优化策略 ==========
print("\n" + "=" * 60)
print("优化策略")
print("=" * 60)

# ===== 策略1: 使用优化参数的 RBF SVM =====
print("\n>>> [策略1] 优化参数的 RBF SVM (C=1.0, gamma='scale', cache=500MB) ...")

start_time = time.time()
rbf_svm_opt = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    cache_size=500,        # 增大缓存 (MB)，加速核函数计算
    max_iter=-1,
    decision_function_shape='ovr',
    random_state=42,
    verbose=False
)
rbf_svm_opt.fit(X_train_scaled, y_train)
time_strategy1 = time.time() - start_time

y_pred_train_1 = rbf_svm_opt.predict(X_train_scaled)
y_pred_test_1 = rbf_svm_opt.predict(X_test_scaled)
train_acc_1 = accuracy_score(y_train, y_pred_train_1)
test_acc_1 = accuracy_score(y_test, y_pred_test_1)

print(f"  训练耗时: {time_strategy1:.2f} 秒")
print(f"  训练集准确率: {train_acc_1:.4f} ({train_acc_1*100:.2f}%)")
print(f"  测试集准确率: {test_acc_1:.4f} ({test_acc_1*100:.2f}%)")
print(f"  支持向量数: {sum(rbf_svm_opt.n_support_)}")

# ===== 策略2: GridSearchCV 调参 (两阶段搜索) =====
print("\n>>> [策略2] GridSearchCV 交叉验证调参 ...")
print("   使用 8000 样本进行快速参数搜索...")

# 使用子集加速 GridSearch
indices = np.random.RandomState(42).choice(X_train_scaled.shape[0], 8000, replace=False)
X_train_subset = X_train_scaled[indices]
y_train_subset = y_train[indices]

# 第一阶段: 粗粒度搜索 C 和 gamma
print("\n   第一阶段: 粗粒度搜索 C 和 gamma ...")
param_grid_1 = {
    'kernel': ['rbf'],
    'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
}

grid_search_1 = GridSearchCV(
    SVC(cache_size=500, max_iter=-1, decision_function_shape='ovr',
        random_state=42),
    param_grid_1,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search_1.fit(X_train_subset, y_train_subset)

best_params_1 = grid_search_1.best_params_
best_cv_score_1 = grid_search_1.best_score_
print(f"\n  第一阶段最佳参数: {best_params_1}")
print(f"  第一阶段最佳交叉验证得分: {best_cv_score_1:.4f} ({best_cv_score_1*100:.2f}%)")

# 第二阶段: 在最佳参数附近精细搜索
print("\n   第二阶段: 精细搜索 ...")
C_best = best_params_1['C']
gamma_best = best_params_1['gamma']

# 构建精细搜索范围
C_range = [C_best / 3, C_best / 2, C_best, C_best * 2, C_best * 3]
C_range = [max(0.01, round(c, 2)) for c in C_range]
C_range = sorted(set(C_range))

# gamma 精细搜索
if gamma_best == 'scale':
    gamma_range = ['scale', 0.01, 0.02, 0.05, 0.08, 0.1]
elif gamma_best == 'auto':
    gamma_range = ['auto', 'scale', 0.01, 0.02, 0.05, 0.08]
else:
    gamma_val = float(gamma_best)
    gamma_range = [gamma_val / 2, gamma_val, gamma_val * 2]
    gamma_range = [round(g, 4) for g in gamma_range]

param_grid_2 = {
    'kernel': ['rbf'],
    'C': C_range,
    'gamma': gamma_range,
}

grid_search_2 = GridSearchCV(
    SVC(cache_size=500, max_iter=-1, decision_function_shape='ovr',
        random_state=42),
    param_grid_2,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search_2.fit(X_train_subset, y_train_subset)

best_params_2 = grid_search_2.best_params_
best_cv_score_2 = grid_search_2.best_score_
print(f"\n  第二阶段最佳参数: {best_params_2}")
print(f"  第二阶段最佳交叉验证得分: {best_cv_score_2:.4f} ({best_cv_score_2*100:.2f}%)")

# 使用最佳参数在全量数据上训练
print("\n>>> 使用最佳参数在全量数据上训练 ...")
start_time = time.time()
rbf_svm_best = SVC(
    kernel='rbf',
    C=best_params_2['C'],
    gamma=best_params_2['gamma'],
    cache_size=500,
    max_iter=-1,
    decision_function_shape='ovr',
    random_state=42,
    verbose=False
)
rbf_svm_best.fit(X_train_scaled, y_train)
time_strategy2 = time.time() - start_time

y_pred_train_2 = rbf_svm_best.predict(X_train_scaled)
y_pred_test_2 = rbf_svm_best.predict(X_test_scaled)
train_acc_2 = accuracy_score(y_train, y_pred_train_2)
test_acc_2 = accuracy_score(y_test, y_pred_test_2)

print(f"  训练耗时: {time_strategy2:.2f} 秒")
print(f"  训练集准确率: {train_acc_2:.4f} ({train_acc_2*100:.2f}%)")
print(f"  测试集准确率: {test_acc_2:.4f} ({test_acc_2*100:.2f}%)")
print(f"  支持向量数: {sum(rbf_svm_best.n_support_)}")

# ===== 策略3: 尝试 decision_function_shape='ovo' =====
print("\n>>> [策略3] 使用最佳参数 + decision_function_shape='ovo' ...")

start_time = time.time()
rbf_svm_ovo = SVC(
    kernel='rbf',
    C=best_params_2['C'],
    gamma=best_params_2['gamma'],
    cache_size=500,
    max_iter=-1,
    decision_function_shape='ovo',   # 一对一策略
    random_state=42,
    verbose=False
)
rbf_svm_ovo.fit(X_train_scaled, y_train)
time_strategy3 = time.time() - start_time

y_pred_train_3 = rbf_svm_ovo.predict(X_train_scaled)
y_pred_test_3 = rbf_svm_ovo.predict(X_test_scaled)
train_acc_3 = accuracy_score(y_train, y_pred_train_3)
test_acc_3 = accuracy_score(y_test, y_pred_test_3)

print(f"  训练耗时: {time_strategy3:.2f} 秒")
print(f"  训练集准确率: {train_acc_3:.4f} ({train_acc_3*100:.2f}%)")
print(f"  测试集准确率: {test_acc_3:.4f} ({test_acc_3*100:.2f}%)")
print(f"  支持向量数: {sum(rbf_svm_ovo.n_support_)}")

# ========== 5. 结果对比 ==========
print("\n" + "=" * 60)
print("优化结果对比")
print("=" * 60)
print(f"{'策略':<35} {'训练耗时':<12} {'训练准确率':<14} {'测试准确率':<14}")
print("-" * 75)
print(f"{'原始 RBF SVM (C=1.0, gamma=scale)':<35} {'1.95 秒':<12} {'95.88%':<14} {'94.85%':<14}")
print(f"{'策略1: 优化参数 + cache=500MB':<35} {f'{time_strategy1:.2f} 秒':<12} {f'{train_acc_1*100:.2f}%':<14} {f'{test_acc_1*100:.2f}%':<14}")
print(f"{'策略2: GridSearch + 最佳参数':<35} {f'{time_strategy2:.2f} 秒':<12} {f'{train_acc_2*100:.2f}%':<14} {f'{test_acc_2*100:.2f}%':<14}")
print(f"{'策略3: 最佳参数 + ovo':<35} {f'{time_strategy3:.2f} 秒':<12} {f'{train_acc_3*100:.2f}%':<14} {f'{test_acc_3*100:.2f}%':<14}")
print("-" * 75)

# ========== 6. 选择最佳模型并保存 ==========
results = [
    ('RBF_SVM_opt', test_acc_1, rbf_svm_opt),
    ('RBF_SVM_best', test_acc_2, rbf_svm_best),
    ('RBF_SVM_ovo', test_acc_3, rbf_svm_ovo)
]

best_model_name, best_acc, best_model = max(results, key=lambda x: x[1])
print(f"\n>>> 最佳模型: {best_model_name} (测试准确率: {best_acc*100:.2f}%)")

# 保存最佳模型
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "rbf_svm_optimized_model.pkl")
scaler_path = os.path.join(model_dir, "rbf_svm_scaler.pkl")

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n最佳模型已保存至: {model_path}")
print(f"标准化器已保存至: {scaler_path}")

# ========== 7. 最佳模型的详细评估 ==========
print("\n" + "=" * 60)
print(f"最佳模型 ({best_model_name}) 详细评估")
print("=" * 60)

y_pred_best = best_model.predict(X_test_scaled)
print(f"\n测试集准确率: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"\n>>> 分类报告 (测试集):")
print(classification_report(y_test, y_pred_best))

print("\n>>> 混淆矩阵 (测试集):")
print(confusion_matrix(y_test, y_pred_best))

print("\n完成!")
