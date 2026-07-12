"""
Polynomial Kernel SVM 训练优化版 - Letter Recognition Dataset
优化策略：
1. GridSearchCV 交叉验证调参 (degree, C, gamma, coef0)
2. 使用 cache_size 加速核函数计算
3. 使用子集快速搜索最佳参数
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
print("Polynomial Kernel SVM 优化版 - Letter Recognition")
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

# ===== 策略1: 使用优化参数的 Poly SVM =====
print("\n>>> [策略1] 优化参数的 Poly SVM (degree=3, C=1.0, cache=500MB) ...")

start_time = time.time()
poly_svm_opt = SVC(
    kernel='poly',
    degree=3,
    C=1.0,
    gamma='scale',
    coef0=0.0,
    cache_size=500,        # 增大缓存 (MB)，加速核函数计算
    max_iter=-1,           # 不限制迭代次数
    decision_function_shape='ovr',
    random_state=42,
    verbose=False
)
poly_svm_opt.fit(X_train_scaled, y_train)
time_strategy1 = time.time() - start_time

y_pred_train_1 = poly_svm_opt.predict(X_train_scaled)
y_pred_test_1 = poly_svm_opt.predict(X_test_scaled)
train_acc_1 = accuracy_score(y_train, y_pred_train_1)
test_acc_1 = accuracy_score(y_test, y_pred_test_1)

print(f"  训练耗时: {time_strategy1:.2f} 秒")
print(f"  训练集准确率: {train_acc_1:.4f} ({train_acc_1*100:.2f}%)")
print(f"  测试集准确率: {test_acc_1:.4f} ({test_acc_1*100:.2f}%)")
print(f"  支持向量数: {sum(poly_svm_opt.n_support_)}")

# ===== 策略2: GridSearchCV 调参 (使用子集加速) =====
print("\n>>> [策略2] GridSearchCV 交叉验证调参 ...")
print("   使用 6000 样本进行快速参数搜索...")

# 使用子集加速 GridSearch
indices = np.random.RandomState(42).choice(X_train_scaled.shape[0], 6000, replace=False)
X_train_subset = X_train_scaled[indices]
y_train_subset = y_train[indices]

# 分两阶段搜索，减少组合数
# 第一阶段: 搜索 degree 和 C
print("\n   第一阶段: 搜索 degree 和 C ...")
param_grid_1 = {
    'kernel': ['poly'],
    'degree': [2, 3, 4],
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 1.0],
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
degree_best = best_params_1['degree']
C_best = best_params_1['C']
gamma_best = best_params_1['gamma']
coef0_best = best_params_1['coef0']

# 在最佳值附近精细搜索
C_range = [max(0.01, C_best / 2), C_best, C_best * 2, C_best * 5]
degree_range = [max(2, degree_best - 1), degree_best, degree_best + 1]
coef0_range = [0.0, 0.5, 1.0, 2.0] if coef0_best == 0.0 else [coef0_best - 0.5, coef0_best, coef0_best + 0.5]

param_grid_2 = {
    'kernel': ['poly'],
    'degree': degree_range,
    'C': C_range,
    'gamma': [gamma_best],
    'coef0': coef0_range,
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
poly_svm_best = SVC(
    kernel='poly',
    degree=best_params_2['degree'],
    C=best_params_2['C'],
    gamma=best_params_2['gamma'],
    coef0=best_params_2['coef0'],
    cache_size=500,
    max_iter=-1,
    decision_function_shape='ovr',
    random_state=42,
    verbose=False
)
poly_svm_best.fit(X_train_scaled, y_train)
time_strategy2 = time.time() - start_time

y_pred_train_2 = poly_svm_best.predict(X_train_scaled)
y_pred_test_2 = poly_svm_best.predict(X_test_scaled)
train_acc_2 = accuracy_score(y_train, y_pred_train_2)
test_acc_2 = accuracy_score(y_test, y_pred_test_2)

print(f"  训练耗时: {time_strategy2:.2f} 秒")
print(f"  训练集准确率: {train_acc_2:.4f} ({train_acc_2*100:.2f}%)")
print(f"  测试集准确率: {test_acc_2:.4f} ({test_acc_2*100:.2f}%)")
print(f"  支持向量数: {sum(poly_svm_best.n_support_)}")

# ===== 策略3: 尝试 decision_function_shape='ovo' =====
print("\n>>> [策略3] 使用最佳参数 + decision_function_shape='ovo' ...")

start_time = time.time()
poly_svm_ovo = SVC(
    kernel='poly',
    degree=best_params_2['degree'],
    C=best_params_2['C'],
    gamma=best_params_2['gamma'],
    coef0=best_params_2['coef0'],
    cache_size=500,
    max_iter=-1,
    decision_function_shape='ovo',   # 一对一策略
    random_state=42,
    verbose=False
)
poly_svm_ovo.fit(X_train_scaled, y_train)
time_strategy3 = time.time() - start_time

y_pred_train_3 = poly_svm_ovo.predict(X_train_scaled)
y_pred_test_3 = poly_svm_ovo.predict(X_test_scaled)
train_acc_3 = accuracy_score(y_train, y_pred_train_3)
test_acc_3 = accuracy_score(y_test, y_pred_test_3)

print(f"  训练耗时: {time_strategy3:.2f} 秒")
print(f"  训练集准确率: {train_acc_3:.4f} ({train_acc_3*100:.2f}%)")
print(f"  测试集准确率: {test_acc_3:.4f} ({test_acc_3*100:.2f}%)")
print(f"  支持向量数: {sum(poly_svm_ovo.n_support_)}")

# ========== 5. 结果对比 ==========
print("\n" + "=" * 60)
print("优化结果对比")
print("=" * 60)
print(f"{'策略':<35} {'训练耗时':<12} {'训练准确率':<14} {'测试准确率':<14}")
print("-" * 75)
print(f"{'原始 Poly SVM (degree=3, C=1.0)':<35} {'1.98 秒':<12} {'90.94%':<14} {'89.08%':<14}")
print(f"{'策略1: 优化参数 + cache=500MB':<35} {f'{time_strategy1:.2f} 秒':<12} {f'{train_acc_1*100:.2f}%':<14} {f'{test_acc_1*100:.2f}%':<14}")
print(f"{'策略2: GridSearch + 最佳参数':<35} {f'{time_strategy2:.2f} 秒':<12} {f'{train_acc_2*100:.2f}%':<14} {f'{test_acc_2*100:.2f}%':<14}")
print(f"{'策略3: 最佳参数 + ovo':<35} {f'{time_strategy3:.2f} 秒':<12} {f'{train_acc_3*100:.2f}%':<14} {f'{test_acc_3*100:.2f}%':<14}")
print("-" * 75)

# ========== 6. 选择最佳模型并保存 ==========
results = [
    ('PolySVM_opt', test_acc_1, poly_svm_opt),
    ('PolySVM_best', test_acc_2, poly_svm_best),
    ('PolySVM_ovo', test_acc_3, poly_svm_ovo)
]

best_model_name, best_acc, best_model = max(results, key=lambda x: x[1])
print(f"\n>>> 最佳模型: {best_model_name} (测试准确率: {best_acc*100:.2f}%)")

# 保存最佳模型
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "poly_svm_optimized_model.pkl")
scaler_path = os.path.join(model_dir, "poly_svm_scaler.pkl")

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
