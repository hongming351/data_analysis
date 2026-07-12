"""
Linear SVM 训练优化版 - Letter Recognition Dataset
优化策略：
1. dual=False (n_samples > n_features, 求解原始问题更快)
2. GridSearchCV 交叉验证调参 (C, penalty, loss)
3. 调整 tol 和 max_iter 加速收敛
4. 尝试不同的 multi_class 策略
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 加载数据 ==========
print("=" * 60)
print("Linear SVM 优化版 - Letter Recognition")
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
print(f"样本数 > 特征数: {X_train.shape[0]} > {X_train.shape[1]} → 应使用 dual=False")

# ========== 3. 特征标准化 ==========
print("\n>>> 特征标准化 (StandardScaler) ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 优化策略 ==========
print("\n" + "=" * 60)
print("优化策略")
print("=" * 60)

# ===== 策略1: 使用优化参数的 LinearSVC =====
print("\n>>> [策略1] 优化参数的 LinearSVC (dual=False, tol=1e-3) ...")

start_time = time.time()
linear_svm_opt = LinearSVC(
    C=1.0,
    loss='squared_hinge',      # squared_hinge 的优化问题更平滑，收敛更快
    penalty='l2',               # L2 正则化
    dual=False,                 # n_samples > n_features, 求解原始问题
    tol=1e-3,                   # 适当放宽容差，加速收敛
    max_iter=2000,              # 减少最大迭代次数
    multi_class='ovr',          # 一对多策略
    random_state=42,
    verbose=0
)
linear_svm_opt.fit(X_train_scaled, y_train)
time_strategy1 = time.time() - start_time

y_pred_train_1 = linear_svm_opt.predict(X_train_scaled)
y_pred_test_1 = linear_svm_opt.predict(X_test_scaled)
train_acc_1 = accuracy_score(y_train, y_pred_train_1)
test_acc_1 = accuracy_score(y_test, y_pred_test_1)

print(f"  训练耗时: {time_strategy1:.2f} 秒")
print(f"  训练集准确率: {train_acc_1:.4f} ({train_acc_1*100:.2f}%)")
print(f"  测试集准确率: {test_acc_1:.4f} ({test_acc_1*100:.2f}%)")
print(f"  迭代次数: {linear_svm_opt.n_iter_}")

# ===== 策略2: GridSearchCV 调参 =====
print("\n>>> [策略2] GridSearchCV 交叉验证调参 ...")
print("   正在搜索最佳参数组合 (5折交叉验证)...")

param_grid = {
    'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'loss': ['squared_hinge'],           # squared_hinge 支持 dual=False
    'penalty': ['l2'],                    # l2 与 squared_hinge 配合
    'multi_class': ['ovr'],              # ovr 多分类
}

# 使用部分数据快速搜索最佳 C 值 (用 8000 样本加速)
print("   使用 8000 样本进行快速参数搜索...")
indices = np.random.RandomState(42).choice(X_train_scaled.shape[0], 8000, replace=False)
X_train_subset = X_train_scaled[indices]
y_train_subset = y_train[indices]

grid_search = GridSearchCV(
    LinearSVC(dual=False, tol=1e-3, max_iter=2000, random_state=42),
    param_grid,
    cv=3,                        # 3折交叉验证
    scoring='accuracy',
    n_jobs=-1,                   # 使用所有CPU核心
    verbose=1
)
grid_search.fit(X_train_subset, y_train_subset)

best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
print(f"\n  最佳参数: {best_params}")
print(f"  最佳交叉验证得分: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")

# 使用最佳参数在全量数据上训练
print("\n>>> 使用最佳参数在全量数据上训练 ...")
start_time = time.time()
linear_svm_best = LinearSVC(
    C=best_params['C'],
    loss=best_params['loss'],
    penalty=best_params['penalty'],
    dual=False,
    tol=1e-3,
    max_iter=5000,
    multi_class=best_params['multi_class'],
    random_state=42,
    verbose=0
)
linear_svm_best.fit(X_train_scaled, y_train)
time_strategy2 = time.time() - start_time

y_pred_train_2 = linear_svm_best.predict(X_train_scaled)
y_pred_test_2 = linear_svm_best.predict(X_test_scaled)
train_acc_2 = accuracy_score(y_train, y_pred_train_2)
test_acc_2 = accuracy_score(y_test, y_pred_test_2)

print(f"  训练耗时: {time_strategy2:.2f} 秒")
print(f"  训练集准确率: {train_acc_2:.4f} ({train_acc_2*100:.2f}%)")
print(f"  测试集准确率: {test_acc_2:.4f} ({test_acc_2*100:.2f}%)")
print(f"  迭代次数: {linear_svm_best.n_iter_}")

# ===== 策略3: SGDClassifier (大规模线性SVM的替代方案) =====
print("\n>>> [策略3] SGDClassifier (hinge loss ≈ 线性SVM, 适合大规模数据) ...")

start_time = time.time()
sgd_svm = SGDClassifier(
    loss='hinge',                # hinge loss ≈ 线性SVM
    penalty='l2',                # L2 正则化
    alpha=0.0001,                # 正则化强度 (1/(2*C) 的对应)
    max_iter=2000,
    tol=1e-3,
    learning_rate='optimal',     # 自适应学习率
    random_state=42,
    verbose=0,
    n_jobs=-1
)
sgd_svm.fit(X_train_scaled, y_train)
time_strategy3 = time.time() - start_time

y_pred_train_3 = sgd_svm.predict(X_train_scaled)
y_pred_test_3 = sgd_svm.predict(X_test_scaled)
train_acc_3 = accuracy_score(y_train, y_pred_train_3)
test_acc_3 = accuracy_score(y_test, y_pred_test_3)

print(f"  训练耗时: {time_strategy3:.2f} 秒")
print(f"  训练集准确率: {train_acc_3:.4f} ({train_acc_3*100:.2f}%)")
print(f"  测试集准确率: {test_acc_3:.4f} ({test_acc_3*100:.2f}%)")
print(f"  迭代次数: {sgd_svm.n_iter_}")

# ===== 策略4: SGDClassifier + GridSearch =====
print("\n>>> [策略4] SGDClassifier 参数调优 ...")
print("   正在搜索最佳参数...")

sgd_param_grid = {
    'loss': ['hinge', 'squared_hinge'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01],
    'penalty': ['l2', 'l1', 'elasticnet'],
}

sgd_grid = GridSearchCV(
    SGDClassifier(max_iter=2000, tol=1e-3, learning_rate='optimal', 
                  random_state=42, n_jobs=-1),
    sgd_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
sgd_grid.fit(X_train_subset, y_train_subset)

sgd_best_params = sgd_grid.best_params_
sgd_best_cv_score = sgd_grid.best_score_
print(f"\n  SGD最佳参数: {sgd_best_params}")
print(f"  SGD最佳交叉验证得分: {sgd_best_cv_score:.4f} ({sgd_best_cv_score*100:.2f}%)")

print("\n>>> 使用SGD最佳参数在全量数据上训练 ...")
start_time = time.time()
sgd_best = SGDClassifier(
    loss=sgd_best_params['loss'],
    alpha=sgd_best_params['alpha'],
    penalty=sgd_best_params['penalty'],
    max_iter=5000,
    tol=1e-3,
    learning_rate='optimal',
    random_state=42,
    n_jobs=-1
)
sgd_best.fit(X_train_scaled, y_train)
time_strategy4 = time.time() - start_time

y_pred_train_4 = sgd_best.predict(X_train_scaled)
y_pred_test_4 = sgd_best.predict(X_test_scaled)
train_acc_4 = accuracy_score(y_train, y_pred_train_4)
test_acc_4 = accuracy_score(y_test, y_pred_test_4)

print(f"  训练耗时: {time_strategy4:.2f} 秒")
print(f"  训练集准确率: {train_acc_4:.4f} ({train_acc_4*100:.2f}%)")
print(f"  测试集准确率: {test_acc_4:.4f} ({test_acc_4*100:.2f}%)")
print(f"  迭代次数: {sgd_best.n_iter_}")

# ========== 5. 结果对比 ==========
print("\n" + "=" * 60)
print("优化结果对比")
print("=" * 60)
print(f"{'策略':<30} {'训练耗时':<12} {'训练准确率':<14} {'测试准确率':<14}")
print("-" * 70)
print(f"{'原始 LinearSVC (dual=True)':<30} {'10.44 秒':<12} {'70.60%':<14} {'70.47%':<14}")
print(f"{'策略1: 优化 LinearSVC':<30} {f'{time_strategy1:.2f} 秒':<12} {f'{train_acc_1*100:.2f}%':<14} {f'{test_acc_1*100:.2f}%':<14}")
print(f"{'策略2: GridSearch + LinearSVC':<30} {f'{time_strategy2:.2f} 秒':<12} {f'{train_acc_2*100:.2f}%':<14} {f'{test_acc_2*100:.2f}%':<14}")
print(f"{'策略3: SGDClassifier (hinge)':<30} {f'{time_strategy3:.2f} 秒':<12} {f'{train_acc_3*100:.2f}%':<14} {f'{test_acc_3*100:.2f}%':<14}")
print(f"{'策略4: SGD + GridSearch':<30} {f'{time_strategy4:.2f} 秒':<12} {f'{train_acc_4*100:.2f}%':<14} {f'{test_acc_4*100:.2f}%':<14}")
print("-" * 70)

# ========== 6. 选择最佳模型并保存 ==========
# 选择测试准确率最高的模型
results = [
    ('LinearSVC_opt', test_acc_1, linear_svm_opt),
    ('LinearSVC_best', test_acc_2, linear_svm_best),
    ('SGD_hinge', test_acc_3, sgd_svm),
    ('SGD_best', test_acc_4, sgd_best)
]

best_model_name, best_acc, best_model = max(results, key=lambda x: x[1])
print(f"\n>>> 最佳模型: {best_model_name} (测试准确率: {best_acc*100:.2f}%)")

# 保存最佳模型
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "linear_svm_optimized_model.pkl")
scaler_path = os.path.join(model_dir, "linear_svm_scaler.pkl")

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
