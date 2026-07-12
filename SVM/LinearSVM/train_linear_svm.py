"""
Linear SVM 训练 - Letter Recognition Dataset
使用 letter_recognition_train.csv 训练，letter_recognition_test.csv 测试
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import os

# ========== 1. 加载数据 ==========
print("=" * 60)
print("Linear SVM - Letter Recognition")
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
print(f"类别: {sorted(np.unique(y_train))}")

# ========== 3. 特征标准化 ==========
print("\n>>> 特征标准化 (StandardScaler) ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 训练 Linear SVM ==========
print("\n>>> 开始训练 Linear SVM ...")
start_time = time.time()

# LinearSVC 使用一对多 (OvR) 策略处理多分类
# max_iter 设大一些确保收敛
linear_svm = LinearSVC(
    C=1.0,
    max_iter=10000,
    random_state=42,
    dual=True,  # n_samples > n_features 时 dual=False 更快
    verbose=1
)

linear_svm.fit(X_train_scaled, y_train)
train_time = time.time() - start_time
print(f"训练完成! 耗时: {train_time:.2f} 秒")

# ========== 5. 预测 ==========
print("\n>>> 预测中 ...")
y_pred_train = linear_svm.predict(X_train_scaled)
y_pred_test = linear_svm.predict(X_test_scaled)

# ========== 6. 评估 ==========
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("\n" + "=" * 60)
print("评估结果")
print("=" * 60)
print(f"训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"测试集准确率:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"训练耗时:       {train_time:.2f} 秒")

print("\n>>> 分类报告 (测试集):")
print(classification_report(y_test, y_pred_test))

print("\n>>> 混淆矩阵 (测试集):")
print(confusion_matrix(y_test, y_pred_test))

# ========== 7. 保存模型 ==========
import joblib

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "linear_svm_model.pkl")
scaler_path = os.path.join(model_dir, "linear_svm_scaler.pkl")

joblib.dump(linear_svm, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n模型已保存至: {model_path}")
print(f"标准化器已保存至: {scaler_path}")
print("\n完成!")
