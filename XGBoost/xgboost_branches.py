# -*- coding: utf-8 -*-
"""
XGBoost 分支算法对比实验
比较 gbtree / gblinear / dart / XGBoost RF 四种分支
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import time
import json

# 加载数据
train_df = pd.read_csv('letter_recognition_train.csv')
test_df = pd.read_csv('letter_recognition_test.csv')

feature_cols = train_df.columns.drop('letter').tolist()
X_train = train_df[feature_cols].values
y_train_labels = train_df['letter'].values
X_test = test_df[feature_cols].values
y_test_labels = test_df['letter'].values

le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 4 个 XGBoost 分支算法 =====
branches = {
    'gbtree (树基)': {
        'booster': 'gbtree',
        'params': {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.2,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0
        }
    },
    'gblinear (线性基)': {
        'booster': 'gblinear',
        'params': {
            'n_estimators': 300, 'learning_rate': 0.2,
            'reg_lambda': 1.0, 'reg_alpha': 0.0,
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0
        }
    },
    'dart (Dropout树)': {
        'booster': 'dart',
        'params': {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.2,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'rate_drop': 0.1, 'skip_drop': 0.5,
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0
        }
    },
    'XGBoost RF 模式': {
        'booster': 'gbtree',
        'params': {
            'n_estimators': 300, 'max_depth': 6,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'num_parallel_tree': 1,  # RF-like behavior
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0
        }
    }
}

results = []
for name, config in branches.items():
    print(f"\n--- 训练 {name} ---")
    model = xgb.XGBClassifier(
        booster=config['booster'],
        **config['params'],
        eval_metric='mlogloss'
    )
    
    start = time.time()
    model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    print(f"  测试准确率: {acc:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  训练耗时: {elapsed:.2f}s")
    
    results.append({
        '分支': name,
        '测试准确率': round(acc, 4),
        'F1分数': round(f1, 4),
        '精确率': round(prec, 4),
        '召回率': round(rec, 4),
        '训练耗时(秒)': round(elapsed, 2)
    })

# 输出结果
print("\n" + "=" * 60)
print("XGBoost 分支算法对比结果")
print("=" * 60)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('测试准确率', ascending=False)
print(results_df.to_string(index=False))

# 保存结果
results_df.to_csv('XGBoost/output/xgboost_branches_results.csv', index=False)
print("\n结果已保存到 XGBoost/output/xgboost_branches_results.csv")