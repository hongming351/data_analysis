# -*- coding: utf-8 -*-
"""
字母识别机器学习预测与分析项目
基于UCI Letter Recognition数据集，使用XGBoost进行字母分类
训练集和测试集均进行标准化处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import xgboost as xgb
import time
import os

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 创建输出目录 ====================
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("阶段1: 数据加载与探索性分析(EDA)")
print("=" * 60)

# 从预分割的文件加载训练集和测试集
train_df = pd.read_csv('letter_recognition_train.csv')
test_df = pd.read_csv('letter_recognition_test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print(f"总样本数: {len(train_df) + len(test_df)}")
print(f"特征维度: {train_df.shape[1] - 1} 个有效特征 + 1 个目标变量")
print(f"字母类别: {sorted(train_df['letter'].unique())}")

# ==================== 2. 基础统计分析 ====================
print("\n--- 基础统计 (训练集) ---")
desc = train_df.describe()
print(desc.round(2))
desc.to_csv(os.path.join(output_dir, 'basic_statistics.csv'), float_format='%.2f')

print(f"\n训练集缺失值总数: {train_df.isnull().sum().sum()}")
print(f"测试集缺失值总数: {test_df.isnull().sum().sum()}")

print("\n训练集各类别样本数:")
class_counts_train = train_df['letter'].value_counts().sort_index()
print(class_counts_train)
print(f"训练集不平衡比率: {class_counts_train.max() / class_counts_train.min():.2f}")

print("\n测试集各类别样本数:")
class_counts_test = test_df['letter'].value_counts().sort_index()
print(class_counts_test)
print(f"测试集不平衡比率: {class_counts_test.max() / class_counts_test.min():.2f}")

# ==================== 3. 可视化 ====================
print("\n--- 生成可视化图表 ---")

feature_cols = train_df.columns.drop('letter').tolist()
full_df = pd.concat([train_df, test_df], axis=0)

# 3.1 类别分布直方图
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, data, title in zip(axes, [train_df, test_df], ['训练集', '测试集']):
    counts = data['letter'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='black')
    ax.set_title(f'{title} 字母类别分布', fontsize=14)
    ax.set_xlabel('字母')
    ax.set_ylabel('样本数量')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
plt.close()
print("  [OK] 类别分布直方图")

# 3.2 特征分布直方图
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(train_df[col], bins=15, edgecolor='black', alpha=0.7, label='训练集')
    axes[i].hist(test_df[col], bins=15, edgecolor='black', alpha=0.3, color='red', label='测试集')
    axes[i].set_title(f'{col}', fontsize=10)
    axes[i].set_xlabel('值')
    axes[i].set_ylabel('频数')
    axes[i].legend(fontsize=6)
plt.suptitle('各特征分布直方图(蓝:训练集, 红:测试集)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_histograms.png'), dpi=150)
plt.close()
print("  [OK] 特征分布直方图")

# 3.3 相关性热力图
plt.figure(figsize=(12, 10))
corr_matrix = full_df[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('特征间相关性热力图(全数据)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("  [OK] 相关性热力图")

# 3.4 箱线图
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].boxplot(train_df[col])
    axes[i].set_title(f'{col}', fontsize=10)
    axes[i].set_ylabel('值')
plt.suptitle('各特征箱线图(训练集)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplots.png'), dpi=150)
plt.close()
print("  [OK] 箱线图")

# 3.5 类别-特征均值热图
plt.figure(figsize=(14, 10))
class_feature_mean = full_df.groupby('letter')[feature_cols].mean()
sns.heatmap(class_feature_mean, annot=True, fmt='.1f', cmap='YlOrRd', 
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('各类别各特征均值热图(全数据)', fontsize=14)
plt.xlabel('特征')
plt.ylabel('字母类别')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_feature_heatmap.png'), dpi=150)
plt.close()
print("  [OK] 类别-特征均值热图")

# ==================== 4. 数据预处理 ====================
print("\n" + "=" * 60)
print("阶段2: 数据预处理与特征工程")
print("=" * 60)

X_train = train_df[feature_cols].values
y_train_labels = train_df['letter'].values
X_test = test_df[feature_cols].values
y_test_labels = test_df['letter'].values

le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)

print(f"训练特征矩阵形状: {X_train.shape}")
print(f"测试特征矩阵形状: {X_test.shape}")
print(f"编码后标签类别数: {len(le.classes_)}")
print(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"特征名称: {feature_cols}")
print(f"\n训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ==================== 标准化处理 ====================
print("\n--- 标准化处理 (StandardScaler) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"标准化前 - 训练集各特征均值范围: [{X_train.mean(axis=0).min():.2f}, {X_train.mean(axis=0).max():.2f}]")
print(f"标准化前 - 训练集各特征标准差范围: [{X_train.std(axis=0).min():.2f}, {X_train.std(axis=0).max():.2f}]")
print(f"标准化后 - 训练集各特征均值范围: [{X_train_scaled.mean(axis=0).min():.2f}, {X_train_scaled.mean(axis=0).max():.2f}]")
print(f"标准化后 - 训练集各特征标准差范围: [{X_train_scaled.std(axis=0).min():.2f}, {X_train_scaled.std(axis=0).max():.2f}]")

# ==================== 5. XGBoost 模型训练 ====================
print("\n" + "=" * 60)
print("阶段3: XGBoost 模型训练与调优 (使用标准化数据)")
print("=" * 60)

xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False)

print("执行网格搜索超参数调优...")
start_time = time.time()

gs = GridSearchCV(
    xgb_model, xgb_params,
    cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
gs.fit(X_train_scaled, y_train)

best_model = gs.best_estimator_
best_params = gs.best_params_
train_time = time.time() - start_time

print(f"\n最佳参数: {best_params}")
print(f"训练时间: {train_time:.2f}秒")

y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n训练准确率: {train_acc:.4f}")
print(f"测试准确率: {test_acc:.4f}")
print(f"精确率(weighted): {precision:.4f}")
print(f"召回率(weighted): {recall:.4f}")
print(f"F1分数(weighted): {f1:.4f}")

# ==================== 6. 混淆矩阵 ====================
print("\n--- 生成混淆矩阵 ---")
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('XGBoost 混淆矩阵 (标准化后)', fontsize=14)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
plt.close()
print("  [OK] 混淆矩阵")

# ==================== 7. 分类报告 ====================
print("\n--- 分类报告 ---")
report = classification_report(y_test, y_pred_test, target_names=le.classes_, digits=4)
print(report)

with open(os.path.join(output_dir, 'XGBoost_classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write("XGBoost Classification Report\n")
    f.write("=" * 60 + "\n")
    f.write(report)

# ==================== 8. 特征重要性 ====================
print("\n--- XGBoost特征重要性 ---")
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feat_imp_df.to_string(index=False))
feat_imp_df.to_csv(os.path.join(output_dir, 'xgboost_feature_importance.csv'), index=False)

plt.figure(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_imp_df)))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=colors)
plt.xlabel('重要性')
plt.title('XGBoost 特征重要性 (标准化后)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'xgboost_feature_importance.png'), dpi=150)
plt.close()
print("  [OK] 特征重要性图")

# ==================== 9. 交叉验证 ====================
print("\n" + "=" * 60)
print("阶段4: 交叉验证评估")
print("=" * 60)

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nXGBoost 5折交叉验证准确率:")
print(f"  各折: {cv_scores.round(4)}")
print(f"  均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==================== 10. 结果总结 ====================
print("\n" + "=" * 60)
print("阶段5: 结果总结")
print("=" * 60)

print(f"\n=== XGBoost + 标准化 最终结论 ===")
summary = f"""
1. 数据集信息:
   - 来源: UCI Letter Recognition数据集
   - 训练集样本数: {len(train_df)}
   - 测试集样本数: {len(test_df)}
   - 总样本数: {len(train_df) + len(test_df)}
   - 特征数: {X_train.shape[1]}
   - 类别数: {len(le.classes_)}
   - 类别: {', '.join(le.classes_)}

2. 数据预处理:
   - 无缺失值，类别分布相对均衡
   - 使用 StandardScaler 标准化 (0均值, 1方差)
   - 仅基于训练集拟合 scaler，再转换测试集 (防止数据泄露)

3. XGBoost 模型性能:
   - 测试集准确率: {test_acc:.2%}
   - 精确率(weighted): {precision:.4f}
   - 召回率(weighted): {recall:.4f}
   - F1分数(weighted): {f1:.4f}
   - 5折交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

4. 最佳超参数: {best_params}

5. 特征重要性 Top 5:
"""
print(summary)

top5 = feat_imp_df.head(5)
for i, row in top5.iterrows():
    print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")

print(f"""
6. 分析:
   - XGBoost（梯度提升树）在标准化后的数据上表现稳定
   - 标准化消除了不同特征的量纲差异，有助于模型收敛
   - 16个视觉特征被有效利用，自动进行特征选择
""")

# 保存完整报告
with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("字母识别机器学习分析报告 (XGBoost + 标准化)\n")
    f.write("=" * 60 + "\n")
    f.write(f"数据集: UCI Letter Recognition\n")
    f.write(f"训练集样本数: {len(train_df)}\n")
    f.write(f"测试集样本数: {len(test_df)}\n")
    f.write(f"总样本数: {len(train_df) + len(test_df)}\n")
    f.write(f"特征数: {X_train.shape[1]}\n")
    f.write(f"类别数: {len(le.classes_)}\n")
    f.write(f"类别: {', '.join(le.classes_)}\n")
    f.write(f"标准化: StandardScaler (fit on training set, transform test set)\n\n")
    f.write("XGBoost 模型性能:\n")
    f.write(f"  最佳参数: {best_params}\n")
    f.write(f"  训练准确率: {train_acc:.4f}\n")
    f.write(f"  测试准确率: {test_acc:.4f}\n")
    f.write(f"  精确率(weighted): {precision:.4f}\n")
    f.write(f"  召回率(weighted): {recall:.4f}\n")
    f.write(f"  F1分数(weighted): {f1:.4f}\n")
    f.write(f"  训练时间: {train_time:.2f}秒\n")
    f.write(f"  5折交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write("特征重要性:\n")
    f.write(feat_imp_df.to_string(index=False))
    f.write("\n\n重点分析：XGBoost模型\n")
    f.write("=" * 60 + "\n")
    f.write("XGBoost（eXtreme Gradient Boosting）是一种基于梯度提升树的集成学习算法。\n")
    f.write("优势：\n")
    f.write("  1. 通过集成多个弱学习器（决策树）来提升模型性能\n")
    f.write("  2. 内置正则化，有效防止过拟合\n")
    f.write("  3. 支持并行计算，训练速度快\n")
    f.write("  4. 能够处理高维特征，自动进行特征选择\n")
    f.write("  5. 对异常值不敏感，鲁棒性强\n")
    f.write("\n在字母识别任务中的应用：\n")
    f.write("  - 16个视觉特征能够被XGBoost有效利用\n")
    f.write("  - 26分类任务中表现优异\n")
    f.write("  - 标准化后模型处理更稳定\n")
    f.write("  - 特征重要性分析揭示了字母识别的关键视觉特征\n")

print("\n所有分析完成! 结果已保存到 'output' 目录")
print("=" * 60)
