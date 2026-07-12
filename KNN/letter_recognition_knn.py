# -*- coding: utf-8 -*-

"""
字母识别KNN模型
基于UCI Letter Recognition数据集，使用K近邻算法进行字母分类
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import time
import os
import json

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 创建输出目录 ====================
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("阶段1: 数据加载")
print("=" * 60)

# 从预分割的文件加载训练集和测试集
train_df = pd.read_csv('letter_recognition_train.csv')
test_df = pd.read_csv('letter_recognition_test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print(f"总样本数: {len(train_df) + len(test_df)}")
print(f"特征维度: {train_df.shape[1] - 1} 个有效特征 + 1 个目标变量")
print(f"字母类别: {sorted(train_df['letter'].unique())}")

# 特征列名
feature_cols = train_df.columns.drop('letter').tolist()
print(f"特征名称: {feature_cols}")

# 缺失值检查
print(f"\n训练集缺失值总数: {train_df.isnull().sum().sum()}")
print(f"测试集缺失值总数: {test_df.isnull().sum().sum()}")

# 目标变量分布
print("\n训练集各类别样本数:")
class_counts_train = train_df['letter'].value_counts().sort_index()
print(class_counts_train)

print("\n测试集各类别样本数:")
class_counts_test = test_df['letter'].value_counts().sort_index()
print(class_counts_test)

# ==================== 2. 数据预处理 ====================
print("\n" + "=" * 60)
print("阶段2: 数据预处理")
print("=" * 60)

# 分别从训练集和测试集提取特征和标签
X_train = train_df[feature_cols].values
y_train_labels = train_df['letter'].values
X_test = test_df[feature_cols].values
y_test_labels = test_df['letter'].values

# 基于训练集编码目标变量
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)

print(f"训练特征矩阵形状: {X_train.shape}")
print(f"测试特征矩阵形状: {X_test.shape}")
print(f"编码后标签类别数: {len(le.classes_)}")
print(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

# KNN对特征尺度敏感，必须进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("已完成标准化 (StandardScaler)")

# ==================== 3. KNN模型训练与调优 ====================
print("\n" + "=" * 60)
print("阶段3: KNN模型训练与超参数调优")
print("=" * 60)

# 定义候选k值（考虑到数据集大小和类别数26）
k_values = list(range(1, 31, 2))  # 1, 3, 5, ..., 29
weight_options = ['uniform', 'distance']
p_values = [1, 2]  # 1:曼哈顿距离, 2:欧氏距离

print(f"候选K值: {k_values}")
print(f"权重方式: {weight_options}")
print(f"距离度量: p={p_values} (1:曼哈顿, 2:欧氏)")

start_time = time.time()

# 网格搜索
param_grid = {
    'n_neighbors': k_values,
    'weights': weight_options,
    'p': p_values
}

knn_base = KNeighborsClassifier(n_jobs=-1)
gs = GridSearchCV(
    knn_base, param_grid, 
    cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
gs.fit(X_train_scaled, y_train)

best_knn = gs.best_estimator_
best_params = gs.best_params_
train_time = time.time() - start_time

print(f"\n最佳参数: {best_params}")
print(f"训练时间: {train_time:.2f}秒")

# ==================== 4. 模型评估 ====================
print("\n" + "=" * 60)
print("阶段4: 模型评估")
print("=" * 60)

# 训练集预测
y_pred_train = best_knn.predict(X_train_scaled)
# 测试集预测
y_pred_test = best_knn.predict(X_test_scaled)

# 计算各项指标
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

# 保存评估结果
results = {
    'model': 'KNN',
    'best_params': best_params,
    'train_accuracy': round(train_acc, 4),
    'test_accuracy': round(test_acc, 4),
    'precision_weighted': round(precision, 4),
    'recall_weighted': round(recall, 4),
    'f1_score_weighted': round(f1, 4),
    'train_time_seconds': round(train_time, 2)
}

with open(os.path.join(output_dir, 'knn_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ==================== 5. 分类报告 ====================
print("\n--- 分类报告 ---")
class_names = le.classes_
report = classification_report(y_test, y_pred_test, target_names=class_names, digits=4)
print(report)

with open(os.path.join(output_dir, 'knn_classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write("KNN Classification Report\n")
    f.write("=" * 60 + "\n")
    f.write(f"最佳参数: {best_params}\n\n")
    f.write(report)

# ==================== 6. 混淆矩阵可视化 ====================
print("\n--- 生成混淆矩阵 ---")
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, y_pred_test)
# 计算每个类别的准确率（对角线百分比）
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 绘制带数字的热图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('KNN 混淆矩阵', fontsize=16)
plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'knn_confusion_matrix.png'), dpi=150)
plt.close()
print("  [OK] 混淆矩阵已保存")

# ==================== 7. 不同K值性能对比 ====================
print("\n--- 不同K值性能对比分析 ---")

cv_results = pd.DataFrame(gs.cv_results_)
# 按权重和距离度量分组，绘制不同K值下的性能曲线
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (ax, p_val) in enumerate(zip(axes, [1, 2])):
    dist_name = "曼哈顿距离(L1)" if p_val == 1 else "欧氏距离(L2)"
    
    for weight in weight_options:
        subset = cv_results[(cv_results['param_p'] == p_val) & 
                            (cv_results['param_weights'] == weight)]
        k_vals = subset['param_n_neighbors'].values
        scores = subset['mean_test_score'].values
        ax.plot(k_vals, scores, marker='o', label=f'weights={weight}')
    
    ax.set_xlabel('K值 (近邻数量)', fontsize=12)
    ax.set_ylabel('平均交叉验证准确率', fontsize=12)
    ax.set_title(f'K值 vs 准确率 ({dist_name})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'knn_k_value_comparison.png'), dpi=150)
plt.close()
print("  [OK] K值对比图已保存")

# ==================== 8. 交叉验证 ====================
print("\n" + "=" * 60)
print("阶段5: 交叉验证评估")
print("=" * 60)

cv_scores = cross_val_score(best_knn, X_train_scaled, y_train, 
                            cv=5, scoring='accuracy')
print(f"5折交叉验证准确率:")
print(f"  各折: {cv_scores.round(4)}")
print(f"  均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==================== 9. 错误分析 ====================
print("\n" + "=" * 60)
print("阶段6: 错误分析")
print("=" * 60)

# 找出预测错误的样本
errors = y_test != y_pred_test
n_errors = errors.sum()
total = len(y_test)
print(f"总测试样本: {total}")
print(f"错误预测数: {n_errors}")
print(f"错误率: {n_errors/total:.4f}")

# 按字母类别统计错误率
error_by_class = {}
for i, cls in enumerate(class_names):
    mask = (y_test == i)
    class_total = mask.sum()
    class_errors = errors[mask].sum()
    error_rate = class_errors / class_total if class_total > 0 else 0
    error_by_class[cls] = {
        'total': int(class_total), 
        'errors': int(class_errors), 
        'error_rate': round(error_rate, 4)
    }

error_df = pd.DataFrame(error_by_class).T
error_df = error_df.sort_values('error_rate', ascending=False)
print("\n各类别错误率（降序）:")
print(error_df.to_string())

# 保存错误分析
error_df.to_csv(os.path.join(output_dir, 'error_analysis.csv'), float_format='%.4f')

# 绘制各类别错误率柱状图
plt.figure(figsize=(14, 6))
plt.bar(error_df.index, error_df['error_rate'], color='coral', edgecolor='black')
plt.xlabel('字母类别', fontsize=12)
plt.ylabel('错误率', fontsize=12)
plt.title('KNN 各类别预测错误率', fontsize=14)
plt.xticks(rotation=0)
for i, (idx, row) in enumerate(error_df.iterrows()):
    plt.text(i, row['error_rate'] + 0.005, f"{row['error_rate']:.2%}", 
             ha='center', va='bottom', fontsize=8)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'knn_error_rate_by_class.png'), dpi=150)
plt.close()
print("  [OK] 错误率分析图已保存")

# 混淆矩阵归一化（按行），显示各类别召回率
plt.figure(figsize=(14, 12))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names)
plt.title('KNN 归一化混淆矩阵 (按行归一化 = 召回率)', fontsize=16)
plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'knn_normalized_confusion_matrix.png'), dpi=150)
plt.close()
print("  [OK] 归一化混淆矩阵已保存")

# ==================== 10. 结果总结 ====================
print("\n" + "=" * 60)
print("阶段7: 结果总结")
print("=" * 60)

print(f"""
=== KNN字母识别模型最终结果 ===

1. 数据集信息:
   - 训练集样本数: {len(train_df)}
   - 测试集样本数: {len(test_df)}
   - 总样本数: {len(train_df) + len(test_df)}
   - 特征数: {X_train.shape[1]}
   - 类别数: {len(le.classes_)}
   - 类别: {', '.join(le.classes_)}

2. 数据预处理:
   - 标准化方法: StandardScaler (Z-score归一化)
   - 原因: KNN基于距离度量，需要消除量纲影响

3. KNN最佳参数:
   - K值 (近邻数量): {best_params['n_neighbors']}
   - 权重方式: {best_params['weights']}
   - 距离度量: {'曼哈顿距离(L1)' if best_params['p'] == 1 else '欧氏距离(L2)'}

4. 模型性能:
   - 训练准确率: {train_acc:.2%}
   - 测试准确率: {test_acc:.2%}
   - 精确率(weighted): {precision:.4f}
   - 召回率(weighted): {recall:.4f}
   - F1分数(weighted): {f1:.4f}
   - 5折交叉验证均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

5. KNN算法特点:
   - 惰性学习: 不需要显式训练过程，预测时计算复杂度高
   - 非参数模型: 不对数据分布做假设
   - 距离度量敏感: 特征标准化对性能至关重要
   - 适用场景: 适合低维小规模分类问题
""")

# 保存总结
with open(os.path.join(output_dir, 'knn_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("KNN字母识别模型分析报告\n")
    f.write("=" * 60 + "\n")
    f.write(f"数据集: UCI Letter Recognition\n")
    f.write(f"训练集样本数: {len(train_df)}\n")
    f.write(f"测试集样本数: {len(test_df)}\n")
    f.write(f"总样本数: {len(train_df) + len(test_df)}\n")
    f.write(f"特征数: {X_train.shape[1]}\n")
    f.write(f"类别数: {len(le.classes_)}\n")
    f.write(f"类别: {', '.join(le.classes_)}\n\n")
    f.write(f"最佳参数: {best_params}\n\n")
    f.write(f"测试准确率: {test_acc:.4f}\n")
    f.write(f"精确率(weighted): {precision:.4f}\n")
    f.write(f"召回率(weighted): {recall:.4f}\n")
    f.write(f"F1分数(weighted): {f1:.4f}\n\n")
    f.write("KNN算法特点:\n")
    f.write("  1. 惰性学习: 不需要显式训练过程，预测时计算复杂度高\n")
    f.write("  2. 非参数模型: 不对数据分布做假设\n")
    f.write("  3. 距离度量敏感: 特征标准化对性能至关重要\n")
    f.write("  4. 适用场景: 适合低维小规模分类问题\n")

print("\n所有分析完成! 结果已保存到 'KNN/output' 目录")
print("=" * 60)