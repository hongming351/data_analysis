"""
随机森林 - 可视化分析
生成各类图表辅助分析模型表现与数据特征
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def main():
    # ======================== 路径设置 ========================
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'letter_recognition_train.csv')
    test_path = os.path.join(base_dir, 'letter_recognition_test.csv')
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # ======================== 加载数据与模型 ========================
    print('加载数据与模型...')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)

    X_train = train_df.drop('letter', axis=1)
    y_train = train_df['letter']
    X_test = test_df.drop('letter', axis=1)
    y_test = test_df['letter']

    y_pred = model.predict(X_test)

    feature_names = list(X_train.columns)

    # ====================================================================
    # 图1: 字母类别分布 (训练集 vs 测试集)
    # ====================================================================
    print('[1/8] 绘制字母类别分布对比图...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df_, title in zip(axes, [train_df, test_df], ['训练集 (16,000 条)', '测试集 (4,000 条)']):
        counts = df_['letter'].value_counts().sort_index()
        colors = plt.cm.tab20(np.linspace(0, 1, 26))
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('字母', fontsize=11)
        ax.set_ylabel('样本数量', fontsize=11)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_class_distribution.png'))
    plt.close()
    print('   → 01_class_distribution.png')

    # ====================================================================
    # 图2: 每个字母的准确率 (混淆矩阵对角线的正确率)
    # ====================================================================
    print('[2/8] 绘制每类字母准确率...')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(y_test.unique())
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#2ecc71' if v >= 0.95 else '#f39c12' if v >= 0.85 else '#e74c3c' for v in per_class_acc]
    bars = ax.bar(classes, per_class_acc, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(y=0.9675, color='red', linestyle='--', linewidth=1.5, label=f'整体准确率: 96.75%')
    ax.set_title('每类字母识别准确率', fontsize=14, fontweight='bold')
    ax.set_xlabel('字母', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.2%}', ha='center', va='bottom', fontsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_per_class_accuracy.png'))
    plt.close()
    print('   → 02_per_class_accuracy.png')

    # ====================================================================
    # 图3: 混淆矩阵热力图
    # ====================================================================
    print('[3/8] 绘制混淆矩阵热力图...')
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    mask = np.zeros_like(cm_norm)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='gray', ax=ax,
                vmin=0, vmax=1, mask=mask, cbar_kws={'label': '比例'})
    ax.set_title('混淆矩阵 (归一化, 仅下三角)', fontsize=14, fontweight='bold')
    ax.set_xlabel('预测值', fontsize=12)
    ax.set_ylabel('真实值', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_confusion_matrix.png'))
    plt.close()
    print('   → 03_confusion_matrix.png')

    # ====================================================================
    # 图4: 特征重要性排名
    # ====================================================================
    print('[4/8] 绘制特征重要性排名...')
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_names)))
    bars = ax.barh(range(len(feature_names)), importances[indices], color=colors_bar, edgecolor='white')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('重要性得分', fontsize=12)
    ax.set_title('随机森林特征重要性排名', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.2%}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_feature_importance.png'))
    plt.close()
    print('   → 04_feature_importance.png')

    # ====================================================================
    # 图5: 累积特征重要性
    # ====================================================================
    print('[5/8] 绘制累积特征重要性...')
    sorted_idx = np.argsort(importances)[::-1]
    sorted_imp = importances[sorted_idx]
    cumsum = np.cumsum(sorted_imp)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(range(len(sorted_imp)), sorted_imp, color='#3498db', edgecolor='white', label='单个特征')
    ax1.set_xlabel('特征 (按重要性降序)', fontsize=11)
    ax1.set_ylabel('重要性得分', fontsize=11, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xticks(range(len(sorted_imp)))
    ax1.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha='right', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(range(len(sorted_imp)), cumsum, 'o-', color='#e74c3c', linewidth=2, markersize=5, label='累积')
    ax2.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('累积重要性', fontsize=11, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    # 标记达到90%的位置
    idx_90 = np.argmax(cumsum >= 0.9) + 1
    ax2.annotate(f'前 {idx_90} 个特征达 90%',
                xy=(idx_90 - 1, cumsum[idx_90 - 1]),
                xytext=(idx_90 - 1, cumsum[idx_90 - 1] - 0.15),
                fontsize=10, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    plt.title('累积特征重要性分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_cumulative_importance.png'))
    plt.close()
    print('   → 05_cumulative_importance.png')

    # ====================================================================
    # 图6: 各类字母 Precision / Recall / F1 对比
    # ====================================================================
    print('[6/8] 绘制 Precision / Recall / F1 对比图...')
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, p, width, label='Precision', color='#3498db', edgecolor='white')
    ax.bar(x, r, width, label='Recall', color='#2ecc71', edgecolor='white')
    ax.bar(x + width, f, width, label='F1-score', color='#e74c3c', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_xlabel('字母', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('各类字母 Precision / Recall / F1 对比', fontsize=14, fontweight='bold')
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=10)
    ax.axhline(y=0.9675, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_prf_comparison.png'))
    plt.close()
    print('   → 06_prf_comparison.png')

    # ====================================================================
    # 图7: 特征分布箱线图 (按前4个最重要的特征)
    # ====================================================================
    print('[7/8] 绘制特征分布箱线图...')
    top4_idx = indices[-4:]
    top4_names = [feature_names[i] for i in top4_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, feat_name in zip(axes, top4_names):
        df_plot = pd.DataFrame({'letter': y_test, 'value': X_test[feat_name]})
        sns.boxplot(x='letter', y='value', data=df_plot, ax=ax,
                    palette='tab20', linewidth=0.8, fliersize=2)
        ax.set_title(f'{feat_name} 在各字母中的分布', fontsize=11, fontweight='bold')
        ax.set_xlabel('字母', fontsize=10)
        ax.set_ylabel('值', fontsize=10)

    plt.suptitle('Top 4 重要特征在各字母中的分布', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_feature_distribution_box.png'))
    plt.close()
    print('   → 07_feature_distribution_box.png')

    # ====================================================================
    # 图8: 模型性能雷达图
    # ====================================================================
    print('[8/8] 绘制模型性能雷达图...')
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    # 计算每个指标
    precision_macro = np.mean(p)
    recall_macro = np.mean(r)
    f1_macro = np.mean(f)

    categories = ['准确率\nAccuracy', '精确率\nPrecision', '召回率\nRecall', 'F1分数']
    values = [acc, precision_macro, recall_macro, f1_macro]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.plot(angles, values, 'o-', color='#3498db', linewidth=2, markersize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0.9, 1.0)
    ax.set_title('随机森林模型性能雷达图', fontsize=14, fontweight='bold', pad=20)

    for angle, val, label in zip(angles[:-1], values[:-1], categories):
        ax.text(angle, val + 0.01, f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_radar_chart.png'))
    plt.close()
    print('   → 08_radar_chart.png')

    # ====================================================================
    # 完成
    # ====================================================================
    print('\n' + '=' * 60)
    print(f'✅ 全部可视化图片已保存至: {output_dir}')
    print('=' * 60)
    print('\n生成文件列表:')
    for f in sorted(os.listdir(output_dir)):
        print(f'  - {f}')


if __name__ == '__main__':
    main()
