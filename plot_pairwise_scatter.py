"""
两两特征散点图 (Pairwise Scatter Plot)
为了可读性，分两组绘制，每组 8 个特征 (8×8 = 36 个子图)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ===== 路径 =====
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'letter-recognition.csv')
output_dir = base_dir  # 保存到 data_analysis 目录

# ===== 设置 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ===== 加载数据 =====
print('加载数据...')
df = pd.read_csv(data_path)

# 随机抽样 800 条（兼顾速度和可读性）
df_sample = df.sample(n=800, random_state=42)

feature_cols = [
    'x_box', 'y_box', 'width', 'high', 'onpix',
    'x_bar', 'y_bar', 'x2bar', 'y2bar', 'xybar',
    'x2ybr', 'xy2br', 'x_ege', 'xegvy', 'y_ege', 'yegvx'
]

# 用 10 个代表性字母着色（否则 26 类颜色太难区分）
letters_subset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
df_plot = df_sample[df_sample['letter'].isin(letters_subset)].copy()

print(f"抽样数据: {len(df_plot)} 条, 字母类别: {sorted(df_plot['letter'].unique())}")

# ====================================================================
# 方案 1: 分组 PairGrid (8+8 特征分成两组，每组内部两两对比)
# ====================================================================

# ---- 第一组: 前 8 个特征 ----
group1 = feature_cols[:8]
print(f"\n[1/2] 绘制第一组特征两两散点图 ({len(group1)} 个特征)...")

g1 = sns.PairGrid(df_plot, vars=group1, hue='letter', palette='tab10', diag_sharey=False)
g1.map_upper(sns.scatterplot, s=10, alpha=0.6, edgecolor='none')
g1.map_lower(sns.scatterplot, s=10, alpha=0.6, edgecolor='none')
g1.map_diag(sns.kdeplot, fill=True, alpha=0.4, linewidth=1)
g1.add_legend(title='Letter', bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8, title_fontsize=9)
g1.fig.suptitle('Pairwise Scatter Plot — Group 1 (Features 1-8)', fontsize=16, fontweight='bold', y=1.02)
g1.fig.subplots_adjust(top=0.95)
g1.savefig(os.path.join(output_dir, 'pairwise_scatter_group1.png'), dpi=150)
plt.close()
print('   → pairwise_scatter_group1.png')

# ---- 第二组: 后 8 个特征 ----
group2 = feature_cols[8:]
print(f"\n[2/2] 绘制第二组特征两两散点图 ({len(group2)} 个特征)...")

g2 = sns.PairGrid(df_plot, vars=group2, hue='letter', palette='tab10', diag_sharey=False)
g2.map_upper(sns.scatterplot, s=10, alpha=0.6, edgecolor='none')
g2.map_lower(sns.scatterplot, s=10, alpha=0.6, edgecolor='none')
g2.map_diag(sns.kdeplot, fill=True, alpha=0.4, linewidth=1)
g2.add_legend(title='Letter', bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8, title_fontsize=9)
g2.fig.suptitle('Pairwise Scatter Plot — Group 2 (Features 9-16)', fontsize=16, fontweight='bold', y=1.02)
g2.fig.subplots_adjust(top=0.95)
g2.savefig(os.path.join(output_dir, 'pairwise_scatter_group2.png'), dpi=150)
plt.close()
print('   → pairwise_scatter_group2.png')

# ====================================================================
# 方案 2: 针对 Top 6 重要特征做详细两两散点图
# ====================================================================
print(f"\n[3/4] 绘制 Top 6 重要特征的两两散点图...")

important_features = ['x_box', 'y_box', 'width', 'high', 'onpix', 'x_bar']
g3 = sns.PairGrid(df_plot, vars=important_features, hue='letter', palette='tab10', diag_sharey=False)
g3.map_upper(sns.scatterplot, s=12, alpha=0.5, edgecolor='none')
g3.map_lower(sns.scatterplot, s=12, alpha=0.5, edgecolor='none')
g3.map_diag(sns.kdeplot, fill=True, alpha=0.3, linewidth=1)
g3.add_legend(title='Letter', bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8, title_fontsize=9)
g3.fig.suptitle('Pairwise Scatter Plot — Top 6 Important Features', fontsize=16, fontweight='bold', y=1.02)
g3.fig.subplots_adjust(top=0.95)
g3.savefig(os.path.join(output_dir, 'pairwise_scatter_top6.png'), dpi=150)
plt.close()
print('   → pairwise_scatter_top6.png')

# ====================================================================
# 方案 3: 相关性热力图（辅助散点图理解）
# ====================================================================
print(f"\n[4/4] 绘制特征相关性热力图...")
fig, ax = plt.subplots(figsize=(14, 12))
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, linecolor='gray',
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
            ax=ax)
ax.set_title('16 Features Correlation Matrix', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=150)
plt.close()
print('   → feature_correlation_heatmap.png')

# ====================================================================
print(f"\n{'='*60}")
print(f"[OK] 所有两两散点图已生成！")
print(f"{'='*60}")
print(f"\n生成文件列表:")
print(f"  - pairwise_scatter_group1.png       (第1-8特征, 8×8=36子图)")
print(f"  - pairwise_scatter_group2.png       (第9-16特征, 8×8=36子图)")
print(f"  - pairwise_scatter_top6.png         (Top 6 重要特征, 6×6=15子图)")
print(f"  - feature_correlation_heatmap.png   (16特征相关性热力图)")
