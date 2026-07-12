"""生成模型对比图（含 V6）"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

models = ['Basic MLP\n(基线)', 'Dropout', 'BatchNorm\n(最佳)', 'AE V1\n(小容量)', 'AE V2-V5\n(中容量)', 'AE V6\n(大容量)']
accuracies = [95.73, 92.70, 97.32, 67.75, 95.65, 97.15]
times = [41.0, 50.1, 54.0, 85.0, 90.0, 120.0]
params_k = [12.1, 12.1, 12.5, 2.8, 11.5, 35.2]

output_dir = os.path.join(os.path.dirname(__file__), 'results')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

# 准确率
bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([60, 100])
for bar, acc in zip(bars1, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0].axhline(y=95.73, color='#3498db', linestyle='--', alpha=0.4, label='Baseline (95.73%)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')

# 训练时间
bars2 = axes[1].bar(models, times, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('Training Time (s)', fontsize=12)
axes[1].set_title('Training Time', fontsize=14, fontweight='bold')
for bar, t in zip(bars2, times):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                f'{t:.0f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

# 参数量
bars3 = axes[2].bar(models, params_k, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
axes[2].set_ylabel('Parameters (K)', fontsize=12)
axes[2].set_title('Model Size', fontsize=14, fontweight='bold')
for bar, pk in zip(bars3, params_k):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{pk:.1f}K', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path = os.path.join(output_dir, 'comparison_summary.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'对比图已保存: {save_path}')

