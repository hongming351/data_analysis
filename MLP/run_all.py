"""
对比实验脚本：一键运行所有 4 个 MLP 模型并生成对比报告
====================================================
1. 基础标准 MLP
2. 带 Dropout 正则的 MLP
3. 带 BatchNorm 批归一化的 MLP
4. 浅层自编码器 + 分类头
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 将上级目录加入路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'MLP'))

RESULTS_DIR = os.path.join(BASE_DIR, 'MLP', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_header(text):
    print('\n' + '#'*70)
    print(f'# {text}')
    print('#'*70)


def run_all():
    results = []

    # ========== 模型 1：基础标准 MLP ==========
    print_header('运行模型 1：基础标准 MLP')
    from mlp_01_basic import main as run_basic
    t_start = time.time()
    res1 = run_basic()
    t_end = time.time()
    res1['train_time'] = t_end - t_start
    results.append(res1)

    # ========== 模型 2：Dropout MLP ==========
    print_header('运行模型 2：带 Dropout 正则的 MLP')
    from mlp_02_dropout import main as run_dropout
    t_start = time.time()
    res2 = run_dropout()
    t_end = time.time()
    res2['train_time'] = t_end - t_start
    results.append(res2)

    # ========== 模型 3：BatchNorm MLP ==========
    print_header('运行模型 3：带 BatchNorm 批归一化的 MLP')
    from mlp_03_batchnorm import main as run_batchnorm
    t_start = time.time()
    res3 = run_batchnorm()
    t_end = time.time()
    res3['train_time'] = t_end - t_start
    results.append(res3)

    # ========== 模型 4：AutoEncoder + 分类头 ==========
    print_header('运行模型 4：浅层自编码器 + 分类头')
    from mlp_04_autoencoder import main as run_ae
    t_start = time.time()
    res4 = run_ae()
    t_end = time.time()
    res4['train_time'] = t_end - t_start
    results.append(res4)

    return results


def plot_comparison(results):
    """绘制对比图"""
    names = [r['model_name'] for r in results]
    accs = [r['best_acc'] * 100 for r in results]
    times = [r['train_time'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率对比
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars1 = ax1.bar(names, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(accs) - 2, max(accs) + 3])
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.tick_params(axis='x', rotation=15)

    # 训练时间对比
    bars2 = ax2.bar(names, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Training Time (s)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'comparison_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n对比图已保存: {save_path}')


def print_summary(results):
    """打印汇总报告"""
    print('\n' + '='*70)
    print('                     📊 模型对比汇总报告')
    print('='*70)
    print(f'{"模型名称":<25} {"测试准确率":<15} {"最佳轮次":<12} {"参数量":<12} {"训练耗时":<10}')
    print('-'*70)
    for r in results:
        name = r['model_name']
        acc = f"{r['best_acc']*100:.2f}%"
        epoch = f"Epoch {r['best_epoch']}"
        params = f"{r.get('params', r.get('ae_params', 0) + r.get('clf_params', 0)):,}"
        t = f"{r['train_time']:.1f}s"
        print(f'{name:<25} {acc:<15} {epoch:<12} {params:<12} {t:<10}')
    print('-'*70)

    # 找出最佳模型
    best = max(results, key=lambda r: r['best_acc'])
    print(f'\n🏆 最佳模型: {best["model_name"]} — 测试准确率: {best["best_acc"]*100:.2f}%')

    # 生成 Markdown 报告
    md_content = """# MLP 模型对比实验报告

## 实验设置

- **数据集**: Letter Recognition (16 个特征, 26 个字母分类)
- **训练集**: 16,000 条
- **测试集**: 4,000 条
- **Batch Size**: 64
- **学习率**: 1e-3
- **优化器**: Adam
- **早停策略**: 验证集准确率 15 轮不提升则停止

## 模型对比

| 模型 | 测试准确率 | 最佳轮次 | 参数量 | 训练耗时 |
|------|-----------|---------|--------|---------|
"""
    for r in results:
        name = r['model_name']
        acc = f"{r['best_acc']*100:.2f}%"
        epoch = f"{r['best_epoch']}"
        params = f"{r.get('params', r.get('ae_params', 0) + r.get('clf_params', 0)):,}"
        t = f"{r['train_time']:.1f}s"
        md_content += f"| {name} | {acc} | {epoch} | {params} | {t} |\n"

    md_content += f"""
## 结论

🏆 **最佳模型**: {best['model_name']} — 测试准确率: {best['best_acc']*100:.2f}%

### 各模型分析

1. **基础标准 MLP**: 作为基线模型，提供基准准确率。
2. **Dropout MLP**: 通过 Dropout 正则化抑制过拟合，观察泛化能力变化。
3. **BatchNorm MLP**: 批归一化加速收敛，对比训练稳定性和最终精度。
4. **AutoEncoder + 分类头**: 先无监督降维再分类，验证特征降维的有效性。

![对比图](results/comparison_summary.png)
"""

    md_path = os.path.join(BASE_DIR, 'MLP', 'comparison_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f'\nMarkdown 报告已保存: {md_path}')


if __name__ == '__main__':
    print('='*70)
    print('          MLP 四种模型对比实验')
    print('='*70)
    print('包含模型:')
    print('  1. 基础标准 MLP')
    print('  2. 带 Dropout 正则的 MLP')
    print('  3. 带 BatchNorm 批归一化的 MLP')
    print('  4. 浅层自编码器 + 分类头')
    print('='*70)

    results = run_all()
    plot_comparison(results)
    print_summary(results)

    print('\n✅ 全部实验完成！')
