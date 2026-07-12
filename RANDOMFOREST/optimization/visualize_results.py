"""
优化结果可视化 - 对比基线 / GridSearch V2 / 变体
"""

import pandas as pd
import numpy as np
import os, json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print('绘制优化结果可视化...')

    # 加载各实验结果
    baseline = load_json('results_baseline.json')
    grid_v2 = load_json('results_grid_search_v2.json')
    variants = load_json('results_variants.json')

    # ====================================================================
    # 图1: 基线 vs GridSearch V2 最优模型 指标对比
    # ====================================================================
    print('[1/5] 基线 vs GridSearch V2 对比...')
    if baseline and grid_v2:
        labels = ['准确率', '精确率', '召回率', 'F1']
        baseline_vals = [
            baseline['accuracy'], baseline['precision_macro'],
            baseline['recall_macro'], baseline['f1_macro']
        ]
        grid_vals = [
            grid_v2['test_accuracy'], grid_v2['test_precision_macro'],
            grid_v2['test_recall_macro'], grid_v2['test_f1_macro']
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        w = 0.3
        bars1 = ax.bar(x - w / 2, baseline_vals, w, label='Baseline', color='#3498db', edgecolor='white')
        bars2 = ax.bar(x + w / 2, grid_vals, w, label='GridSearch V2 最优', color='#e74c3c', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0.94, 0.98)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title('基线 vs GridSearch V2 最优模型', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '01_baseline_vs_grid_v2.png'))
        plt.close()
        print('   → 01_baseline_vs_grid_v2.png')

    # ====================================================================
    # 图2: GridSearch V2 参数得分 (n_estimators vs criterion × max_features)
    # ====================================================================
    print('[2/5] 搜索参数热力图...')
    if grid_v2 and 'all_results' in grid_v2:
        rows = []
        for r in grid_v2['all_results']:
            row = r['params'].copy()
            row['score'] = r['mean_score']
            rows.append(row)
        df_cv = pd.DataFrame(rows)

        # n_estimators vs criterion 热力图
        if 'n_estimators' in df_cv.columns and 'criterion' in df_cv.columns:
            pivot = df_cv.pivot_table(
                values='score', index='criterion', columns='n_estimators', aggfunc='mean'
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax,
                        linewidths=1, linecolor='white')
            ax.set_title('不同 n_estimators × criterion 的 CV 准确率', fontsize=13, fontweight='bold')
            ax.set_xlabel('n_estimators', fontsize=11)
            ax.set_ylabel('criterion', fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, '02_param_heatmap.png'))
            plt.close()
            print('   → 02_param_heatmap.png')

    # ====================================================================
    # 图3: 变体模型对比柱状图
    # ====================================================================
    print('[3/5] 变体模型对比...')
    if variants:
        names = list(variants.keys())
        accs = [variants[n]['accuracy'] for n in names]
        f1s = [variants[n]['f1_macro'] for n in names]
        times = [variants[n]['train_time_sec'] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 准确率对比
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = axes[0].barh(names, accs, color=colors, edgecolor='white', height=0.6)
        axes[0].set_title('各变体准确率对比', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('准确率', fontsize=11)
        axes[0].set_xlim(0.93, 0.98)
        for bar, val in zip(bars, accs):
            axes[0].text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', ha='left', va='center', fontsize=9)

        # 训练时间对比
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(names)))
        bars2 = axes[1].barh(names, times, color=colors2, edgecolor='white', height=0.6)
        axes[1].set_title('各变体训练耗时对比', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('训练时间 (秒)', fontsize=11)
        for bar, val in zip(bars2, times):
            axes[1].text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                         f'{val:.2f}s', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '03_variants_comparison.png'))
        plt.close()
        print('   → 03_variants_comparison.png')

    # ====================================================================
    # 图4: 所有模型综合排名 (含基线 + GridSearch V2 + 变体)
    # ====================================================================
    print('[4/5] 综合排名...')
    all_models = {}

    if baseline:
        all_models['Baseline'] = {
            'accuracy': baseline['accuracy'],
            'f1': baseline['f1_macro'],
            'time': baseline['train_time_sec']
        }
    if grid_v2:
        all_models['GridSearch V2'] = {
            'accuracy': grid_v2['test_accuracy'],
            'f1': grid_v2['test_f1_macro'],
            'time': grid_v2['search_time_sec']
        }
    if variants:
        for name, metrics in variants.items():
            short_name = name.split('(')[0].strip()
            all_models[short_name] = {
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1_macro'],
                'time': metrics['train_time_sec']
            }

    if all_models:
        df_all = pd.DataFrame(all_models).T.sort_values('f1', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_all))
        w = 0.3
        ax.barh(x + w / 2, df_all['accuracy'], w, label='准确率', color='#3498db', edgecolor='white')
        ax.barh(x - w / 2, df_all['f1'], w, label='F1 分数', color='#2ecc71', edgecolor='white')

        ax.set_yticks(x)
        ax.set_yticklabels(df_all.index, fontsize=10)
        ax.set_xlabel('分数', fontsize=12)
        ax.set_title('所有模型综合排名 (按 F1 升序)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(0.93, 0.98)

        for i, (acc, f1) in enumerate(zip(df_all['accuracy'], df_all['f1'])):
            ax.text(acc + 0.001, i + w / 2, f'{acc:.4f}', va='center', fontsize=8)
            ax.text(f1 + 0.001, i - w / 2, f'{f1:.4f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '04_overall_ranking.png'))
        plt.close()
        print('   → 04_overall_ranking.png')

    # ====================================================================
    # 图5: GridSearch V2 Top 10 参数组合
    # ====================================================================
    print('[5/5] Top 10 参数组合...')
    if grid_v2 and 'all_results' in grid_v2:
        sorted_results = sorted(grid_v2['all_results'], key=lambda x: x['mean_score'], reverse=True)[:10]

        names = []
        scores = []
        stds = []
        for r in sorted_results:
            p = r['params']
            label = f"n={p.get('n_estimators','?')} f={p.get('max_features','?')} cr={p.get('criterion','?')}"
            names.append(label)
            scores.append(r['mean_score'])
            stds.append(r['std_score'])

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(names)), scores, yerr=stds, capsize=5,
                      color=plt.cm.Blues(np.linspace(0.4, 0.9, len(names))),
                      edgecolor='white')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('CV 平均准确率', fontsize=12)
        ax.set_title('GridSearch V2 Top 10 参数组合 (含标准差)', fontsize=14, fontweight='bold')
        ax.set_ylim(0.95, 0.965)

        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '05_top10_params.png'))
        plt.close()
        print('   → 05_top10_params.png')

    print(f'\n[完成] 所有图片已保存至: {FIG_DIR}')
    for f in sorted(os.listdir(FIG_DIR)):
        print(f'  - {f}')


if __name__ == '__main__':
    main()
