"""
=============================================================================
网格搜索 V2 - 基于上次结果的针对性优化
=============================================================================
上一轮最佳参数: criterion=entropy, max_depth=None, max_features=sqrt,
                 min_samples_leaf=1, min_samples_split=2, n_estimators=200
CV 得分: 0.9523  |  测试集: 96.92%

本轮优化方向:
  1. n_estimators 扩大到 500 (上次最佳是 200, 越多越好)
  2. max_depth 固定 None (上次最佳)
  3. 重点对比 criterion=gini vs entropy
  4. 去掉 max_features=6,8 和 min_samples_leaf=2,3 (上次这些组合表现差)
  5. 整体组合从 1152 降到 192, 时间缩短 6 倍
=============================================================================
"""

import pandas as pd
import numpy as np
import os, time, json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print('=' * 60)
    print('网格搜索 V2 - 针对性优化')
    print('=' * 60)

    # 加载数据
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_test.csv'))
    X_train, y_train = train_df.drop('letter', axis=1), train_df['letter']
    X_test, y_test = test_df.drop('letter', axis=1), test_df['letter']

    # ========== V2 优化搜索空间 ==========
    # 基于上轮结论:
    #   - max_depth=None 最好, 固定 None
    #   - min_samples_leaf=1 最好, 固定 1
    #   - min_samples_split=2 最好, 固定 2
    #   - max_features=sqrt/log2 最好, 去掉 6 和 8
    #   - n_estimators 扩大到 500 看上限
    #   - criterion gini vs entropy 都保留
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    total = 4 * 1 * 1 * 1 * 2 * 2  # 32 种组合
    print(f'\n搜索空间: {total} 种组合 x 3折 = {total * 3} 次训练')
    print(f'相比 V1 (1152 种) 缩减 {1152//total}x')

    model = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)

    grid = GridSearchCV(
        model, param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print('\n开始搜索...')
    t0 = time.time()
    grid.fit(X_train, y_train)
    search_time = time.time() - t0

    best_model = grid.best_estimator_

    # 测试集评估
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print('\n' + '=' * 60)
    print('最优参数 & 结果')
    print('=' * 60)
    print(f'\n最佳参数:')
    for k, v in grid.best_params_.items():
        print(f'  {k}: {v}')
    print(f'\nCV 最佳得分:  {grid.best_score_:.4f}')
    print(f'测试集准确率:  {acc:.4f}')
    print(f'测试集精确率:  {precision:.4f}')
    print(f'测试集召回率:  {recall:.4f}')
    print(f'测试集 F1:     {f1:.4f}')
    print(f'搜索耗时:      {search_time:.1f}s')

    # ========== 保存结果 ==========
    cv_results = []
    for mean, std, params in zip(grid.cv_results_['mean_test_score'],
                                 grid.cv_results_['std_test_score'],
                                 grid.cv_results_['params']):
        cv_results.append({
            'mean_score': round(mean, 4),
            'std_score': round(std, 4),
            'params': {k: str(v) for k, v in params.items()}
        })

    results = {
        'version': 'V2 - targeted optimization',
        'best_params': {k: str(v) for k, v in grid.best_params_.items()},
        'best_cv_score': round(grid.best_score_, 4),
        'test_accuracy': round(acc, 4),
        'test_precision_macro': round(precision, 4),
        'test_recall_macro': round(recall, 4),
        'test_f1_macro': round(f1, 4),
        'search_time_sec': round(search_time, 2),
        'all_results': cv_results
    }

    with open(os.path.join(OUTPUT_DIR, 'results_grid_search_v2.json'), 'w') as f:
        json.dump(results, f, indent=2)
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'model_grid_search_v2.pkl'))

    print(f'\n[完成] 结果已保存')

    # ========== 与 V1 对比 ==========
    print('\n' + '-' * 60)
    print('V1 vs V2 对比')
    print('-' * 60)
    print(f'{"":<20s} {"V1 (全覆盖)":<15s} {"V2 (针对性)":<15s}')
    print(f'{"组合数":<20s} {"1,152":<15s} {"32":<15s}')
    print(f'{"耗时":<20s} {"780s":<15s} {f"{search_time:.0f}s":<15s}')
    print(f'{"CV 最佳":<20s} {"0.9523":<15s} {f"{grid.best_score_:.4f}":<15s}')
    print(f'{"测试准确率":<20s} {"96.92%":<15s} {f"{acc:.2%}":<15s}')


if __name__ == '__main__':
    main()
