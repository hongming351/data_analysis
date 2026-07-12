"""
随机森林变体实验：
  1. ExtraTrees (极端随机树)
  2. Balanced RF (类别平衡)
  3. 特征筛选版 (只用 Top 8 重要特征)
  4. 更深树 + 更多树
"""

import pandas as pd
import numpy as np
import os, time, json, joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision_macro': round(precision_score(y_test, y_pred, average='macro'), 4),
        'recall_macro': round(recall_score(y_test, y_pred, average='macro'), 4),
        'f1_macro': round(f1_score(y_test, y_pred, average='macro'), 4)
    }


def main():
    print('=' * 60)
    print('随机森林变体实验')
    print('=' * 60)

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_test.csv'))

    X_train_full, y_train = train_df.drop('letter', axis=1), train_df['letter']
    X_test_full, y_test = test_df.drop('letter', axis=1), test_df['letter']

    feature_names = list(X_train_full.columns)

    # ========== 先训练基线模型获取特征重要性 ==========
    print('\n[0] 训练基线获取特征重要性...')
    base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    base.fit(X_train_full, y_train)
    importances = base.feature_importances_
    top8_idx = np.argsort(importances)[-8:]
    top8_names = [feature_names[i] for i in top8_idx]
    print(f'    Top 8 特征: {top8_names}')

    # ========== 定义模型变体 ==========
    models = {
        'ExtraTrees (极端随机树)': ExtraTreesClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        'Balanced RF (类别平衡)': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight='balanced'
        ),
        'Top8 Features (特征筛选)': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Deep RF (深树 x200)': RandomForestClassifier(
            n_estimators=200, max_depth=30, random_state=42, n_jobs=-1
        ),
    }

    all_results = {}

    for name, model in models.items():
        print(f'\n[{name}] 训练中...')

        if 'Top8' in name:
            # 只用 Top 8 特征
            X_train_8 = X_train_full[top8_names]
            X_test_8 = X_test_full[top8_names]
            t0 = time.time()
            model.fit(X_train_8, y_train)
            train_time = time.time() - t0
            metrics = evaluate(model, X_test_8, y_test)
        else:
            t0 = time.time()
            model.fit(X_train_full, y_train)
            train_time = time.time() - t0
            metrics = evaluate(model, X_test_full, y_test)

        metrics['train_time_sec'] = round(train_time, 2)
        all_results[name] = metrics
        print(f'    准确率: {metrics["accuracy"]:.4f}  |  F1: {metrics["f1_macro"]:.4f}  |  耗时: {train_time:.2f}s')

    # ========== 汇总对比 ==========
    print('\n' + '=' * 60)
    print('变体对比汇总')
    print('=' * 60)
    print(f'{"模型":<30s} {"准确率":<10s} {"F1":<10s} {"耗时(s)":<10s}')
    print('-' * 60)
    for name, metrics in all_results.items():
        print(f'{name:<30s} {metrics["accuracy"]:<10.4f} {metrics["f1_macro"]:<10.4f} {metrics["train_time_sec"]:<10.2f}')

    # 保存
    with open(os.path.join(OUTPUT_DIR, 'results_variants.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # 保存各个模型
    for name, model in models.items():
        safe_name = name.split('(')[0].strip().replace(' ', '_')
        joblib.dump(model, os.path.join(OUTPUT_DIR, f'model_{safe_name}.pkl'))

    print(f'\n✅ 变体实验完成，结果已保存')


if __name__ == '__main__':
    main()
