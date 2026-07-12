"""
随机森林 - Letter Recognition 分类
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import time
import json


def main():
    # 获取数据文件路径（脚本所在目录的上一级）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'letter_recognition_train.csv')
    test_path = os.path.join(base_dir, 'letter_recognition_test.csv')

    # ======================== 1. 加载数据 ========================
    print('=' * 60)
    print('随机森林 - Letter Recognition 分类')
    print('=' * 60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('letter', axis=1).values
    y_train = train_df['letter'].values
    X_test = test_df.drop('letter', axis=1).values
    y_test = test_df['letter'].values

    print(f'\n训练集: {X_train.shape}')
    print(f'测试集: {X_test.shape}')
    print(f'类别数: {len(np.unique(y_train))}')

    # ======================== 2. 训练模型 ========================
    print('\n' + '-' * 60)
    print('开始训练...')

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f'训练完成! 耗时: {train_time:.2f} 秒')

    # ======================== 3. 预测 ========================
    print('\n' + '-' * 60)
    print('预测中...')

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    print(f'预测完成! 耗时: {predict_time:.2f} 秒')

    # ======================== 4. 评估 ========================
    print('\n' + '-' * 60)
    print('模型评估')
    print('-' * 60)

    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f'\n准确率 (Accuracy):       {acc:.4f} ({acc * 100:.2f}%)')
    print(f'精确率 (Precision macro): {precision_macro:.4f}')
    print(f'召回率 (Recall macro):    {recall_macro:.4f}')
    print(f'F1 分数 (F1 macro):       {f1_macro:.4f}')

    # ======================== 5. 分类报告 ========================
    print('\n' + '-' * 60)
    print('分类报告 (按字母)')
    print('-' * 60)
    report = classification_report(y_test, y_pred)
    print(report)

    # ======================== 6. 混淆矩阵 ========================
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(np.unique(y_test))

    print('\n' + '-' * 60)
    print('混淆矩阵 (前10个字母)')
    print('-' * 60)
    # 打印前10个字母的混淆矩阵摘要
    print(' ' * 4 + ' '.join(classes[:10]))
    for i in range(10):
        row = ' '.join(f'{cm[i][j]:3d}' for j in range(10))
        print(f'{classes[i]} {row}')

    # ======================== 7. 特征重要性 ========================
    print('\n' + '-' * 60)
    print('特征重要性 (Top 10)')
    print('-' * 60)

    feature_names = [
        'x_box', 'y_box', 'width', 'high', 'onpix',
        'x_bar', 'y_bar', 'x2bar', 'y2bar', 'xybar',
        'x2ybr', 'xy2br', 'x_ege', 'xegvy', 'y_ege', 'yegvx'
    ]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(10):
        idx = indices[i]
        print(f'  {i+1:2d}. {feature_names[idx]:8s}  {importances[idx]:.4f}')

    # ======================== 8. 保存结果 ========================
    results = {
        'model': 'RandomForest',
        'n_estimators': 100,
        'accuracy': round(acc, 4),
        'precision_macro': round(precision_macro, 4),
        'recall_macro': round(recall_macro, 4),
        'f1_macro': round(f1_macro, 4),
        'train_time_sec': round(train_time, 2),
        'predict_time_sec': round(predict_time, 2),
        'random_state': 42
    }

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n结果已保存至 results.json')

    # ======================== 9. 模型保存 ========================
    import joblib
    joblib.dump(model, 'random_forest_model.pkl')
    print(f'模型已保存至 random_forest_model.pkl')

    print('\n' + '=' * 60)
    print('完成!')
    print('=' * 60)


if __name__ == '__main__':
    main()