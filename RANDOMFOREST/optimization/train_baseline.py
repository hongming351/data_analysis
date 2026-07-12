"""
基线模型 - 默认随机森林 (用于对比)
"""

import pandas as pd
import numpy as np
import os, time, json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print('=' * 60)
    print('基线模型: 默认随机森林 (n_estimators=100)')
    print('=' * 60)

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_test.csv'))

    X_train, y_train = train_df.drop('letter', axis=1), train_df['letter']
    X_test, y_test = test_df.drop('letter', axis=1), test_df['letter']

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'\n  准确率:     {acc:.4f}')
    print(f'  精确率:     {precision:.4f}')
    print(f'  召回率:     {recall:.4f}')
    print(f'  F1:         {f1:.4f}')
    print(f'  OOB Score:  {model.oob_score_:.4f}')
    print(f'  训练时间:   {train_time:.2f}s')
    print(f'  预测时间:   {pred_time:.2f}s')

    results = {
        'model': 'Baseline (default)',
        'accuracy': round(acc, 4),
        'precision_macro': round(precision, 4),
        'recall_macro': round(recall, 4),
        'f1_macro': round(f1, 4),
        'oob_score': round(model.oob_score_, 4),
        'train_time_sec': round(train_time, 2),
        'pred_time_sec': round(pred_time, 2),
        'params': {k: str(v) for k, v in model.get_params().items()}
    }

    with open(os.path.join(OUTPUT_DIR, 'results_baseline.json'), 'w') as f:
        json.dump(results, f, indent=2)
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'model_baseline.pkl'))

    print(f'\n[完成] 基线模型结果已保存')


if __name__ == '__main__':
    main()
