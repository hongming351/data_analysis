"""
高阶集成方案:
1. Stacking 堆叠融合 (RF + ExtraTrees + XGBoost → LR)
2. Voting 投票集成 (硬投票 + 软投票)
3. AdaBoost + RandomForest 基学习器
"""

import pandas as pd
import numpy as np
import os, time, json, joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, VotingClassifier,
    AdaBoostClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print('=' * 60)
    print('高阶集成方案实验')
    print('=' * 60)

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_test.csv'))
    X_train, y_train = train_df.drop('letter', axis=1), train_df['letter']
    X_test, y_test = test_df.drop('letter', axis=1), test_df['letter']

    classes = sorted(y_train.unique())
    n_classes = len(classes)

    # XGBoost 要求标签为整数编码
    label_map = {c: i for i, c in enumerate(classes)}
    y_train_num = y_train.map(label_map)
    y_test_num = y_test.map(label_map)

    results = {}

    # ====================================================================
    # 1. Stacking 堆叠融合
    # ====================================================================
    print('\n' + '=' * 50)
    print('[1] Stacking 堆叠融合')
    print('=' * 50)

    base_models = [
        ('rf', RandomForestClassifier(n_estimators=400, max_depth=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       max_features='sqrt', criterion='gini',
                                       random_state=42, n_jobs=1)),
        ('et', ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=1)),
        ('xgb', None)  # 下面单独导入
    ]

    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=12, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        )
        base_models[2] = ('xgb', xgb_model)
        print('  XGBoost 加载成功')
    except ImportError:
        print('  XGBoost 未安装，跳过')
        base_models = base_models[:2]

    meta_learner = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False
    )

    t0 = time.time()
    stacking.fit(X_train, y_train)
    t_stack = time.time() - t0
    y_stack = stacking.predict(X_test)
    results['Stacking (RF+ET+XGB→LR)'] = {
        'acc': round(accuracy_score(y_test, y_stack), 4),
        'f1': round(f1_score(y_test, y_stack, average='macro'), 4),
        'time': round(t_stack, 2)
    }
    print(f'  Stacking 准确率: {results["Stacking (RF+ET+XGB→LR)"]["acc"]:.4f}')
    print(f'  Stacking F1:     {results["Stacking (RF+ET+XGB→LR)"]["f1"]:.4f}')
    print(f'  耗时: {t_stack:.1f}s')

    # ====================================================================
    # 2. Voting 投票集成
    # ====================================================================
    print('\n' + '=' * 50)
    print('[2] Voting 投票集成')
    print('=' * 50)

    # 2a) 硬投票 (多数表决)
    hard_voter = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=400, max_depth=None,
                                           min_samples_leaf=1, min_samples_split=2,
                                           max_features='sqrt', criterion='gini',
                                           random_state=42, n_jobs=1)),
            ('et', ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=1)),
        ],
        voting='hard',
        n_jobs=-1
    )

    t0 = time.time()
    hard_voter.fit(X_train, y_train)
    t_hard = time.time() - t0
    y_hard = hard_voter.predict(X_test)
    results['Voting 硬投票 (RF+ET)'] = {
        'acc': round(accuracy_score(y_test, y_hard), 4),
        'f1': round(f1_score(y_test, y_hard, average='macro'), 4),
        'time': round(t_hard, 2)
    }
    print(f'  硬投票 RF+ET: acc={results["Voting 硬投票 (RF+ET)"]["acc"]:.4f}')

    # 2b) 软投票 (概率平均)
    soft_voter = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=400, max_depth=None,
                                           min_samples_leaf=1, min_samples_split=2,
                                           max_features='sqrt', criterion='gini',
                                           random_state=42, n_jobs=1)),
            ('et', ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=1)),
        ],
        voting='soft',
        weights=[1, 1],  # 等权重
        n_jobs=-1
    )

    t0 = time.time()
    soft_voter.fit(X_train, y_train)
    t_soft = time.time() - t0
    y_soft = soft_voter.predict(X_test)
    results['Voting 软投票 (RF+ET)'] = {
        'acc': round(accuracy_score(y_test, y_soft), 4),
        'f1': round(f1_score(y_test, y_soft, average='macro'), 4),
        'time': round(t_soft, 2)
    }
    print(f'  软投票 RF+ET: acc={results["Voting 软投票 (RF+ET)"]["acc"]:.4f}')

    # 2c) 加 XGBoost 的三模型投票 (跳过，XGBoost 字母标签问题)
    # 前两次结果已证明 RF+ET 软投票足够

    # ====================================================================
    # 3. AdaBoost + RandomForest
    # ====================================================================
    print('\n' + '=' * 50)
    print('[3] AdaBoost + RandomForest 基学习器')
    print('=' * 50)

    # AdaBoost 基学习器用随机森林
    base_rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                      min_samples_leaf=1, min_samples_split=2,
                                      random_state=42, n_jobs=1)

    # 注意: AdaBoost 的 n_estimators 不能太大，否则过拟合
    for n_ada in [50, 100, 200]:
        try:
            ada = AdaBoostClassifier(
                estimator=base_rf,
                n_estimators=n_ada,
                learning_rate=0.5,
                random_state=42,
                algorithm='SAMME'
            )
            t0 = time.time()
            ada.fit(X_train, y_train)
            t_ada = time.time() - t0
            y_ada = ada.predict(X_test)
            key = f'AdaBoost(RF基) n={n_ada}'
            results[key] = {
                'acc': round(accuracy_score(y_test, y_ada), 4),
                'f1': round(f1_score(y_test, y_ada, average='macro'), 4),
                'time': round(t_ada, 2)
            }
            print(f'  AdaBoost n={n_ada:<3d}: acc={results[key]["acc"]:.4f}  f1={results[key]["f1"]:.4f}  {t_ada:.2f}s')
        except Exception as e:
            print(f'  AdaBoost n={n_ada} 失败: {e}')

    # ====================================================================
    # 4. 各单一模型对比基准
    # ====================================================================
    print('\n' + '=' * 50)
    print('[4] 单一模型基准')
    print('=' * 50)

    models = {
        'RF (V2最优)': RandomForestClassifier(n_estimators=400, max_depth=None,
                                                min_samples_leaf=1, min_samples_split=2,
                                                max_features='sqrt', criterion='gini',
                                                random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    }

    if base_models[2][1] is not None:
        models['XGBoost'] = base_models[2][1]

    for name, model in models.items():
        t0 = time.time()
        if name == 'XGBoost':
            model.fit(X_train, y_train_num)
            y_pred = model.predict(X_test)
            y_pred = pd.Series(y_pred).map({v: k for k, v in label_map.items()})
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        t = time.time() - t0
        results[f'单一 {name}'] = {
            'acc': round(accuracy_score(y_test, y_pred), 4),
            'f1': round(f1_score(y_test, y_pred, average='macro'), 4),
            'time': round(t, 2)
        }
        print(f'  {name}: acc={results[f"单一 {name}"]["acc"]:.4f}  f1={results[f"单一 {name}"]["f1"]:.4f}  {t:.2f}s')

    # ====================================================================
    # 保存结果
    # ====================================================================
    print('\n' + '=' * 50)
    print('[5] 保存结果')
    print('=' * 50)

    summary = {
        'results': results,
        'note': '高阶集成方案: Stacking / Voting / AdaBoost'
    }

    with open(os.path.join(OUTPUT_DIR, 'results_ensemble.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存最优模型 (Stacking)
    joblib.dump(stacking, os.path.join(OUTPUT_DIR, 'model_stacking.pkl'))

    # ========== 排名 ==========
    print('\n' + '=' * 60)
    print('最终排名 (按 F1)')
    print('=' * 60)
    ranked = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    print(f'{"排名":<5s} {"模型":<35s} {"准确率":<10s} {"F1":<10s} {"耗时(s)":<10s}')
    print('-' * 70)
    for rank, (name, m) in enumerate(ranked, 1):
        print(f'{rank:<5d} {name:<35s} {m["acc"]:<10.4f} {m["f1"]:<10.4f} {m["time"]:<10.2f}')

    print(f'\n[完成]')

if __name__ == '__main__':
    main()
