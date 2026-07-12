"""
PCA 降维 + 随机森林实验

思路: 16 维特征做 PCA，保留前 k 个主成分 (k=8~15)，
去除线性相关冗余，然后喂入随机森林 / ExtraTrees。
"""

import pandas as pd
import numpy as np
import os, time, json, joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print('=' * 60)
    print('PCA 降维 + 随机森林实验')
    print('=' * 60)

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_test.csv'))

    X_train, y_train = train_df.drop('letter', axis=1), train_df['letter']
    X_test, y_test = test_df.drop('letter', axis=1), test_df['letter']
    feature_names = list(X_train.columns)

    # ========== 1. 先看 PCA 方差解释率 ==========
    print('\n[1] PCA 方差解释率分析')
    pca_full = PCA().fit(X_train)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)

    print(f'  主成分  方差解释率(累计)')
    for i in range(16):
        print(f'  PC{i+1:<4d}  {cumsum[i]:.4f}')

    # 找到达到 95%, 98%, 99.5% 所需的主成分数
    k95 = np.argmax(cumsum >= 0.95) + 1
    k98 = np.argmax(cumsum >= 0.98) + 1
    k99 = np.argmax(cumsum >= 0.995) + 1
    print(f'\n  95% 方差: {k95} 个主成分')
    print(f'  98% 方差: {k98} 个主成分')
    print(f'  99.5%方差: {k99} 个主成分')

    # ========== 2. 尝试不同 k 值 ==========
    print('\n[2] 不同主成分数 + 随机森林')
    k_values = list(range(8, 17))  # 8~16

    all_results = {}

    for k in k_values:
        # PCA 降维
        pca = PCA(n_components=k, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # RF (使用 GridSearch V2 最优参数)
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            min_samples_split=2, max_features='sqrt', criterion='gini',
            random_state=42, n_jobs=-1
        )

        t0 = time.time()
        rf.fit(X_train_pca, y_train)
        train_time = time.time() - t0

        y_pred = rf.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        all_results[k] = {
            'accuracy': round(acc, 4),
            'f1_macro': round(f1, 4),
            'explained_var': round(float(cumsum[k-1]), 4),
            'train_time_sec': round(train_time, 2)
        }

        print(f'  k={k:<2d}  (方差={cumsum[k-1]:.2%})  acc={acc:.4f}  f1={f1:.4f}  {train_time:.2f}s')

    # ========== 3. 最佳 k 用 ExtraTrees 再试 ==========
    print('\n[3] 最佳 PCA 方案 + ExtraTrees')
    best_k = max(all_results, key=lambda k: all_results[k]['accuracy'])
    print(f'  最佳 k = {best_k}')

    pca_best = PCA(n_components=best_k, random_state=42)
    X_train_pca = pca_best.fit_transform(X_train)
    X_test_pca = pca_best.transform(X_test)

    et = ExtraTreesClassifier(
        n_estimators=150, random_state=42, n_jobs=-1
    )
    t0 = time.time()
    et.fit(X_train_pca, y_train)
    et_time = time.time() - t0
    y_pred_et = et.predict(X_test_pca)
    acc_et = accuracy_score(y_test, y_pred_et)
    f1_et = f1_score(y_test, y_pred_et, average='macro')

    print(f'  PCA({best_k})+ExtraTrees: acc={acc_et:.4f}  f1={f1_et:.4f}  {et_time:.2f}s')

    all_results[f'PCA_{best_k}+ExtraTrees'] = {
        'accuracy': round(acc_et, 4),
        'f1_macro': round(f1_et, 4),
        'explained_var': round(float(cumsum[best_k-1]), 4),
        'train_time_sec': round(et_time, 2),
        'note': 'ExtraTrees on PCA features'
    }

    # ========== 4. 无 PCA 的原始 RF/ET 作为对比基准 ==========
    print('\n[4] 对比: 原始特征 (无 PCA)')
    rf_raw = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=1,
        min_samples_split=2, max_features='sqrt', criterion='gini',
        random_state=42, n_jobs=-1
    )
    t0 = time.time()
    rf_raw.fit(X_train, y_train)
    t_raw = time.time() - t0
    y_raw = rf_raw.predict(X_test)
    all_results['原始RF(无PCA)'] = {
        'accuracy': round(accuracy_score(y_test, y_raw), 4),
        'f1_macro': round(f1_score(y_test, y_raw, average='macro'), 4),
        'train_time_sec': round(t_raw, 2),
        'note': '原始16维特征, GridSearch V2最优参数'
    }
    acc_rf = all_results['原始RF(无PCA)']['accuracy']
    f1_rf = all_results['原始RF(无PCA)']['f1_macro']
    print(f'  原始RF: acc={acc_rf:.4f}  f1={f1_rf:.4f}  {t_raw:.2f}s')

    et_raw = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    t0 = time.time()
    et_raw.fit(X_train, y_train)
    t_et_raw = time.time() - t0
    y_et_raw = et_raw.predict(X_test)
    all_results['原始ET(无PCA)'] = {
        'accuracy': round(accuracy_score(y_test, y_et_raw), 4),
        'f1_macro': round(f1_score(y_test, y_et_raw, average='macro'), 4),
        'train_time_sec': round(t_et_raw, 2),
        'note': '原始16维特征, ExtraTrees'
    }
    acc_et = all_results['原始ET(无PCA)']['accuracy']
    f1_et = all_results['原始ET(无PCA)']['f1_macro']
    print(f'  原始ET: acc={acc_et:.4f}  f1={f1_et:.4f}  {t_et_raw:.2f}s')

    # ========== 保存结果 ==========
    summary = {
        'pca_explained_var_ratio': [round(float(x), 4) for x in pca_full.explained_variance_ratio_],
        'pca_cumsum': [round(float(x), 4) for x in cumsum],
        'k_for_95percent': int(k95),
        'k_for_98percent': int(k98),
        'k_for_995percent': int(k99),
        'best_pca_k': int(best_k),
        'results': {str(k): v for k, v in all_results.items()}
    }

    with open(os.path.join(OUTPUT_DIR, 'results_pca_rf.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存最佳 PCA 模型
    joblib.dump((pca_best, et), os.path.join(OUTPUT_DIR, 'model_pca_et.pkl'))

    # ========== 打印最终排名 ==========
    print('\n' + '=' * 60)
    print('最终排名 (按 F1)')
    print('=' * 60)
    ranked = sorted(all_results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    print(f'{"排名":<6s} {"模型":<28s} {"准确率":<10s} {"F1":<10s} {"耗时(s)":<10s}')
    print('-' * 64)
    for rank, (name, m) in enumerate(ranked, 1):
        label = f'PCA(k={name})' if isinstance(name, int) else name
        print(f'{rank:<6d} {label:<28s} {m["accuracy"]:<10.4f} {m["f1_macro"]:<10.4f} {m["train_time_sec"]:<10.2f}')

    print(f'\n[完成] 结果已保存至 results_pca_rf.json')


if __name__ == '__main__':
    main()
