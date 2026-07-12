"""
特征工程 + 随机森林实验

原字段:
  x_box, y_box:        边框左上角坐标
  width, high:         边框宽高
  onpix:               ON 像素总数
  x_bar, y_bar:        水平/垂直方向像素重心
  x2bar, y2bar, xybar: 二阶矩 (方差/协方差)
  x2ybr, xy2br:        三阶矩 (偏度相关)
  x_ege, xegvy:        水平方向边缘计数
  y_ege, yegvx:        垂直方向边缘计数

构造的组合特征:
  1. 长宽比:        width / high                  → 区分宽扁/窄高字母
  2. 像素密度:      onpix / (width * high)        → 笔画粗细
  3. 重心偏移:      x_bar - x_box, y_bar - y_box → 对称性
  4. 像素占面积比:  onpix / width, onpix / high   → 笔画分布
  5. 重心偏离中心:  (x_bar - x_box - width/2), (y_bar - y_box - high/2) → 重心偏移比例
  6. 边缘密度:      x_ege / width, y_ege / high   → 每单位宽度/高度的边缘数
  7. 纵横边缘比:    x_ege / (y_ege + 1)           → 边缘方向倾向
  8. 二阶矩比值:    x2bar / (y2bar + 1)           → 水平/垂直展开比
  9. 面积:          width * high                   → 字母大小
  10. 周长:         2 * (width + high)             → 字母轮廓
"""

import pandas as pd
import numpy as np
import os, time, json, joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def add_engineered_features(df):
    """添加组合特征，返回新 DataFrame"""
    d = df.copy()
    eps = 1e-6  # 避免除零

    # 1. 长宽比
    d['aspect_ratio'] = d['width'] / (d['high'] + eps)

    # 2. 像素密度
    d['pixel_density'] = d['onpix'] / (d['width'] * d['high'] + eps)

    # 3. 重心偏移（绝对偏移量）
    d['x_offset'] = d['x_bar'] - d['x_box']
    d['y_offset'] = d['y_bar'] - d['y_box']

    # 4. 每行列平均像素数
    d['pix_per_width'] = d['onpix'] / (d['width'] + eps)
    d['pix_per_high'] = d['onpix'] / (d['high'] + eps)

    # 5. 重心偏离中心（相对偏移比例）
    d['x_center_dev'] = (d['x_bar'] - d['x_box'] - d['width'] / 2) / (d['width'] + eps)
    d['y_center_dev'] = (d['y_bar'] - d['y_box'] - d['high'] / 2) / (d['high'] + eps)

    # 6. 边缘密度
    d['edge_density_x'] = d['x_ege'] / (d['width'] + eps)
    d['edge_density_y'] = d['y_ege'] / (d['high'] + eps)

    # 7. 纵横边缘比
    d['edge_ratio'] = d['x_ege'] / (d['y_ege'] + eps)

    # 8. 二阶矩比值（水平展开 vs 垂直展开）
    d['moment_ratio'] = d['x2bar'] / (d['y2bar'] + eps)

    # 9. 面积和周长
    d['area'] = d['width'] * d['high']
    d['perimeter'] = 2 * (d['width'] + d['high'])

    # 10. 三阶矩（偏度指标）— 用来判断笔画分布的不对称性
    d['skew_x'] = d['x2ybr'] / (d['x2bar'] + eps)
    d['skew_y'] = d['xy2br'] / (d['y2bar'] + eps)

    return d


def main():
    print('=' * 60)
    print('特征工程 + 随机森林实验')
    print('=' * 60)

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_test.csv'))

    y_train, y_test = train_df['letter'], test_df['letter']

    original_features = ['x_box', 'y_box', 'width', 'high', 'onpix', 'x_bar', 'y_bar',
                         'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x_ege', 'xegvy',
                         'y_ege', 'yegvx']

    # ========== 1. 添加组合特征 ==========
    print('\n[1] 构造组合特征...')
    train_fe = add_engineered_features(train_df.drop('letter', axis=1))
    test_fe = add_engineered_features(test_df.drop('letter', axis=1))

    engineered_cols = [c for c in train_fe.columns if c not in original_features]
    print(f'  原始特征: {len(original_features)} 个')
    print(f'  组合特征: {len(engineered_cols)} 个 ({", ".join(engineered_cols)})')
    print(f'  总特征数: {train_fe.shape[1]} 个')

    # ========== 2. 训练对比 ==========
    print('\n[2] 训练模型对比...')

    results = {}

    # ---- 原始特征 + RF ----
    rf_orig = RandomForestClassifier(n_estimators=400, max_depth=None,
                                     min_samples_leaf=1, min_samples_split=2,
                                     max_features='sqrt', criterion='gini',
                                     random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_orig.fit(train_df[original_features], y_train)
    t1 = time.time() - t0
    y1 = rf_orig.predict(test_df[original_features])
    results['原始RF'] = {
        'acc': round(accuracy_score(y_test, y1), 4),
        'f1': round(f1_score(y_test, y1, average='macro'), 4),
        'time': round(t1, 2)
    }
    print(f'  原始RF:       acc={results["原始RF"]["acc"]:.4f}  f1={results["原始RF"]["f1"]:.4f}  {t1:.2f}s')

    # ---- 组合特征 + RF ----
    rf_fe = RandomForestClassifier(n_estimators=400, max_depth=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   max_features='sqrt', criterion='gini',
                                   random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_fe.fit(train_fe, y_train)
    t2 = time.time() - t0
    y2 = rf_fe.predict(test_fe)
    results['组合特征RF'] = {
        'acc': round(accuracy_score(y_test, y2), 4),
        'f1': round(f1_score(y_test, y2, average='macro'), 4),
        'time': round(t2, 2)
    }
    print(f'  组合特征RF:   acc={results["组合特征RF"]["acc"]:.4f}  f1={results["组合特征RF"]["f1"]:.4f}  {t2:.2f}s')

    # ---- 组合特征 + ExtraTrees ----
    et_fe = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    t0 = time.time()
    et_fe.fit(train_fe, y_train)
    t3 = time.time() - t0
    y3 = et_fe.predict(test_fe)
    results['组合特征ET'] = {
        'acc': round(accuracy_score(y_test, y3), 4),
        'f1': round(f1_score(y_test, y3, average='macro'), 4),
        'time': round(t3, 2)
    }
    print(f'  组合特征ET:   acc={results["组合特征ET"]["acc"]:.4f}  f1={results["组合特征ET"]["f1"]:.4f}  {t3:.2f}s')

    # ---- 原始特征 + ExtraTrees (对比) ----
    et_orig = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    t0 = time.time()
    et_orig.fit(train_df[original_features], y_train)
    t4 = time.time() - t0
    y4 = et_orig.predict(test_df[original_features])
    results['原始ET'] = {
        'acc': round(accuracy_score(y_test, y4), 4),
        'f1': round(f1_score(y_test, y4, average='macro'), 4),
        'time': round(t4, 2)
    }
    print(f'  原始ET:       acc={results["原始ET"]["acc"]:.4f}  f1={results["原始ET"]["f1"]:.4f}  {t4:.2f}s')

    # ========== 3. 特征重要性分析 ==========
    print('\n[3] 特征重要性排名 (组合特征RF)')
    importances = rf_fe.feature_importances_
    all_feature_names = list(train_fe.columns)
    indices = np.argsort(importances)[::-1]

    print(f'  {"排名":<4s} {"特征名":<20s} {"重要性":<10s} {"来源":<15s}')
    print('  ' + '-' * 49)
    for i in range(min(20, len(all_feature_names))):
        idx = indices[i]
        name = all_feature_names[idx]
        is_engineered = '⭐' if name in engineered_cols else ''
        print(f'  {i+1:<4d} {name:<20s} {importances[idx]:<10.4f} {is_engineered}')

    # ========== 4. 只使用 Top 组合特征 + 原始特征 ==========
    print('\n[4] 特征筛选: 保留重要组合特征...')
    # 选出重要性 >= 0.01 的组合特征
    feat_imp = [(all_feature_names[i], importances[i]) for i in range(len(all_feature_names))]
    feat_imp.sort(key=lambda x: x[1], reverse=True)

    # 从组合特征中选重要的
    selected_engineered = []
    for name, imp in feat_imp:
        if name in engineered_cols and imp >= 0.01:
            selected_engineered.append(name)
    print(f'  保留的组合特征: {selected_engineered}')

    selected_features = original_features + selected_engineered
    print(f'  最终特征数: {len(selected_features)}')

    rf_sel = RandomForestClassifier(n_estimators=400, max_depth=None,
                                    min_samples_leaf=1, min_samples_split=2,
                                    max_features='sqrt', criterion='gini',
                                    random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_sel.fit(train_fe[selected_features], y_train)
    t5 = time.time() - t0
    y5 = rf_sel.predict(test_fe[selected_features])
    results['精选特征RF'] = {
        'acc': round(accuracy_score(y_test, y5), 4),
        'f1': round(f1_score(y_test, y5, average='macro'), 4),
        'time': round(t5, 2)
    }
    acc5 = results['精选特征RF']['acc']
    f15 = results['精选特征RF']['f1']
    print(f'  精选特征RF:   acc={acc5:.4f}  f1={f15:.4f}  {t5:.2f}s')

    # ========== 5. 保存 ==========
    print('\n[5] 保存结果...')
    summary = {
        'original_features': original_features,
        'engineered_features': engineered_cols,
        'selected_engineered': selected_engineered,
        'feature_importance': [
            {'name': all_feature_names[i], 'importance': round(float(importances[i]), 4)}
            for i in range(len(all_feature_names))
        ],
        'results': results
    }

    with open(os.path.join(OUTPUT_DIR, 'results_feature_engineering.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    joblib.dump(rf_fe, os.path.join(OUTPUT_DIR, 'model_fe_rf.pkl'))

    # ========== 排名 ==========
    print('\n' + '=' * 60)
    print('最终排名')
    print('=' * 60)
    ranked = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    print(f'{"排名":<6s} {"模型":<20s} {"准确率":<10s} {"F1":<10s} {"耗时(s)":<10s}')
    print('-' * 56)
    for rank, (name, m) in enumerate(ranked, 1):
        print(f'{rank:<6d} {name:<20s} {m["acc"]:<10.4f} {m["f1"]:<10.4f} {m["time"]:<10.2f}')

    print(f'\n[完成] 结果已保存')


if __name__ == '__main__':
    main()
