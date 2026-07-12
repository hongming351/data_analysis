"""
ExtraTrees 参数探索
"""
import pandas as pd, numpy as np, time, json, joblib, os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_train.csv'))
test = pd.read_csv(os.path.join(BASE_DIR, 'letter_recognition_test.csv'))
X_train, y_train = train.drop('letter', axis=1), train['letter']
X_test, y_test = test.drop('letter', axis=1), test['letter']

results = {}

# 1. 默认参数 (150树)
et = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
t0 = time.time(); et.fit(X_train, y_train)
y = et.predict(X_test)
results['ET_150_default'] = {'acc': round(accuracy_score(y_test,y),4), 'f1': round(f1_score(y_test,y,average='macro'),4), 'time': round(time.time()-t0,2)}
print(f'ET_150_default: acc={results["ET_150_default"]["acc"]:.4f}  f1={results["ET_150_default"]["f1"]:.4f}')

# 2. 增加 n_estimators
for n in [200, 300, 400, 500]:
    et = ExtraTreesClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    t0 = time.time(); et.fit(X_train, y_train)
    y = et.predict(X_test)
    key = f'ET_{n}_default'
    results[key] = {'acc': round(accuracy_score(y_test,y),4), 'f1': round(f1_score(y_test,y,average='macro'),4), 'time': round(time.time()-t0,2)}
    print(f'{key}: acc={results[key]["acc"]:.4f}  f1={results[key]["f1"]:.4f}')

# 3. 调 criterion=max_features 组合
for crit in ['gini', 'entropy']:
    for mf in ['sqrt', 'log2', None]:
        et = ExtraTreesClassifier(n_estimators=300, criterion=crit, max_features=mf, random_state=42, n_jobs=-1)
        t0 = time.time(); et.fit(X_train, y_train)
        y = et.predict(X_test)
        key = f'ET_300_{crit}_{mf}'
        results[key] = {'acc': round(accuracy_score(y_test,y),4), 'f1': round(f1_score(y_test,y,average='macro'),4), 'time': round(time.time()-t0,2)}
        print(f'{key}: acc={results[key]["acc"]:.4f}  f1={results[key]["f1"]:.4f}')

# 4. bootstrap 影响
for boot in [True, False]:
    et = ExtraTreesClassifier(n_estimators=300, bootstrap=boot, random_state=42, n_jobs=-1)
    t0 = time.time(); et.fit(X_train, y_train)
    y = et.predict(X_test)
    key = f'ET_300_boot{boot}'
    results[key] = {'acc': round(accuracy_score(y_test,y),4), 'f1': round(f1_score(y_test,y,average='macro'),4), 'time': round(time.time()-t0,2)}
    print(f'{key}: acc={results[key]["acc"]:.4f}  f1={results[key]["f1"]:.4f}')

# 5. min_samples_leaf=1, min_samples_split=2 (和RF最优一致)
et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1, min_samples_split=2, random_state=42, n_jobs=-1)
t0 = time.time(); et.fit(X_train, y_train)
y = et.predict(X_test)
results['ET_300_leaf1_split2'] = {'acc': round(accuracy_score(y_test,y),4), 'f1': round(f1_score(y_test,y,average='macro'),4), 'time': round(time.time()-t0,2)}
print(f'ET_300_leaf1_split2: acc={results["ET_300_leaf1_split2"]["acc"]:.4f}  f1={results["ET_300_leaf1_split2"]["f1"]:.4f}')

# 排名
print('\n' + '='*60)
print('最终排名 (按 F1)')
print('='*60)
ranked = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
for rank, (name, m) in enumerate(ranked[:10], 1):
    print(f'{rank}. {name:<30s} acc={m["acc"]:.4f}  f1={m["f1"]:.4f}  {m["time"]:.2f}s')

with open(os.path.join(OUTPUT_DIR, 'results_et_search.json'), 'w') as f:
    json.dump(results, f, indent=2)

print('\n[完成]')
