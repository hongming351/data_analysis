"""
数据集划分脚本 - 随机分层抽样
Letter Recognition 数据集 (A-Z 共 26 类)

划分方案：
  - 训练集 80% (16,000 条)
  - 测试集 20% (4,000 条)
  - 使用分层抽样 (stratify) 保证各类字母在训练/测试集中占比一致
  - 统一随机种子 random_state=42，确保可复现
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # 1. 读取原始数据
    df = pd.read_csv('letter-recognition.csv')
    print(f'原始数据: {df.shape[0]} 行, {df.shape[1]} 列')

    # 2. 分离特征与目标
    X = df.drop('letter', axis=1)
    y = df['letter']

    # 3. 分层划分 (随机种子 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,          # 20% 测试集
        random_state=42,        # 固定随机种子，保证可复现
        stratify=y              # 按字母分层抽样
    )

    # 4. 重组完整 DataFrame
    train_df = pd.concat([y_train, X_train], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)

    # 5. 保存为 CSV
    train_df.to_csv('letter_recognition_train.csv', index=False)
    test_df.to_csv('letter_recognition_test.csv', index=False)

    # 6. 输出统计信息
    print(f'\n✅ 数据集划分完成')
    print(f'   训练集: {len(train_df):>6} 条 ({len(train_df) * 100 // len(df)}%)')
    print(f'   测试集: {len(test_df):>6} 条 ({len(test_df) * 100 // len(df)}%)')
    print(f'   保存为: letter_recognition_train.csv / letter_recognition_test.csv')

    print(f'\n📊 训练集字母分布:')
    train_counts = train_df['letter'].value_counts().sort_index()
    for letter, count in train_counts.items():
        print(f'   {letter}: {count:>4}')

    print(f'\n📊 测试集字母分布:')
    test_counts = test_df['letter'].value_counts().sort_index()
    for letter, count in test_counts.items():
        print(f'   {letter}: {count:>4}')

    # 验证分层比例一致性
    print(f'\n📐 分层比例验证 (训练/测试 比例):')
    for letter in sorted(train_counts.index):
        train_c = train_counts[letter]
        test_c = test_counts[letter]
        ratio = train_c / (train_c + test_c) * 100
        print(f'   {letter}: 训练 {train_c:>4} / 测试 {test_c:>4} → 训练占比 {ratio:.1f}%')


if __name__ == '__main__':
    main()
