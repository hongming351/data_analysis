"""
绘制每个MLP模型对每个字母的预测准确率柱状图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, '..', 'letter_recognition_train.csv')
TEST_PATH = os.path.join(BASE_DIR, '..', 'letter_recognition_test.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
test_df = pd.read_csv(TEST_PATH)
letter_to_idx = {chr(ord('A') + i): i for i in range(26)}
idx_to_letter = {i: chr(ord('A') + i) for i in range(26)}

X_test = torch.from_numpy(test_df.drop('letter', axis=1).values.astype(np.float32))
y_test = torch.from_numpy(test_df['letter'].map(letter_to_idx).values.astype(np.int64))

letters = [chr(ord('A') + i) for i in range(26)]

# ============================================================
# 1. 基础标准 MLP
# ============================================================
print('=== 基础标准 MLP ===')
import mlp_01_basic as basic
basic_model = basic.BasicMLP().to(device)
basic_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'basic_mlp_best.pth')))
basic_model.eval()
with torch.no_grad():
    basic_pred = basic_model(X_test.to(device)).argmax(1).cpu().numpy()
basic_acc_per_class = {}
for i, letter in enumerate(letters):
    mask = y_test.numpy() == i
    basic_acc_per_class[letter] = (basic_pred[mask] == y_test.numpy()[mask]).mean()

# ============================================================
# 2. Dropout MLP
# ============================================================
print('=== Dropout MLP ===')
import mlp_02_dropout as drop
drop_model = drop.DropoutMLP().to(device)
drop_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'dropout_mlp_best.pth')))
drop_model.eval()
with torch.no_grad():
    drop_pred = drop_model(X_test.to(device)).argmax(1).cpu().numpy()
drop_acc_per_class = {}
for i, letter in enumerate(letters):
    mask = y_test.numpy() == i
    drop_acc_per_class[letter] = (drop_pred[mask] == y_test.numpy()[mask]).mean()

# ============================================================
# 3. BatchNorm MLP
# ============================================================
print('=== BatchNorm MLP ===')
import mlp_03_batchnorm as bn
bn_model = bn.BatchNormMLP().to(device)
bn_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'batchnorm_mlp_best.pth')))
bn_model.eval()
with torch.no_grad():
    bn_pred = bn_model(X_test.to(device)).argmax(1).cpu().numpy()
bn_acc_per_class = {}
for i, letter in enumerate(letters):
    mask = y_test.numpy() == i
    bn_acc_per_class[letter] = (bn_pred[mask] == y_test.numpy()[mask]).mean()

# ============================================================
# 4. AutoEncoder + 分类头
# ============================================================
print('=== AutoEncoder + 分类头 ===')
import mlp_04_autoencoder as ae_mod
ae_model = ae_mod.AutoEncoder().to(device)
clf_model = ae_mod.Classifier().to(device)
ckpt = torch.load(os.path.join(SAVE_DIR, 'ae_ft_best.pth'))
ae_model.load_state_dict(ckpt['ae'])
clf_model.load_state_dict(ckpt['clf'])
ae_model.eval()
clf_model.eval()
with torch.no_grad():
    z = ae_model.encode(X_test.to(device))
    ae_pred = clf_model(z).argmax(1).cpu().numpy()
ae_acc_per_class = {}
for i, letter in enumerate(letters):
    mask = y_test.numpy() == i
    ae_acc_per_class[letter] = (ae_pred[mask] == y_test.numpy()[mask]).mean()

# ============================================================
# 绘图：四个模型分开画，每个字母准确率
# ============================================================
model_data = [
    ('Basic MLP', '#3498db', basic_acc_per_class),
    ('Dropout MLP', '#e74c3c', drop_acc_per_class),
    ('BatchNorm MLP', '#2ecc71', bn_acc_per_class),
    ('AE + Classifier', '#1abc9c', ae_acc_per_class),
]

# 方式1：四个子图，每个模型一个图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, (name, color, data) in zip(axes, model_data):
    vals = [data[l] * 100 for l in letters]
    colors_bar = [color] * 26
    # 标出低于平均的字母
    avg_val = np.mean(vals)
    bars = ax.bar(letters, vals, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.axhline(y=avg_val, color='red', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.text(25.5, avg_val + 0.5, f'Avg: {avg_val:.1f}%', color='red', fontsize=9, ha='right')
    ax.set_title(f'{name} (Overall: {np.mean(vals):.2f}%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xlabel('Letter', fontsize=11)
    ax.set_ylim([70, 102])
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'已保存: per_class_accuracy.png')

# 方式2：对比柱状图（四个模型并列显示）
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(letters))
width = 0.2

colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#1abc9c']
model_names = ['Basic MLP', 'Dropout', 'BatchNorm', 'AE+CLF']
all_data = [
    [basic_acc_per_class[l] * 100 for l in letters],
    [drop_acc_per_class[l] * 100 for l in letters],
    [bn_acc_per_class[l] * 100 for l in letters],
    [ae_acc_per_class[l] * 100 for l in letters],
]

for i, (vals, name, color) in enumerate(zip(all_data, model_names, colors_list)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(letters, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Letter Accuracy Comparison across MLP Models', fontsize=14, fontweight='bold')
ax.set_ylim([65, 102])
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'per_class_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'已保存: per_class_accuracy_comparison.png')

print('Done!')
