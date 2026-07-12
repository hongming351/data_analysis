"""
模型 3：带 BatchNorm 批归一化的 MLP
===================================
结构：输入层(16) → 隐藏层(128, BN, ReLU) → 隐藏层(64, BN, ReLU) → 输出层(26, Softmax)
- 在每个隐藏层后加入 BatchNorm 层
- 加速收敛、稳定训练
- 与基础 MLP 对比精度差异
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 设备配置 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ---------- 路径配置 ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'letter_recognition_train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'letter_recognition_test.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'MLP', 'results')
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 超参数 ----------
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
INPUT_SIZE = 16
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
NUM_CLASSES = 26
PATIENCE = 15

# ---------- 数据加载 ----------
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    letter_to_idx = {chr(ord('A') + i): i for i in range(26)}

    X_train = train_df.drop('letter', axis=1).values.astype(np.float32)
    y_train = train_df['letter'].map(letter_to_idx).values.astype(np.int64)
    X_test = test_df.drop('letter', axis=1).values.astype(np.float32)
    y_test = test_df['letter'].map(letter_to_idx).values.astype(np.int64)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f'训练集: {len(train_df)} 条')
    print(f'测试集: {len(test_df)} 条')
    print(f'特征维度: {INPUT_SIZE}')
    print(f'类别数: {NUM_CLASSES}')

    return train_loader, test_loader, X_test_t, y_test_t


# ---------- 模型定义 ----------
class BatchNormMLP(nn.Module):
    def __init__(self):
        super(BatchNormMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN1_SIZE),
            nn.BatchNorm1d(HIDDEN1_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
            nn.BatchNorm1d(HIDDEN2_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN2_SIZE, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


# ---------- 训练函数 ----------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler=None):
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        test_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['test_acc'].append(test_acc)

        if scheduler:
            scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'batchnorm_mlp_best.pth'))
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Test Acc: {test_acc:.4f}')

        if patience_counter >= PATIENCE:
            print(f'早停于 Epoch {epoch}，最佳测试准确率: {best_test_acc:.4f} (Epoch {best_epoch})')
            break

    return history, best_test_acc, best_epoch


# ---------- 评估函数 ----------
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        y_true = y_test.numpy()

    acc = accuracy_score(y_true, predicted)
    print(f'\n{"="*60}')
    print(f'最终测试准确率: {acc:.4f} ({acc*100:.2f}%)')
    print(f'{"="*60}')

    print('\n分类报告:')
    target_names = [chr(ord('A') + i) for i in range(26)]
    print(classification_report(y_true, predicted, target_names=target_names, digits=4))

    return y_true, predicted


# ---------- 绘图函数 ----------
def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (BatchNorm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['test_acc'], label='Test Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy (BatchNorm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'训练曲线已保存: {save_path}')


# ---------- 主函数 ----------
def main():
    print(f'{"="*60}')
    print('模型 3：带 BatchNorm 批归一化的 MLP')
    print(f'{"="*60}')
    print(f'结构: {INPUT_SIZE} → {HIDDEN1_SIZE}(BN,ReLU) → {HIDDEN2_SIZE}(BN,ReLU) → {NUM_CLASSES}')
    print(f'Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LEARNING_RATE} | 早停: {PATIENCE}')
    print(f'{"="*60}\n')

    train_loader, test_loader, X_test_t, y_test_t = load_data()

    model = BatchNormMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n模型参数量: {total_params:,}\n')

    start_time = time.time()
    history, best_acc, best_epoch = train_model(model, train_loader, test_loader, criterion, optimizer)
    train_time = time.time() - start_time
    print(f'\n训练耗时: {train_time:.2f} 秒')

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'batchnorm_mlp_best.pth')))
    y_true, y_pred = evaluate_model(model, X_test_t, y_test_t)

    plot_history(history, os.path.join(SAVE_DIR, 'batchnorm_mlp_curve.png'))

    return {
        'model_name': 'BatchNorm MLP',
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'train_time': train_time,
        'params': total_params,
        'history': history,
        'y_true': y_true,
        'y_pred': y_pred
    }


if __name__ == '__main__':
    result = main()
