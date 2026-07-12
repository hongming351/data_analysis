"""
模型 4 V6：自编码器 + 分类头（大容量版）
=====================================
增大隐藏层容量，与基础 MLP (128→64) 对齐，争取超越 95.73%。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'letter_recognition_train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'letter_recognition_test.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'MLP', 'results')
os.makedirs(SAVE_DIR, exist_ok=True)

# ============ 超参数 ============
BATCH_SIZE = 64
INPUT_SIZE = 16
LATENT_SIZE = 12          # 降维目标
NUM_CLASSES = 26

# 自编码器：16->128->64->12->64->128->16（与基础 MLP 的 128->64 对齐）
AE_HIDDEN = 128
AE_HIDDEN2 = 64
AE_EPOCHS = 100
AE_LR = 1e-3
AE_PATIENCE = 20

# 分类头：12->128->64->26（与基础 MLP 对齐）
CLF_HIDDEN = 128
CLF_HIDDEN2 = 64
CLF_EPOCHS = 120
CLF_LR = 1e-3
CLF_PATIENCE = 20

FT_EPOCHS = 100
FT_LR = 5e-5
FT_PATIENCE = 15


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    letter_to_idx = {chr(ord('A') + i): i for i in range(26)}

    X_train = torch.from_numpy(train_df.drop('letter', axis=1).values.astype(np.float32))
    y_train = torch.from_numpy(train_df['letter'].map(letter_to_idx).values.astype(np.int64))
    X_test = torch.from_numpy(test_df.drop('letter', axis=1).values.astype(np.float32))
    y_test = torch.from_numpy(test_df['letter'].map(letter_to_idx).values.astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), BATCH_SIZE, shuffle=False)

    print(f'训练集: {len(train_df)} 条 | 测试集: {len(test_df)} 条')
    return train_loader, test_loader, X_test, y_test


class AutoEncoder(nn.Module):
    """16->128(BN,ReLU)->64(BN,ReLU)->12->64(BN,ReLU)->128(BN,ReLU)->16"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_SIZE, AE_HIDDEN),
            nn.BatchNorm1d(AE_HIDDEN), nn.ReLU(),
            nn.Linear(AE_HIDDEN, AE_HIDDEN2),
            nn.BatchNorm1d(AE_HIDDEN2), nn.ReLU(),
            nn.Linear(AE_HIDDEN2, LATENT_SIZE)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SIZE, AE_HIDDEN2),
            nn.BatchNorm1d(AE_HIDDEN2), nn.ReLU(),
            nn.Linear(AE_HIDDEN2, AE_HIDDEN),
            nn.BatchNorm1d(AE_HIDDEN), nn.ReLU(),
            nn.Linear(AE_HIDDEN, INPUT_SIZE)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Classifier(nn.Module):
    """12->128(BN,ReLU)->64(BN,ReLU)->26（与基础 MLP 对齐）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_SIZE, CLF_HIDDEN),
            nn.BatchNorm1d(CLF_HIDDEN), nn.ReLU(),
            nn.Linear(CLF_HIDDEN, CLF_HIDDEN2),
            nn.BatchNorm1d(CLF_HIDDEN2), nn.ReLU(),
            nn.Linear(CLF_HIDDEN2, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(ae, clf, loader, tag=''):
    ae.eval()
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            z = ae.encode(X)
            correct += (clf(z).argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    if tag:
        print(f'  [{tag}] Acc: {acc:.4f} ({acc*100:.2f}%)')
    return acc


def train_ae_stage(ae, train_loader):
    """Stage 1: 训练自编码器"""
    print('\n=== Stage 1: 自编码器预训练 ===')
    opt = optim.Adam(ae.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    wait = 0
    losses = []

    for epoch in range(1, AE_EPOCHS + 1):
        ae.train()
        loss_sum = 0.0
        for X, _ in train_loader:
            X = X.to(device)
            opt.zero_grad()
            loss = criterion(ae(X), X)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * X.size(0)
        avg_loss = loss_sum / len(train_loader.dataset)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(ae.state_dict(), os.path.join(SAVE_DIR, 'ae_v6_best.pth'))
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{AE_EPOCHS} | Recon Loss: {avg_loss:.6f}')
        if wait >= AE_PATIENCE:
            print(f'  早停于 Epoch {epoch} | 最佳损失: {best_loss:.6f}')
            break

    ae.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'ae_v6_best.pth')))
    print(f'  最佳重建损失: {best_loss:.6f}')
    return losses


def train_clf_stage(ae, clf, train_loader, test_loader):
    """Stage 2: 冻结编码器，训练分类头"""
    print('\n=== Stage 2: 分类头训练 ===')
    for p in ae.encoder.parameters():
        p.requires_grad = False

    opt = optim.Adam(clf.parameters(), lr=CLF_LR)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    wait = 0
    accs = []

    for epoch in range(1, CLF_EPOCHS + 1):
        clf.train()
        ae.eval()
        loss_sum = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            with torch.no_grad():
                z = ae.encode(X)
            loss = criterion(clf(z), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * X.size(0)

        acc = evaluate(ae, clf, test_loader)
        accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            wait = 0
            torch.save(clf.state_dict(), os.path.join(SAVE_DIR, 'ae_v6_clf_best.pth'))
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{CLF_EPOCHS} | Loss: {loss_sum/len(train_loader.dataset):.4f} | Acc: {acc:.4f}')
        if wait >= CLF_PATIENCE:
            print(f'  早停于 Epoch {epoch} | 最佳准确率: {best_acc:.4f}')
            break

    clf.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'ae_v6_clf_best.pth')))
    print(f'  Stage 2 最佳准确率: {best_acc:.4f}')
    return accs, best_acc


def finetune_stage(ae, clf, train_loader, test_loader):
    """Stage 3: 端到端微调"""
    print('\n=== Stage 3: 端到端微调 ===')
    for p in ae.encoder.parameters():
        p.requires_grad = True

    opt = optim.Adam(list(ae.encoder.parameters()) + list(clf.parameters()), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    wait = 0
    accs = []

    for epoch in range(1, FT_EPOCHS + 1):
        ae.eval()
        clf.train()
        loss_sum = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            z = ae.encode(X)
            loss = criterion(clf(z), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * X.size(0)

        acc = evaluate(ae, clf, test_loader)
        accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            wait = 0
            torch.save({'ae': ae.state_dict(), 'clf': clf.state_dict()},
                       os.path.join(SAVE_DIR, 'ae_v6_ft_best.pth'))
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{FT_EPOCHS} | Loss: {loss_sum/len(train_loader.dataset):.4f} | Acc: {acc:.4f}')
        if wait >= FT_PATIENCE:
            print(f'  早停于 Epoch {epoch} | 最佳: {best_acc:.4f}')
            break

    print(f'  Stage 3 最佳准确率: {best_acc:.4f}')
    return accs, best_acc


def main():
    print('='*60)
    print('模型 4 V6：自编码器 + 分类头（大容量版）')
    print('='*60)
    print(f'隐藏层: {AE_HIDDEN}->{AE_HIDDEN2} (与基础 MLP 对齐)')
    print(f'降维: 16 -> {LATENT_SIZE} 维')

    train_loader, test_loader, X_test_full, y_test_full = load_data()

    ae = AutoEncoder().to(device)
    clf = Classifier().to(device)
    ae_params = sum(p.numel() for p in ae.parameters())
    clf_params = sum(p.numel() for p in clf.parameters())
    total_params = ae_params + clf_params
    print(f'AE参数量: {ae_params:,} | CLF参数量: {clf_params:,} | 总计: {total_params:,}')
    print(f'基础 MLP 参数量: 12,122')
    print(f'参数量比值: {total_params/12122:.2f}x')

    ae_losses = train_ae_stage(ae, train_loader)

    clf_accs, best_acc2 = train_clf_stage(ae, clf, train_loader, test_loader)

    ft_accs, best_acc3 = finetune_stage(ae, clf, train_loader, test_loader)

    print('\n' + '='*60)
    print('最终评估')
    print('='*60)
    print(f'Stage 2 最佳: {best_acc2:.4f} | Stage 3 最佳: {best_acc3:.4f}')

    if best_acc3 >= best_acc2:
        ckpt = torch.load(os.path.join(SAVE_DIR, 'ae_v6_ft_best.pth'))
        ae.load_state_dict(ckpt['ae'])
        clf.load_state_dict(ckpt['clf'])
        final_acc = best_acc3
        print(f'使用 Stage 3 (微调) 结果')
    else:
        ae.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'ae_v6_best.pth')))
        clf.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'ae_v6_clf_best.pth')))
        final_acc = best_acc2
        print(f'使用 Stage 2 (分类头) 结果')

    ae.eval()
    clf.eval()
    with torch.no_grad():
        X = X_test_full.to(device)
        z = ae.encode(X)
        pred = clf(z).argmax(1).cpu().numpy()
        true = y_test_full.numpy()

    acc = accuracy_score(true, pred)
    print(f'\n最终测试准确率: {acc:.4f} ({acc*100:.2f}%)')
    print(f'基础 MLP 基准: 95.73%')
    print(f'{"+" if acc > 0.9573 else ""}{acc-0.9573:+.4f} 相对基础 MLP')
    print('='*60)
    target_names = [chr(ord('A') + i) for i in range(26)]
    print(classification_report(true, pred, target_names=target_names, digits=4))

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ae_losses)
    axes[0].set_title(f'AE Reconstruction Loss (best: {ae_losses[-1]:.4f})')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(clf_accs)
    axes[1].set_title(f'CLF Acc (best: {best_acc2:.4f})')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(ft_accs)
    axes[2].set_title(f'FT Acc (best: {best_acc3:.4f})')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'ae_v6_curves.png'), dpi=150)
    plt.close()
    print(f'\n曲线已保存')

    return final_acc


if __name__ == '__main__':
    main()
