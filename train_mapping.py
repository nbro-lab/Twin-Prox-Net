import argparse
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mdi_function import calculate_delta_S, calculate_R2

# 导入你原有的物理模块 (请确保这些文件在同级目录下)
from model_deep_snms import Learned_R2, Learned_S2
from torch.utils.data import DataLoader, Dataset


# =====================================================================
#  1. 内存映射极速 Dataset (防 RAM 爆炸的核心)
# ==========================================
class Mapping_Dataset_12Echo(Dataset):
    def __init__(self, data_dir, vol_names):
        """
        data_dir: .npy 文件所在的目录 (Stage 1 保存的路径)
        vol_names: 要加载的 volume 列表，例如 ['vol1', 'vol2', ..., 'vol9']
        """
        self.recon_files = [
            os.path.join(data_dir, f"{vol}_recon.npy") for vol in vol_names
        ]
        self.label_files = [
            os.path.join(data_dir, f"{vol}_label.npy") for vol in vol_names
        ]

        # 使用 mmap_mode='r' 进行内存映射读取，几百GB的数据也不会撑爆内存
        self.recon_data = [np.load(f, mmap_mode="r") for f in self.recon_files]
        self.label_data = [np.load(f, mmap_mode="r") for f in self.label_files]

        # 计算总切片数
        self.lengths = [data.shape[0] for data in self.recon_data]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]

        print(
            f" 成功加载 {len(vol_names)} 个 Volume，共包含 {self.total_length} 张切片。"
        )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 自动定位当前切片属于哪一个 Volume
        vol_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        if vol_idx == 0:
            slice_idx = idx
        else:
            slice_idx = idx - self.cum_lengths[vol_idx - 1]

        # 提取单层数据，转换为复数 tensor
        # 使用 np.array() 触发真正的磁盘读取
        recon_slice = np.array(self.recon_data[vol_idx][slice_idx])
        label_slice = np.array(self.label_data[vol_idx][slice_idx])

        return torch.tensor(recon_slice, dtype=torch.complex64), torch.tensor(
            label_slice, dtype=torch.complex64
        )


# =====================================================================
#  2. 独立的 12回波 Mapping 网络 (物理推断 + 卷积矫正)
# ==========================================
class DeepMTP_Mapping_12Echo(nn.Module):
    def __init__(self):
        super(DeepMTP_Mapping_12Echo, self).__init__()
        # 独立的两个参数矫正网络
        self.learned_S = Learned_S2()
        self.learned_R = Learned_R2()

    def forward(self, recon_im, mask_brain):
        """
        recon_im: 预先重建好的 12 回波复数图像 [nb, nc, ne(12), ny, nz]
        mask_brain: 大脑掩膜 [nb, ny, nz]
        """
        # 拆分出 FA4 和 FA16
        recon_FA4 = recon_im[:, :, 0:6, :, :]  # 前 6 个
        recon_FA16 = recon_im[:, :, 6:12, :, :]  # 后 6 个

        # 堆叠成 MDI 物理公式需要的结构: [nb, nc, na(2), ne(6), ny, nz]
        im_stacked = torch.stack([recon_FA4, recon_FA16], dim=2)

        # 提取用于计算的后 5 个回波 (摒弃第1个回波避免非稳态效应)
        s2 = im_stacked[:, :, :, 1:, :, :]  # 形状: [nb, nc, 2, 5, ny, nz]

        # 施加大脑掩膜
        s2_masked = s2 * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # 送入学习网络
        delta_S = self.learned_S(s2_masked, mask_brain)
        R2 = self.learned_R(s2_masked, mask_brain)

        return delta_S, R2


# =====================================================================
#  3. 主训练脚本
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data0/zhaojiaying/Mapping_Training_Data_12Echo",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data0/zhaojiaying/DeepSNMs_MTP_checkpoints/Mapping_Stage2",
    )
    args = parser.parse_args()

    # GPU 设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    results_file = os.path.join(
        args.save_dir,
        f"mapping_log_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
    )

    # 1. 准备 DataLoader
    print("----------- 初始化 DataLoader ----------")
    vol_names = [f"vol{i}" for i in range(1, 10)]  # vol1 到 vol9
    train_dataset = Mapping_Dataset_12Echo(args.data_dir, vol_names)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    # 2. 初始化网络和优化器
    print("----------- 初始化 Mapping 网络 ----------")
    mapping_net = DeepMTP_Mapping_12Echo().to(device)
    optim = torch.optim.Adam(
        mapping_net.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=40, gamma=0.5)

    # 3. 开始训练
    print("----------- 开始极速训练 ----------")
    for epoch in range(args.num_epoch):
        mapping_net.train()
        loss_epoch = []

        t0_epoch = time.time()
        for step, (recon_im, label_im) in enumerate(train_loader):
            recon_im = recon_im.to(device)
            label_im = label_im.to(device)

            nb = recon_im.size(0)

            # ====================================================
            # A. GPU 端生成大脑掩膜 (Mask) 和 归一化
            # ====================================================
            with torch.no_grad():
                # 1. 计算真实的 SOS 图像 (Sum of Squares)
                label_sos = torch.sqrt(
                    torch.sum(torch.abs(label_im) ** 2, dim=1)
                )  # [nb, 12, ny, nz]
                # 2. 在回波维度上求平均
                label_mean = label_sos.mean(dim=1)  # [nb, ny, nz]

                # 3. 动态计算每张切片的最大值，生成 Mask (阈值 0.05)
                max_vals = label_mean.view(nb, -1).max(dim=1)[0].view(nb, 1, 1)
                mask_brain = (label_mean > 0.05 * max_vals).float()

            # ====================================================
            # B. 准备金标准 (Ground Truth) 物理参数
            # ====================================================
            with torch.no_grad():
                label_FA4 = label_im[:, :, 0:6, :, :]
                label_FA16 = label_im[:, :, 6:12, :, :]
                label_stacked = torch.stack([label_FA4, label_FA16], dim=2)
                s2_ref = label_stacked[:, :, :, 1:, :, :]

                # 计算物理真值并过滤 NaN/Inf
                deltaS_ref = calculate_delta_S(s2_ref) * mask_brain
                deltaS_ref = torch.nan_to_num(
                    deltaS_ref, nan=0.0, posinf=0.0, neginf=0.0
                )

                R2_ref = calculate_R2(s2_ref) * mask_brain
                R2_ref = torch.nan_to_num(R2_ref, nan=0.0, posinf=0.0, neginf=0.0)

            # ====================================================
            # C. 前向传播 & 计算损失
            # ====================================================
            pred_deltaS_complex, pred_R2_complex = mapping_net(recon_im, mask_brain)

            #  核心修复 1：物理意义对齐，提取复数的模长 (Magnitude) 变为实数张量
            # torch.abs 在底层是完美支持复数反向传播的
            pred_deltaS_mag = torch.abs(pred_deltaS_complex)
            pred_R2_mag = torch.abs(pred_R2_complex)

            #  核心修复 2：现在的张量是 float32 实数了，nan_to_num 完美支持求导！
            pred_deltaS = torch.nan_to_num(
                pred_deltaS_mag, nan=0.0, posinf=0.0, neginf=0.0
            )
            pred_R2 = torch.nan_to_num(pred_R2_mag, nan=0.0, posinf=0.0, neginf=0.0)

            # 使用 L1 Loss (MAE) 计算误差 (现在预测值和真值都是纯实数了，门当户对)
            loss_S = F.l1_loss(pred_deltaS, deltaS_ref)
            loss_R = F.l1_loss(pred_R2, R2_ref)

            loss_total = loss_S + loss_R
            # pred_deltaS, pred_R2 = mapping_net(recon_im, mask_brain)

            # pred_deltaS = torch.nan_to_num(pred_deltaS, nan=0.0, posinf=0.0, neginf=0.0)
            # pred_R2 = torch.nan_to_num(pred_R2, nan=0.0, posinf=0.0, neginf=0.0)

            # # 使用 L1 Loss (MAE) 作为强鲁棒性的物理映射损失
            # loss_S = F.l1_loss(pred_deltaS, deltaS_ref)
            # loss_R = F.l1_loss(pred_R2, R2_ref)

            # loss_total = loss_S + loss_R

            # ====================================================
            # D. 反向传播
            # ====================================================
            optim.zero_grad()
            loss_total.backward()
            optim.step()

            loss_epoch.append(loss_total.item())

            if (step + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.num_epoch}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss_total.item():.4f} | L_S: {loss_S.item():.4f} | L_R: {loss_R.item():.4f}"
                )

        # 学习率衰减
        scheduler.step()

        # Epoch 总结与保存
        avg_loss = np.mean(loss_epoch)
        t_epoch = time.time() - t0_epoch
        print(
            f"\n Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 耗时: {t_epoch:.2f}s | LR: {optim.param_groups[0]['lr']:.6f}\n"
        )

        with open(results_file, "a") as f:
            f.write(
                f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {optim.param_groups[0]['lr']:.6f}\n"
            )

        # 每 5 个 Epoch 保存一次权重
        if (epoch + 1) % 5 == 0:
            torch.save(
                mapping_net.state_dict(),
                os.path.join(args.save_dir, f"mapping_params_{epoch+1}.pkl"),
            )
            print(f"💾 已保存权重: mapping_params_{epoch+1}.pkl")

    print("\n Mapping 网络训练圆满完成！")
