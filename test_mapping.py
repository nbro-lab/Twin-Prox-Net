import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import c2r, ifft2c, readcfl, sos, writecfl

# =====================================================================
#  0. CFL 格式保存工具函数
# =====================================================================


# =====================================================================
#  1. 物理组件 (纯传统金标准解析器)
# =====================================================================
class MDI_Physics_Solver:
    def __init__(self, TR1=0.00797, TR2=0.03823, a1_deg=4, a2_deg=16, delta_TE=0.0063):
        self.TR1, self.TR2, self.delta_TE = TR1, TR2, delta_TE
        self.n = TR2 / TR1
        a1, a2 = math.radians(a1_deg), math.radians(a2_deg)
        self.ca1, self.ca2 = math.cos(a1), math.cos(a2)
        self.sa1, self.sa2 = math.sin(a1), math.sin(a2)

    def calculate_T2star(self, S2, eps=1e-8):
        a = torch.sum(
            torch.conj(S2[:, :, :, :-1, :, :]) * S2[:, :, :, 1:, :, :], dim=(1, 2, 3)
        )
        b = torch.sum(torch.abs(S2[:, :, :, :-1, :, :]) ** 2, dim=(1, 2, 3))
        ratio = torch.clamp(torch.abs(a / (b + eps)), min=0.01, max=0.999)
        T2star = -self.delta_TE / torch.log(ratio)
        return T2star

    def calculate_T1(self, S1, S2, eps=1e-8):
        A1 = torch.sum(
            torch.conj(S1[:, :, 1, :, :, :]) * S1[:, :, 0, :, :, :], dim=(1, 2)
        )
        B1 = torch.sum(torch.abs(S1[:, :, 1, :, :, :]) ** 2, dim=(1, 2))
        R1 = torch.abs(A1 / (B1 + eps)) * (self.sa2 / self.sa1)
        a_TR1 = self.n * (self.n + 1) * self.ca1 * self.ca2 * (R1 * self.ca1 - self.ca2)
        b_TR1 = (self.n + 1) * (
            (self.n + self.ca1) * self.ca2**2 - (self.n + self.ca2) * self.ca1**2 * R1
        ) + self.n * (R1 * self.sa1**2 * self.ca2 - self.ca1 * self.sa2**2)
        c_TR1 = self.sa2**2 * (self.n + self.ca1) - R1 * self.sa1**2 * (
            self.n + self.ca2
        )
        delta_TR1 = torch.clamp(b_TR1**2 - 4 * a_TR1 * c_TR1, min=0.0)
        k1 = (-b_TR1 - torch.sqrt(delta_TR1)) / (2 * a_TR1 + eps)
        T1_TR1 = self.TR1 / (torch.abs(k1) + eps)

        A2 = torch.sum(
            torch.conj(S2[:, :, 1, :, :, :]) * S2[:, :, 0, :, :, :], dim=(1, 2)
        )
        B2 = torch.sum(torch.abs(S2[:, :, 1, :, :, :]) ** 2, dim=(1, 2))
        R2 = torch.abs(A2 / (B2 + eps)) * (self.sa2 / self.sa1)
        a_TR2 = self.n * (self.n + 1) * self.ca1 * self.ca2 * (self.ca1 * R2 - self.ca2)
        b_TR2 = (self.n + 1) * (
            (1 + self.n * self.ca1) * self.ca2**2
            - (1 + self.n * self.ca2) * self.ca1**2 * R2
        ) + self.n * (R2 * self.sa1**2 * self.ca2 - self.ca1 * self.sa2**2)
        c_TR2 = self.sa2**2 * (1 + self.n * self.ca1) - R2 * self.sa1**2 * (
            1 + self.n * self.ca2
        )
        delta_TR2 = torch.clamp(b_TR2**2 - 4 * a_TR2 * c_TR2, min=0.0)
        k2 = (-b_TR2 - torch.sqrt(delta_TR2)) / (2 * a_TR2 + eps)
        T1_TR2 = self.TR1 / (torch.abs(k2) + eps)

        T1_Final = torch.clamp((T1_TR1 + T1_TR2) / 2, min=0.0, max=5.0)
        return T1_Final


# =====================================================================
#  2. 纯推断数据集加载器 (无 Label)
# =====================================================================
class Mapping_Inference_Dataset(Dataset):
    def __init__(self, data_dir, vol_name):
        # 请根据你的实际路径调整
        # recon_data = readcfl('/data0/zhaojiaying/DeepSNMs_MTP_checkpoints/mtp-echos-mslice/ns_iter10_9x_12echo_diluconv_nogru_cginit/vol_1_mechos_mslice_dic_12echo')
        recon_data = readcfl(
            "/data0/zhaojiaying/DeepSNMs_MTP_checkpoints/mtp-echos-mslice/ns_iter10_9x_12echo_cginit_sensecc_phaseloss_middleslice_15niter/vol_1_mechos_mslice_dic_12echo"
        )
        # recon_data = readcfl('/data0/zhaojiaying/MTP_data/img_gra_12echo')
        # recon_data = readcfl('/data0/zhaojiaying/MTP_data/perspective_data/20260614_121230_wanghongming_18393/3_mtp_caipi4x_0p68_2mm/recon_grappa')
        self.recon_data = np.transpose(recon_data, (0, 3, 4, 1, 2))
        self.nx = self.recon_data.shape[0]

    def __len__(self):
        return self.nx

    def __getitem__(self, idx):
        recon_slice = np.array(self.recon_data[idx])
        return torch.tensor(recon_slice, dtype=torch.complex64)


# =====================================================================
#  3. 主程序
# =====================================================================
if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--data_dir", type=str, default="/data0/zhaojiaying/Mappping_test_data/"
    )
    parser.add_argument("--save_dir", type=str, default="./persp_mapping_results")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    analytical_solver = MDI_Physics_Solver()
    print(" 物理求解器准备完毕！")

    test_vols = ["vol1"]  # 可扩展为多个数据卷

    for vol in test_vols:
        print(f"\n================ 开始处理 {vol} ================")
        dataset = Mapping_Inference_Dataset(args.data_dir, vol)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # -------------------------------------------------------------
        # 步骤 A：预遍历一遍，通过 Recon 计算全局最大信号，用于后续去噪 Mask
        # -------------------------------------------------------------
        global_max_signal = 0.0
        for recon_slice in loader:
            recon_echo1 = recon_slice[:, :, 0, :, :].to(device)
            current_max = (
                torch.sqrt(torch.sum(torch.abs(recon_echo1) ** 2, dim=1)).max().item()
            )
            global_max_signal = max(global_max_signal, current_max)

        print(f" 全局最大信号提取完毕: {global_max_signal:.4f}")

        # 获取维度信息并预分配内存
        nx_vol = dataset.nx
        ny_vol, nz_vol = (
            dataset.recon_data[0].shape[-2],
            dataset.recon_data[0].shape[-1],
        )

        T1_recon_all = np.zeros((nx_vol, ny_vol, nz_vol), dtype=np.float32)
        T2s_recon_all = np.zeros_like(T1_recon_all)

        # -------------------------------------------------------------
        # 步骤 B：正式计算 T1 和 T2*
        # -------------------------------------------------------------
        with torch.no_grad():
            for step, recon_im in enumerate(loader):
                recon_im = recon_im.to(device)
                nb = recon_im.size(0)

                # 计算基于 Recon 的 SOS 图像及 Mask
                recon_echo1 = recon_im[:, :, 0, :, :]
                recon_sos = torch.sqrt(torch.sum(torch.abs(recon_echo1) ** 2, dim=1))

                mask_brain = torch.zeros(
                    (nb, ny_vol, nz_vol), device=device, dtype=torch.float32
                )
                for b in range(nb):
                    slice_img = recon_sos[b]
                    if slice_img.mean().item() < 0.03 * global_max_signal:
                        continue
                    else:
                        robust_max = torch.quantile(slice_img.view(-1), 0.99).item()
                        mask_brain[b, :, :] = (slice_img > 0.12 * robust_max).float()

                # 分割回波 S1 (0-6) 和 S2 (6-12)
                recon_stacked = torch.stack(
                    [recon_im[:, :, 0:6, :, :], recon_im[:, :, 6:12, :, :]], dim=2
                )
                S1_recon = recon_stacked[:, :, :, 0:1, :, :]
                S2_recon = recon_stacked[:, :, :, 1:6, :, :]

                # 物理推断
                T1_map_pred = analytical_solver.calculate_T1(S1_recon, S2_recon)
                T2s_map_pred = analytical_solver.calculate_T2star(S2_recon)

                # 施加 Mask 约束并转换单位到毫秒 (ms)
                T1_map_pred = torch.clamp(T1_map_pred, 0.0, 5.0) * mask_brain
                t1_mask_pred = (T1_map_pred > 0.01).float()

                T2s_map_pred = (
                    torch.clamp(T2s_map_pred, 0.0, 0.2) * mask_brain * t1_mask_pred
                )

                # 存入全局 Numpy 数组
                start_idx = step * args.batch_size
                end_idx = start_idx + nb

                T1_recon_all[start_idx:end_idx] = (T1_map_pred * 1000).cpu().numpy()
                T2s_recon_all[start_idx:end_idx] = (T2s_map_pred * 1000).cpu().numpy()

        # -------------------------------------------------------------
        # 步骤 C：将结果保存为 CFL 格式
        # -------------------------------------------------------------
        t1_save_path = os.path.join(args.save_dir, f"{vol}_T1_map_1mm")
        t2s_save_path = os.path.join(args.save_dir, f"{vol}_T2s_map_1mm")

        writecfl(t1_save_path, T1_recon_all)
        writecfl(t2s_save_path, T2s_recon_all)

    print(f"\n 所有数据推断及保存完毕！总耗时: {time.time() - t0:.2f} 秒")


import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

# =====================================================================
#  0. 导入网络结构
# =====================================================================
from train_mapping import DeepMTP_Mapping_12Echo


# =====================================================================
#  1. 物理组件 (网络可微层 & 传统金标准解析器)
# =====================================================================
class Physics_Layer(torch.nn.Module):
    def __init__(self, TR1=0.00797, TR2=0.03823, a1_deg=4, a2_deg=16, delta_TE=0.0063):
        super().__init__()
        self.TR1, self.TR2, self.delta_TE = TR1, TR2, delta_TE
        self.n = TR2 / TR1
        a1, a2 = math.radians(a1_deg), math.radians(a2_deg)
        self.ca1, self.ca2 = math.cos(a1), math.cos(a2)
        self.sa1, self.sa2 = math.sin(a1), math.sin(a2)

    def forward(self, pred_deltaS, pred_R2, mask_brain):
        delta_S = torch.clamp(torch.abs(pred_deltaS), min=0.01, max=0.999)
        R_ratio = torch.abs(pred_R2) * (self.sa2 / self.sa1)

        T2star_pred = -self.delta_TE / torch.log(delta_S)

        a = (
            self.n
            * (self.n + 1)
            * self.ca1
            * self.ca2
            * (R_ratio * self.ca1 - self.ca2)
        )
        b = (self.n + 1) * (
            (self.n + self.ca1) * self.ca2**2
            - (self.n + self.ca2) * self.ca1**2 * R_ratio
        ) + self.n * (R_ratio * self.sa1**2 * self.ca2 - self.ca1 * self.sa2**2)
        c = self.sa2**2 * (self.n + self.ca1) - R_ratio * self.sa1**2 * (
            self.n + self.ca2
        )

        delta = torch.clamp(b**2 - 4 * a * c, min=0.0)
        k = (-b - torch.sqrt(delta)) / (2 * a + 1e-8)
        T1_pred = self.TR1 / (torch.abs(k) + 1e-8)

        T1_pred = torch.clamp(T1_pred, min=0.0, max=5.0) * mask_brain
        t1_mask = (T1_pred > 0.01).float()

        #  修复：解除 T1 对 T2* 的强掩膜绑定
        T2star_pred = torch.clamp(T2star_pred, min=0.0, max=0.2) * mask_brain * t1_mask

        valid_mask = T2star_pred > 0.002
        R2star_pred = torch.zeros_like(T2star_pred)
        R2star_pred[valid_mask] = 1.0 / T2star_pred[valid_mask]
        R2star_pred = (
            torch.clamp(R2star_pred, min=0.0, max=200.0) * mask_brain * t1_mask
        )

        return T1_pred, T2star_pred, R2star_pred


class MDI_Physics_Solver:
    def __init__(self, TR1=0.00797, TR2=0.03823, a1_deg=4, a2_deg=16, delta_TE=0.0063):
        self.TR1, self.TR2, self.delta_TE = TR1, TR2, delta_TE
        self.n = TR2 / TR1
        a1, a2 = math.radians(a1_deg), math.radians(a2_deg)
        self.ca1, self.ca2 = math.cos(a1), math.cos(a2)
        self.sa1, self.sa2 = math.sin(a1), math.sin(a2)

    def calculate_T2star(self, S2, eps=1e-8):
        """修复：恢复复数域最优匹配滤波，完美对齐金标准"""
        a = torch.sum(
            torch.conj(S2[:, :, :, :-1, :, :]) * S2[:, :, :, 1:, :, :], dim=(1, 2, 3)
        )
        b = torch.sum(torch.abs(S2[:, :, :, :-1, :, :]) ** 2, dim=(1, 2, 3))
        ratio = torch.clamp(torch.abs(a / (b + eps)), min=0.01, max=0.999)
        T2star = -self.delta_TE / torch.log(ratio)
        return T2star

    def calculate_T1(self, S1, S2, eps=1e-8):
        A1 = torch.sum(
            torch.conj(S1[:, :, 1, :, :, :]) * S1[:, :, 0, :, :, :], dim=(1, 2)
        )
        B1 = torch.sum(torch.abs(S1[:, :, 1, :, :, :]) ** 2, dim=(1, 2))
        R1 = torch.abs(A1 / (B1 + eps)) * (self.sa2 / self.sa1)
        a_TR1 = self.n * (self.n + 1) * self.ca1 * self.ca2 * (R1 * self.ca1 - self.ca2)
        b_TR1 = (self.n + 1) * (
            (self.n + self.ca1) * self.ca2**2 - (self.n + self.ca2) * self.ca1**2 * R1
        ) + self.n * (R1 * self.sa1**2 * self.ca2 - self.ca1 * self.sa2**2)
        c_TR1 = self.sa2**2 * (self.n + self.ca1) - R1 * self.sa1**2 * (
            self.n + self.ca2
        )
        delta_TR1 = torch.clamp(b_TR1**2 - 4 * a_TR1 * c_TR1, min=0.0)
        k1 = (-b_TR1 - torch.sqrt(delta_TR1)) / (2 * a_TR1 + eps)
        T1_TR1 = self.TR1 / (torch.abs(k1) + eps)

        A2 = torch.sum(
            torch.conj(S2[:, :, 1, :, :, :]) * S2[:, :, 0, :, :, :], dim=(1, 2)
        )
        B2 = torch.sum(torch.abs(S2[:, :, 1, :, :, :]) ** 2, dim=(1, 2))
        R2 = torch.abs(A2 / (B2 + eps)) * (self.sa2 / self.sa1)
        a_TR2 = self.n * (self.n + 1) * self.ca1 * self.ca2 * (self.ca1 * R2 - self.ca2)
        b_TR2 = (self.n + 1) * (
            (1 + self.n * self.ca1) * self.ca2**2
            - (1 + self.n * self.ca2) * self.ca1**2 * R2
        ) + self.n * (R2 * self.sa1**2 * self.ca2 - self.ca1 * self.sa2**2)
        c_TR2 = self.sa2**2 * (1 + self.n * self.ca1) - R2 * self.sa1**2 * (
            1 + self.n * self.ca2
        )
        delta_TR2 = torch.clamp(b_TR2**2 - 4 * a_TR2 * c_TR2, min=0.0)
        k2 = (-b_TR2 - torch.sqrt(delta_TR2)) / (2 * a_TR2 + eps)
        T1_TR2 = self.TR1 / (torch.abs(k2) + eps)

        T1_Final = torch.clamp((T1_TR1 + T1_TR2) / 2, min=0.0, max=5.0)
        return T1_Final

    def calculate_aT1w(self, S1, S2, eps=1e-8):
        a1 = torch.sum(
            torch.conj(S1[:, :, 0, :, :, :]) * S1[:, :, 1, :, :, :], dim=(1, 2)
        )
        b1 = torch.sum(torch.abs(S1[:, :, 0, :, :, :]) ** 2, dim=(1, 2))
        a2 = torch.sum(
            torch.conj(S2[:, :, 0, :, :, :]) * S2[:, :, 1, :, :, :], dim=(1, 2)
        )
        b2 = torch.sum(torch.abs(S2[:, :, 0, :, :, :]) ** 2, dim=(1, 2))
        S_aT1w = torch.abs(a1 + a2) / (b1 + b2 + eps)
        return S_aT1w


# =====================================================================
#  2. 数据集加载器
# =====================================================================
class Mapping_Test_Dataset(Dataset):
    def __init__(self, data_dir, vol_name):
        # self.recon_data = np.load(os.path.join(data_dir, f'{vol_name}_recon.npy'), mmap_mode='r')
        # self.label_data = np.load(os.path.join(data_dir, f'{vol_name}_label.npy'), mmap_mode='r')
        self.recon_data = np.load(
            os.path.join(data_dir, f"{vol_name}_Recon.npy"), mmap_mode="r"
        )
        self.label_data = np.load(
            os.path.join(data_dir, f"{vol_name}_Label.npy"), mmap_mode="r"
        )
        self.nx = self.recon_data.shape[0]

    def __len__(self):
        return self.nx

    def __getitem__(self, idx):
        recon_slice = np.array(self.recon_data[idx])
        label_slice = np.array(self.label_data[idx])
        return torch.tensor(recon_slice, dtype=torch.complex64), torch.tensor(
            label_slice, dtype=torch.complex64
        )


def complex_nrmse_np(recon, label):
    denom = np.sqrt(np.mean(np.abs(label) ** 2, dtype=np.float64))
    mse_val = np.mean(np.abs(recon - label) ** 2)
    return np.sqrt(mse_val) / (denom + 1e-8)


def calc_metrics(recon, label):
    data_range = label.max() if label.max() > 0 else 1.0
    psnr = peak_signal_noise_ratio(label, recon, data_range=data_range)
    ssim = structural_similarity(label, recon, data_range=data_range)
    nrmse = complex_nrmse_np(recon, label)
    return psnr, ssim, nrmse


# =====================================================================
#  3. 主程序
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument('--data_dir', type=str, default='/data0/zhaojiaying/Mapping_testing_Data_12Echo/')
    parser.add_argument(
        "--data_dir", type=str, default="/data0/zhaojiaying/Mappping_test_data/"
    )
    parser.add_argument("--save_dir", type=str, default="./deep_quantitative_results")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data0/zhaojiaying/DeepSNMs_MTP_checkpoints/Mapping_Stage2_Complex/mapping_params_100_jointloss.pkl",
    )
    # parser.add_argument('--ckpt_path', type=str, default='/data0/zhaojiaying/DeepSNMs_MTP_checkpoints/Mapping_Stage2_Complex_sensecc_midslice/mapping_params_100.pkl')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("⏳ 正在加载预训练模型...")
    mapping_net = DeepMTP_Mapping_12Echo().to(device)
    mapping_net.load_state_dict(
        torch.load(args.ckpt_path, map_location=device, weights_only=True)
    )
    mapping_net.eval()

    physics_layer = Physics_Layer().to(device)
    analytical_solver = MDI_Physics_Solver()
    print(" 网络及物理转换层准备完毕！")

    test_vols = ["vol1"]

    for vol in test_vols:
        print(f"\n================ 开始双轨误差定量评估 {vol} ================")
        dataset = Mapping_Test_Dataset(args.data_dir, vol)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        global_max_signal = 0.0
        for recon_slice, _ in loader:
            recon_echo1 = recon_slice[:, :, 0, :, :].to(device)
            global_max_signal = max(
                global_max_signal,
                torch.sqrt(torch.sum(torch.abs(recon_echo1) ** 2, dim=1)).max().item(),
            )

        nx_vol, ny_vol, nz_vol = (
            dataset.nx,
            dataset.recon_data[0].shape[-2],
            dataset.recon_data[0].shape[-1],
        )

        # 预分配
        T1_label_all = np.zeros((nx_vol, ny_vol, nz_vol), dtype=np.float32)
        T2s_label_all, R2s_label_all, aT1w_label_all = (
            np.zeros_like(T1_label_all),
            np.zeros_like(T1_label_all),
            np.zeros_like(T1_label_all),
        )

        T1_recon_all = np.zeros_like(T1_label_all)
        T2s_recon_all, R2s_recon_all, aT1w_recon_all = (
            np.zeros_like(T1_label_all),
            np.zeros_like(T1_label_all),
            np.zeros_like(T1_label_all),
        )
        mask_all = np.zeros_like(T1_label_all)

        with torch.no_grad():
            for step, (recon_im, label_im) in enumerate(loader):
                recon_im, label_im = recon_im.to(device), label_im.to(device)
                nb = recon_im.size(0)

                recon_echo1 = label_im[:, :, 0, :, :]
                recon_sos = torch.sqrt(torch.sum(torch.abs(recon_echo1) ** 2, dim=1))

                mask_dynamic = torch.zeros(
                    (nb, ny_vol, nz_vol), device=device, dtype=torch.float32
                )
                for b in range(nb):
                    slice_img = recon_sos[b]
                    if slice_img.mean().item() < 0.03 * global_max_signal:
                        mask_dynamic[b, :, :] = 0.0
                    else:
                        robust_max = torch.quantile(slice_img.view(-1), 0.99).item()
                        mask_dynamic[b, :, :] = (slice_img > 0.12 * robust_max).float()
                mask_brain = mask_dynamic

                label_stacked = torch.stack(
                    [label_im[:, :, 0:6, :, :], label_im[:, :, 6:12, :, :]], dim=2
                )
                S1_label, S2_label = (
                    label_stacked[:, :, :, 0:1, :, :],
                    label_stacked[:, :, :, 1:6, :, :],
                )

                recon_stacked = torch.stack(
                    [recon_im[:, :, 0:6, :, :], recon_im[:, :, 6:12, :, :]], dim=2
                )
                S1_recon, S2_recon = (
                    recon_stacked[:, :, :, 0:1, :, :],
                    recon_stacked[:, :, :, 1:6, :, :],
                )

                # ---- 金标准 ----
                T1_map_L = analytical_solver.calculate_T1(S1_label, S2_label)
                T2s_map_L = analytical_solver.calculate_T2star(S2_label)
                aT1w_map_L = analytical_solver.calculate_aT1w(S1_label, S2_label)

                T1_map_L = torch.clamp(T1_map_L, 0.0, 5.0) * mask_brain
                t1_mask_L = (T1_map_L > 0.01).float()
                T2s_map_L = torch.clamp(T2s_map_L, 0.0, 0.2) * mask_brain

                valid_mask_L = T2s_map_L > 0.002
                R2s_map_L = torch.zeros_like(T2s_map_L)
                R2s_map_L[valid_mask_L] = 1.0 / T2s_map_L[valid_mask_L]
                R2s_map_L = torch.clamp(R2s_map_L, 0.0, 200.0) * mask_brain * t1_mask_L
                aT1w_map_L = torch.nan_to_num(aT1w_map_L, nan=0.0) * mask_brain

                # ---- 网络推断 ----
                pred_deltaS_c, pred_R2_c = mapping_net(recon_im, mask_brain)
                T1_map_pred, T2s_map_pred, R2s_map_pred = physics_layer(
                    pred_deltaS_c, pred_R2_c, mask_brain
                )

                aT1w_map_pred = analytical_solver.calculate_aT1w(S1_recon, S2_recon)
                aT1w_map_pred = torch.nan_to_num(aT1w_map_pred, nan=0.0) * mask_brain

                # ---- 保存 ----
                start_idx = step * args.batch_size
                end_idx = start_idx + nb

                T1_label_all[start_idx:end_idx] = (T1_map_L * 1000).cpu().numpy()
                T2s_label_all[start_idx:end_idx] = (T2s_map_L * 1000).cpu().numpy()
                R2s_label_all[start_idx:end_idx] = R2s_map_L.cpu().numpy()
                aT1w_label_all[start_idx:end_idx] = aT1w_map_L.cpu().numpy()

                T1_recon_all[start_idx:end_idx] = (T1_map_pred * 1000).cpu().numpy()
                T2s_recon_all[start_idx:end_idx] = (T2s_map_pred * 1000).cpu().numpy()
                R2s_recon_all[start_idx:end_idx] = R2s_map_pred.cpu().numpy()
                aT1w_recon_all[start_idx:end_idx] = aT1w_map_pred.cpu().numpy()

                mask_all[start_idx:end_idx] = mask_brain.cpu().numpy()

        # ====================================================
        #  4. 画图保存 (3x4 包含热力误差图及指标展示)
        # ====================================================
        print(f"💾 正在计算指标并保存 {vol} 的 3x4 误差对比图...")

        mid_z = nz_vol // 2
        cur_mask = mask_all[:, :, mid_z]

        # 提取当前切片数据并施加 Mask 保证背景干净
        L_T1 = T1_label_all[:, :, mid_z] * cur_mask
        R_T1 = T1_recon_all[:, :, mid_z] * cur_mask

        L_T2s = T2s_label_all[:, :, mid_z] * cur_mask
        R_T2s = T2s_recon_all[:, :, mid_z] * cur_mask

        L_R2s = R2s_label_all[:, :, mid_z] * cur_mask
        R_R2s = R2s_recon_all[:, :, mid_z] * cur_mask

        L_aT1w = aT1w_label_all[:, :, mid_z] * cur_mask
        R_aT1w = aT1w_recon_all[:, :, mid_z] * cur_mask

        # ====================================================
        #  计算当前切片的定量指标
        # ====================================================
        p_t1, s_t1, n_t1 = calc_metrics(R_T1, L_T1)
        p_t2, s_t2, n_t2 = calc_metrics(R_T2s, L_T2s)
        p_r2, s_r2, n_r2 = calc_metrics(R_R2s, L_R2s)
        p_at, s_at, n_at = calc_metrics(R_aT1w, L_aT1w)

        text_T1 = f"PSNR: {p_t1:.1f} | SSIM: {s_t1:.3f} | NRMSE: {n_t1:.3f}"
        text_T2s = f"PSNR: {p_t2:.1f} | SSIM: {s_t2:.3f} | NRMSE: {n_t2:.3f}"
        text_R2s = f"PSNR: {p_r2:.1f} | SSIM: {s_r2:.3f} | NRMSE: {n_r2:.3f}"
        text_aT1w = f"PSNR: {p_at:.1f} | SSIM: {s_at:.3f} | NRMSE: {n_at:.3f}"

        # 1. 计算当前切片的绝对误差
        err_T1 = np.abs(R_T1 - L_T1)
        err_T2s = np.abs(R_T2s - L_T2s)
        err_R2s = np.abs(R_R2s - L_R2s)
        err_aT1w = np.abs(R_aT1w - L_aT1w)

        # 2. 原图画图窗宽
        v_T1, v_T2s, v_R2s = 2500, 200, 100
        v_aT1w = max(
            np.percentile(L_aT1w[L_aT1w > 0], 99) if L_aT1w.max() > 0 else 1.2,
            np.percentile(R_aT1w[R_aT1w > 0], 99) if R_aT1w.max() > 0 else 1.2,
        )

        # 3. 误差图的上限 (通常设为原图的 10%~20%，以清晰暴露微小细节差异)
        e_T1, e_T2s, e_R2s, e_aT1w = (
            v_T1 * 0.15,
            v_T2s * 0.15,
            v_R2s * 0.20,
            v_aT1w * 0.15,
        )

        fig, axs = plt.subplots(3, 4, figsize=(24, 18), facecolor="black")

        #  升级绘图函数，使其支持指标绘制
        def plot_sub(ax, data, title, vmax, cmap="gray", metrics_text=None):
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(title, color="white", fontsize=18, pad=12)
            ax.axis("off")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")

            # 如果传入了指标文本，将其绘制在图像底部居中，带有半透明黑色背景框
            if metrics_text:
                ax.text(
                    0.5,
                    0.03,
                    metrics_text,
                    color="gold",
                    fontsize=14,
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                )

        # 第一排：金标准 Label (无指标)
        plot_sub(axs[0, 0], L_T1, f"Label T1 (ms) - Z:{mid_z}", v_T1)
        plot_sub(axs[0, 1], L_T2s, f"Label T2* (ms)", v_T2s)
        plot_sub(axs[0, 2], L_R2s, f"Label R2* (Hz)", v_R2s)
        plot_sub(axs[0, 3], L_aT1w, f"Label aT1w", v_aT1w)

        #  第二排：深度推断 Recon (填入刚刚计算的指标！)
        plot_sub(axs[1, 0], R_T1, f"Deep Pred T1 (ms)", v_T1, metrics_text=text_T1)
        plot_sub(axs[1, 1], R_T2s, f"Deep Pred T2* (ms)", v_T2s, metrics_text=text_T2s)
        plot_sub(axs[1, 2], R_R2s, f"Deep Pred R2* (Hz)", v_R2s, metrics_text=text_R2s)
        plot_sub(axs[1, 3], R_aT1w, f"Deep Pred aT1w", v_aT1w, metrics_text=text_aT1w)

        # 第三排：绝对误差 Error (使用热力图 cmap='hot' 高亮差异)
        plot_sub(axs[2, 0], err_T1, f"|Error| T1 (ms)", e_T1, cmap="jet")
        plot_sub(axs[2, 1], err_T2s, f"|Error| T2* (ms)", e_T2s, cmap="jet")
        plot_sub(axs[2, 2], err_R2s, f"|Error| R2* (Hz)", e_R2s, cmap="jet")
        plot_sub(axs[2, 3], err_aT1w, f"|Error| aT1w", e_aT1w, cmap="jet")

        plt.tight_layout()
        save_path_single = os.path.join(
            args.save_dir, f"{vol}_ErrorMap_T1_T2s_R2s_aT1w_Z{mid_z}.png"
        )
        plt.savefig(
            save_path_single,
            dpi=250,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="black",
        )
        plt.close()

        print(f" 带有指标的误差对比图已保存至: {save_path_single}")

    print("\n 三重对照版评估脚本执行完毕！")
