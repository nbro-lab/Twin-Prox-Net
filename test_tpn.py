import argparse
import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import (
    MTP_dataset_echos_mslice,
    MTP_perspective_dataset,
    collate_fn_keep_12echo,
)

# Import the model exactly as used in your training script
from model_cginit import Deep_snms_echos_mslice, init_net
from scipy.io import loadmat

# Evaluation metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ssim import SSIM
from torch.utils.data import ConcatDataset, DataLoader
from utils import c2r, ifft2c, sos, writecfl


# ==========================================
# Metrics Calculation (Numpy)
# ==========================================
def complex_nrmse_np(recon, label):
    denom = np.sqrt(np.mean(np.abs(label) ** 2, dtype=np.float64))
    mse_val = np.mean(np.abs(recon - label) ** 2)
    return np.sqrt(mse_val) / (denom + 1e-8)


# ==========================================
# Visualization Function
# ==========================================
def visualize_echoes_slice(recon_slice, label_slice, save_path, echo_indices):
    """
    Visualizes specific echo slices including Recon, Label, and Error.
    """
    nx, ny, ne = recon_slice.shape
    error_slice = np.abs(np.abs(recon_slice) - np.abs(label_slice))

    fig, axs = plt.subplots(3, ne, figsize=(ne * 3, 9))

    for i in range(ne):
        cur_recon = np.abs(recon_slice[:, :, i])
        cur_label = np.abs(label_slice[:, :, i])
        cur_error = np.abs(error_slice[:, :, i])

        data_range = cur_label.max()
        cur_psnr = peak_signal_noise_ratio(cur_label, cur_recon, data_range=data_range)
        cur_ssim = structural_similarity(cur_label, cur_recon, data_range=data_range)
        cur_nrmse = complex_nrmse_np(cur_recon, cur_label)

        # 1. Recon
        axs[0, i].imshow(cur_recon, cmap="gray", vmin=0, vmax=data_range * 0.8)
        axs[0, i].axis("off")
        axs[0, i].set_title(f"Echo {echo_indices[i]}", fontsize=16, pad=8)

        metrics_text = (
            f"PSNR: {cur_psnr:.2f}\nSSIM: {cur_ssim:.3f}\nNRMSE: {cur_nrmse:.3f}"
        )
        axs[0, i].text(
            0.95,
            0.95,
            metrics_text,
            ha="right",
            va="top",
            transform=axs[0, i].transAxes,
            color="orange",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

        # 2. Label
        axs[1, i].imshow(cur_label, cmap="gray", vmin=0, vmax=data_range * 0.8)
        axs[1, i].axis("off")

        # 3. Error
        axs[2, i].imshow(cur_error, cmap="jet", vmin=0, vmax=data_range * 0.2)
        axs[2, i].axis("off")

        if i == 0:
            axs[0, i].text(
                -0.05,
                0.5,
                "Recon",
                va="center",
                ha="right",
                transform=axs[0, i].transAxes,
                fontsize=14,
                rotation=90,
            )
            axs[1, i].text(
                -0.05,
                0.5,
                "Label",
                va="center",
                ha="right",
                transform=axs[1, i].transAxes,
                fontsize=14,
                rotation=90,
            )
            axs[2, i].text(
                -0.05,
                0.5,
                "Error",
                va="center",
                ha="right",
                transform=axs[2, i].transAxes,
                fontsize=14,
                rotation=90,
            )

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f" Saved visualization to: {save_path}")
    plt.close()


def get_mask(mask_name):
    mask_dir = "./mask"
    data = loadmat(os.path.join(mask_dir, mask_name + ".mat"))
    mask = data["mask"]
    mask = torch.tensor(mask, dtype=torch.complex64)
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", metavar="int", nargs=1, default=["1"], help="batch size"
    )
    parser.add_argument(
        "--niter",
        metavar="int",
        nargs=1,
        default=["15"],
        help="number of network iterations",
    )
    parser.add_argument(
        "--nEcho", metavar="int", nargs=1, default=["12"], help="number of echos"
    )
    parser.add_argument(
        "--nSlice", metavar="int", nargs=1, default=["7"], help="number of slice"
    )
    parser.add_argument(
        "--net", metavar="str", nargs=1, default=["mtp-echos-mslice"], help="network"
    )
    parser.add_argument(
        "--data",
        metavar="str",
        nargs=1,
        default=["MTP_datesets_c12"],
        help="dataset name",
    )
    parser.add_argument(
        "--acc", metavar="str", nargs=1, default=["9x"], help="accelerate rate"
    )
    parser.add_argument(
        "--mask",
        metavar="str",
        nargs=1,
        default=["caipi_9x"],
        help="under-sampling pattern",
    )
    parser.add_argument("--manualSeed", type=int, help="manual seed")

    args = parser.parse_args()

    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name = args.data[0]
    model_name = args.net[0]
    niter = int(args.niter[0])
    nEcho = int(args.nEcho[0])
    acc = args.acc[0]
    mask_name = args.mask[0]

    print("-----------load data----------")
    test_datasets = []
    # vol_name = ['vol1']
    # for j in range(len(vol_name)):
    #     vol = vol_name[j]
    #     # 注意：这里 stride 是多少都不重要了，因为我们会直接读取其内部完整的属性
    #     test_dataset = MTP_dataset_echos_mslice('/data0/zhaojiaying/MTP_datasets', 'test', vol, nEcho, acc, nslice=7, stride=7)
    #     test_datasets.append(test_dataset)
    vol_name = ["caipi_c32_9x"]
    for j in range(len(vol_name)):
        vol = vol_name[j]
        test_dataset = MTP_dataset_echos_mslice(
            "/data0/zhaojiaying/MTP_data/perspective_data",
            "20260614_121230_wanghongming_18393",
            vol,
            nEcho,
            acc,
            nslice=7,
            stride=7,
        )
        # test_dataset = MTP_dataset_echos_mslice('/data0/zhaojiaying/MTP_data/perspective_data','presp_2_sxx',  vol, nEcho, acc, nslice=7,
        #                                            stride=7)
        test_datasets.append(test_dataset)

    base_dir = "/data0/zhaojiaying/DeepSNMs_MTP_checkpoints"
    model_save_path = os.path.join(base_dir, model_name)
    ckpt_path = os.path.join(
        model_save_path,
        "ns_iter10_9x_12echo_cginit_sensecc_phaseloss_middleslice_15niter",
    )

    DSNMs = Deep_snms_echos_mslice(in_channel=2, niter=niter).to(device)

    # 请确保这里的模型权重名字是你训练出的最好的那一个！
    last_ckpt_path = ckpt_path
    assert os.path.exists(last_ckpt_path), f"path '{last_ckpt_path}' does not exist."
    model_weights_path = os.path.join(last_ckpt_path, "net_params_133.pkl")
    DSNMs.load_state_dict(torch.load(model_weights_path, map_location=device))
    print(f"Successfully Load Model Parameters from {model_weights_path}")

    mask = get_mask(mask_name=mask_name).to(device)

    print("----------- Test network----------")
    t0_total = time.time()

    for i, test_dataset in enumerate(test_datasets):
        print(f"------------------------vol_{i+1}--------------------------")
        nx, nc, ne, ny, nz = (
            test_dataset.nx,
            test_dataset.nc,
            test_dataset.ne,
            test_dataset.ny,
            test_dataset.nz,
        )

        # =========================================================================
        #  核心革新：放弃 DataLoader！直接把完整的 3D 卷提取到 CPU 内存
        # =========================================================================
        print("Extracting full volume into memory...")
        ksp_vol = torch.tensor(test_dataset.ksp, dtype=torch.complex64)
        csm_vol = torch.tensor(test_dataset.csm, dtype=torch.complex64)

        # 严格执行全局能量归一化
        sum_squares = torch.sum(torch.abs(csm_vol) ** 2, dim=1, keepdim=True)
        csm_vol = csm_vol / (torch.sqrt(sum_squares) + 1e-8)

        # 提取字典掩码并恢复到图像域
        Null_2d_vol = []
        for i_idx in range(nx):
            null_2d = test_dataset.Null_2d[i_idx]
            null_2d = torch.tensor(null_2d, dtype=torch.complex64)
            Null_2d_vol.append(test_dataset.Null_to_image(null_2d))
        Null_2d_vol = torch.stack(Null_2d_vol, dim=0)

        #  为了让切片完全连贯没有亮度差异，这里计算一个全局的 Scale Factor
        print("Computing Global Normalization Scale Factor...")
        label_coils_full = ifft2c(ksp_vol)
        image_full = sos(label_coils_full, dim=1)  # dim 1 is coil
        global_scale_factor = torch.max(image_full)

        # =========================================================================
        #  重叠推理魔法：首尾 Pad 2 层
        # =========================================================================
        print("Padding volume for overlapping sliding window...")

        pad_front = 2
        pad_back = 5  # 尾部多垫几层，保证最后一次滑窗绝对能切出 7 层，防止网络尺寸越界

        # 取出第一层和最后一层，强行复制扩展
        ksp_padded = torch.cat(
            [
                ksp_vol[0:1].expand(pad_front, -1, -1, -1, -1),
                ksp_vol,
                ksp_vol[-1:].expand(pad_back, -1, -1, -1, -1),
            ],
            dim=0,
        )

        csm_padded = torch.cat(
            [
                csm_vol[0:1].expand(pad_front, -1, -1, -1),
                csm_vol,
                csm_vol[-1:].expand(pad_back, -1, -1, -1),
            ],
            dim=0,
        )

        Null_padded = torch.cat(
            [
                Null_2d_vol[0:1].expand(pad_front, -1, -1, -1, -1),
                Null_2d_vol,
                Null_2d_vol[-1:].expand(pad_back, -1, -1, -1, -1),
            ],
            dim=0,
        )

        # 顺手把全局 scale factor 也推到 GPU，防止某些 PyTorch 版本在张量除法时跨设备报错
        global_scale_factor = global_scale_factor.to(device)

        # 准备一个空盘子，用来装所有推导出来的完美的中心层
        recon_coils_full = torch.zeros((nx, nc, ne, ny, nz), dtype=torch.complex64)
        recon_cginit = torch.zeros((nx, nc, ne, ny, nz), dtype=torch.complex64)
        DSNMs.eval()
        t0 = time.time()

        print(" Starting overlapping inference (7-in, 3-out)...")
        with torch.no_grad():
            # 以 3 层为步长，滑过整个大脑
            for start in range(0, nx, 3):
                # 1. 挖出 7 层送给网络
                ksp_chunk = (
                    ksp_padded[start : start + 7].unsqueeze(0).to(device)
                )  # 增加 Batch 维度 [1, 7, nc, ...]
                csm_chunk = csm_padded[start : start + 7].unsqueeze(0).to(device)
                Null_chunk = Null_padded[start : start + 7].unsqueeze(0).to(device)

                # 2. 全局归一化与掩码应用
                k0_chunk = ksp_chunk / global_scale_factor
                k0_u = k0_chunk * mask
                x0 = torch.zeros_like(k0_u)

                # 3. 前向推理！
                recon_ksp, recon_coils_chunk, x_cginit = DSNMs(
                    k0_u, Null_chunk, csm_chunk, mask, x0
                )

                # 4. 把幅度缩放回原样
                recon_coils_chunk = recon_coils_chunk * global_scale_factor
                x_cginit = x_cginit * global_scale_factor
                # 5. 🪓 最核心的一步：切掉炮灰，只留中间完美的 3 层 (索引 2, 3, 4)
                # 处理最尾部的情况：如果大脑总层数不能被 3 整除，最后一次可能只需要 1 到 2 层
                valid_slices = min(3, nx - start)
                recon_mid = recon_coils_chunk[0, 2 : 2 + valid_slices, ...].cpu()
                cginit_mid = x_cginit[0, 2 : 2 + valid_slices, ...].cpu()
                # 6. 把完美的 3 层填入大数组对应位置！
                recon_coils_full[start : start + valid_slices] = recon_mid
                recon_cginit[start : start + valid_slices] = cginit_mid
                print(
                    f"  Processed perfectly middle slices: {start} to {start + valid_slices - 1} / {nx}"
                )

        print("Time for Model Inference:", time.time() - t0)

        # =========================================================================
        #  将数据转换为原来的格式，平滑交接给你原本的画图/存盘代码
        # =========================================================================
        recon_images = recon_coils_full.numpy()
        label_images = recon_cginit.numpy()
        # label_images = label_coils_full.numpy()
        csm_images = csm_vol.numpy()

        global_loss = complex_nrmse_np(recon_images, label_images)
        print(f"\n================ Test Summary ================")
        print(f"Global Average Test NRMSE: {global_loss:.4f}")
        print("==============================================\n")

        # Transpose to (nx, ny, nz, nc, ne)
        recon_images = np.transpose(recon_images, (0, 3, 4, 1, 2))
        label_images = np.transpose(label_images, (0, 3, 4, 1, 2))
        csm_images = np.transpose(csm_images, (0, 2, 3, 1))

        csm_expanded = np.expand_dims(csm_images, axis=-1)
        csm_mag_sq = np.sum(np.abs(csm_expanded) ** 2, axis=3) + 1e-8

        # 完整的 Roemer 公式
        recon_sense_cplx = (
            np.sum(recon_images * np.conj(csm_expanded), axis=3) / csm_mag_sq
        )
        label_sense_cplx = (
            np.sum(label_images * np.conj(csm_expanded), axis=3) / csm_mag_sq
        )

        print("csm_images shape:", csm_images.shape)
        print("csm_expand shape:", csm_expanded.shape)
        print("recon_sense_cplx shape:", recon_sense_cplx.shape)

        recon_sos = np.abs(recon_sense_cplx)
        label_sos = np.abs(label_sense_cplx)
        # print("Calculating SOS...")
        # recon_sos = np.sqrt(np.sum(np.abs(recon_images)**2, axis=3)) # (nx, ny, nz, ne)
        # label_sos = np.sqrt(np.sum(np.abs(label_images)**2, axis=3)) # (nx, ny, nz, ne)

        # Calculate intermediate slices
        z_start, z_end = 20, 64
        print(f"\n====== Calculating Avg Metrics for Z: {z_start} -> {z_end-1} ======")
        for e in range(nEcho):
            nrmse_list = []
            psnr_list = []
            ssim_list = []

            for z in range(z_start, z_end):
                cur_recon = recon_sos[:, :, z, e]
                cur_label = label_sos[:, :, z, e]

                data_range = cur_label.max()
                cur_nrmse = complex_nrmse_np(cur_recon, cur_label)
                cur_psnr = peak_signal_noise_ratio(
                    cur_label, cur_recon, data_range=data_range
                )
                cur_ssim = structural_similarity(
                    cur_label, cur_recon, data_range=data_range
                )

                nrmse_list.append(cur_nrmse)
                psnr_list.append(cur_psnr)
                ssim_list.append(cur_ssim)

            avg_nrmse = np.mean(nrmse_list)
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            print(
                f"  Echo {e+1:02d}: Avg NRMSE = {avg_nrmse:.4f} | Avg PSNR = {avg_psnr:.2f} | Avg SSIM = {avg_ssim:.4f}"
            )
        print("=================================================================\n")

        # Visualization for slice 44
        z_slice_idx = 44
        recon_slice = recon_sos[:, :, z_slice_idx, :]
        label_slice = label_sos[:, :, z_slice_idx, :]

        images_dir = os.path.join(ckpt_path, "eval_images")
        os.makedirs(images_dir, exist_ok=True)

        # Save Echo 1-6
        save_path_fa4 = os.path.join(images_dir, f"vol_{i+1}_echoes_01_06.png")
        visualize_echoes_slice(
            recon_slice[:, :, :6],
            label_slice[:, :, :6],
            save_path_fa4,
            list(range(1, 7)),
        )

        # Save Echo 7-12
        if nEcho >= 12:
            save_path_fa16 = os.path.join(images_dir, f"vol_{i+1}_echoes_07_12.png")
            visualize_echoes_slice(
                recon_slice[:, :, 6:12],
                label_slice[:, :, 6:12],
                save_path_fa16,
                list(range(7, 13)),
            )

        print("Time for single Volume:", time.time() - t0)

        # 1. Save multi-coil data (5D)
        cflname = "vol_" + str(i + 1) + "_mechos_mslice_dic_12echo"
        save_path = ckpt_path + "/" + cflname
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writecfl(save_path, recon_images)
        print(" Multi-coil raw data saved to:", save_path)

        cflname = "vol_" + str(i + 1) + "_mechos_mslice_dic_12echo_label"
        save_path = ckpt_path + "/" + cflname
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writecfl(save_path, label_images)
        print(" Multi-coil raw data saved to:", save_path)

        # 2. Save SOS data (4D)
        cflname_sos = "vol_" + str(i + 1) + "_mechos_mslice_dic_12echo_cc"
        save_path_sos = ckpt_path + "/" + cflname_sos
        writecfl(save_path_sos, recon_sense_cplx.astype(np.complex64))
        print(" SOS lightweight data saved to:", save_path_sos)

        cflname_sos = "vol_" + str(i + 1) + "_mechos_mslice_dic_12echo_cc_label"
        save_path_sos = ckpt_path + "/" + cflname_sos
        writecfl(save_path_sos, label_sense_cplx.astype(np.complex64))
        print(" SOS lightweight data saved to:", save_path_sos)

        cflname_sos = "vol_" + str(i + 1) + "_csm"
        save_path_sos = ckpt_path + "/" + cflname_sos
        writecfl(save_path_sos, csm_mag_sq.astype(np.complex64))
        print(" SOS lightweight data saved to:", save_path_sos)

    print("Total time is ", time.time() - t0_total)
