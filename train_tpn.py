import argparse
import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import MTP_dataset_echos_mslice, collate_fn_keep_12echo

# from model_dic_12echo import  init_net, Deep_snms_echos_mslice
from model_cginit_16 import Deep_snms_echos_mslice, init_net
from scipy.io import loadmat
from ssim import SSIM
from torch.utils.data import ConcatDataset, DataLoader
from utils import c2r, ifft2c, sos


# Loss function
def mse(recon, label):
    if recon.dtype == torch.complex64:
        residual_cplx = recon - label
        # residual = tf.stack([tf.math.real(residual_cplx), tf.math.imag(residual_cplx)], axis=-1)
        residual = c2r(residual_cplx)
        mse = torch.mean(residual**2)
        loss = mse
    else:
        residual = recon - label
        mse = torch.mean(residual**2)
        loss = mse

    return loss


def complex_nrmse(recon, label):
    # l2_norm = torch.sqrt(torch.mean(torch.pow(label, 2)))
    l2_norm = torch.sqrt(torch.mean(c2r(label) ** 2))
    mse_ = torch.sqrt(mse(recon, label))
    nrmse = mse_ / l2_norm

    return nrmse


def Dh(x):
    # horizontal gradient operator
    Dh = x[:, :, :, :-1] - x[:, :, :, 1:]
    Dh = torch.cat(
        (Dh, torch.zeros((Dh.shape[0], Dh.shape[1], Dh.shape[2], 1), device=x.device)),
        dim=-1,
    )
    return Dh


def Dv(x):
    # vertical gradient operator
    Dv = x[:, :, :-1, :] - x[:, :, 1:, :]
    Dv = torch.cat(
        (Dv, torch.zeros((Dv.shape[0], Dv.shape[1], 1, Dv.shape[3]), device=x.device)),
        dim=-2,
    )
    return Dv


def JointTV_loss(x):
    # x: [nb nx ny nc]
    h_grad = Dh(x)
    v_grad = Dv(x)
    jtv = torch.sum(torch.sqrt(torch.sum(h_grad**2 + v_grad**2, dim=1)), dim=[1, 2])
    jtv = torch.mean(torch.abs(jtv), dim=0)
    return jtv


def get_mask(mask_name):
    mask_dir = "./mask"
    data = loadmat(os.path.join(mask_dir, mask_name + ".mat"))
    mask = data["mask"]
    mask = torch.tensor(mask, dtype=torch.complex64)

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epoch", metavar="int", nargs=1, default=["200"], help="number of epochs"
    )
    parser.add_argument(
        "--batch_size", metavar="int", nargs=1, default=["2"], help="batch size"
    )
    parser.add_argument(
        "--learning_rate",
        metavar="float",
        nargs=1,
        default=["0.0005"],
        help="initial learning rate",
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
        "--net", metavar="str", nargs=1, default=["mtp-echos-mslice"], help="network"
    )
    parser.add_argument(
        "--data",
        metavar="str",
        nargs=1,
        default=["MTP_dateset_c12"],
        help="dataset name",
    )
    parser.add_argument(
        "--acc", metavar="str", nargs=1, default=["9x"], help="accelerate rate"
    )
    parser.add_argument(
        "--mask",
        metavar="str",
        nargs=1,
        default=["caipi_16x_cen30"],
        help="under-sampling pattern",
    )
    parser.add_argument(
        "--pre-training",
        metavar="bool",
        nargs=1,
        default=["False"],
        help="pre-training or not",
    )
    parser.add_argument("--manualSeed", type=int, help="manual seed")

    args = parser.parse_args()

    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
        # args.manualSeed = 3407
    print("Random seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get arguments
    dataset_name = args.data[0]
    model_name = args.net[0]

    epoches = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    learning_rate = float(args.learning_rate[0])
    niter = int(args.niter[0])
    nEcho = int(args.nEcho[0])
    acc = args.acc[0]
    mask_name = args.mask[0]

    # Datasets
    print("-----------load data----------")
    train_datasets = []
    vol_name = [
        "vol1",
        "vol2",
        "vol3",
        "vol4",
        "vol5",
        "vol6",
        "vol7",
        "vol8",
        "vol9",
        "vol10",
        "vol11",
        "vol12",
        "vol13",
        "vol14",
        "vol15",
        "vol16",
        "vol17",
        "vol18",
        "vol19",
        "vol20",
        "vol21",
        "vol22",
        "vol23",
        "vol24",
    ]
    # vol_name = ['vol1', 'vol2', 'vol3', 'vol4', 'vol5', 'vol6', 'vol7', 'vol8', 'vol9',
    #            'vol10', 'vol11', 'vol12', 'vol13', 'vol14', 'vol15']
    # vol_name = ['vol1']

    for j in range(len(vol_name)):
        vol = vol_name[j]
        train_dataset = MTP_dataset_echos_mslice(
            "/data0/zhaojiaying/MTP_datasets_total/",
            "train",
            vol,
            nEcho,
            acc,
            nslice=6,
            stride=1,
        )
        train_datasets.append(train_dataset)

    trainset = ConcatDataset(train_datasets)
    num = int(trainset.__len__())
    print("The length of train dataset is:", num)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_keep_12echo,
    )

    # log dir
    base_dir = "/data0/zhaojiaying/DeepSNMs_MTP_checkpoints"
    model_save_path = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    ckpt_path = os.path.join(
        model_save_path, "ns_iter10_9x_12echo_diluconv_nogru_cginit_capi_16x"
    )
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    # creat network
    DSNMs = Deep_snms_echos_mslice(in_channel=2, niter=niter).to(device)

    param_num = np.sum([np.prod(v.nelement()) for v in DSNMs.parameters()])
    print("number of params:", param_num)

    print("-----------------------training---------------------------")

    params = DSNMs.parameters()
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, DSNMs.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)

    # paramss = list(DSNMs.parameters())
    #
    # for i, param in enumerate(paramss):
    #     print("Param %d: %s\n%s\n%s" % (i, param.size(), param.device, param))

    # get mask
    mask = get_mask(mask_name=mask_name)
    mask = mask.to(device)

    # train
    load_path = os.path.join(ckpt_path, "net_params_118.pkl")
    DSNMs.load_state_dict(torch.load(load_path))
    print(f"Successfully loaded checkpoint: {load_path}")

    SSIM_metric = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
    print("----------- Train network----------")
    each_epoch_steps = int(num / batch_size)
    total_loss = []
    loss = 0
    SSIM = SSIM(data_range=1, size_average=True, channel=nEcho)
    results_file = "./log/results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    for epoch in range(118, epoches):
        DSNMs.train()
        train_loss_epoch = []

        for step, sample in enumerate(train_loader):
            t0 = time.time()
            loss_iter = 0

            ksp = sample["ksp"].to(device)
            Null_2d = sample["Null_2d"].to(device)
            csm = sample["csm"].to(device)
            nb, ns, nc, ne, ny, nz = ksp.size()

            label = ifft2c(ksp)  # [nb ns nc ne nx ny]
            image = sos(label, dim=2)  # [nb ns ne nx ny]
            scale_factor, _ = torch.max(image.view(batch_size, -1), dim=1)

            k0 = torch.div(ksp, scale_factor.view(batch_size, 1, 1, 1, 1, 1))

            # 2. 删除了关于 ku (ksp_init) 的归一化代码

            label_coils = ifft2c(k0)  # [nb ns nc ne nx ny]
            label = sos(label_coils, dim=2)  # [nb ns ne nx ny]

            k0_u = k0 * mask

            # 3. 直接创建一个全 0 的张量作为 x0 传给模型，模型内部会自动拿它去启动 CG
            x0 = torch.zeros_like(k0_u)

            recon_ksp, recon_coils = DSNMs(k0_u, Null_2d, csm, mask, x0)
            recon = sos(recon_coils, dim=2)  # [nb ns ne nx ny]

            # recon = recon * scale_factor.view(batch_size, 1, 1, 1)

            optim.zero_grad()
            loss_mse = mse(recon, label)
            # loss_mae_image = F.l1_loss(recon, label)
            loss_mae = F.l1_loss(recon_ksp, k0)
            # loss_ssim = SSIM(recon.reshape(nb*ns, ne, ny, nz),label.reshape(nb*ns, ne, ny, nz))
            # loss1 = loss_mae

            recon_coils_ = recon_coils.reshape(nb * ns, nc, ne, ny, nz)
            jtv_coils = JointTV_loss(
                recon_coils_.permute(0, 2, 1, 3, 4).reshape(nb * ns * ne, nc, ny, nz)
            )
            jtv_echos = JointTV_loss(recon_coils_.reshape(nb * ns * nc, ne, ny, nz))
            loss2 = 1e-5 * jtv_coils + 2e-5 * jtv_echos

            echo_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            loss_img_mae = 0
            loss_ksp_mae = 0
            weight_sum = sum(echo_weights)
            loss_ssim_val = 0  #  新增 SSIM

            for e in range(nEcho):
                w = echo_weights[e] / weight_sum

                # A. 图像域 MAE
                loss_img_mae += w * F.l1_loss(recon[:, :, e, ...], label[:, :, e, ...])

                # B.  K 空间高频惩罚 (核心修改)
                # 加上 log1p 后，原本被掩盖的高频噪声误差会被成百倍地放大！
                ksp_recon_log = torch.log1p(torch.abs(recon_ksp[:, :, :, e, ...]))
                ksp_target_log = torch.log1p(torch.abs(k0[:, :, :, e, ...]))
                loss_ksp_mae += w * F.l1_loss(ksp_recon_log, ksp_target_log)

                # C.  图像域 SSIM (核心修改)
                # 注意：假设你的图片大小是 ny, nz，unsqueeze(1) 将其变为 [batch, 1, ny, nz]
                recon_e_ssim = recon[:, :, e, ...].reshape(nb * ns, 1, ny, nz)
                label_e_ssim = label[:, :, e, ...].reshape(nb * ns, 1, ny, nz)

                # 现在 SSIM 算子会将其视为 (nb*ns) 张独立的 2D 图像进行完美并行计算
                loss_ssim_val += w * (1.0 - SSIM_metric(recon_e_ssim, label_e_ssim))

            # 2. TV Loss 保持不变 (它现在的贡献是 0.0024，刚刚好)
            loss_tv = 1e-5 * jtv_coils + 2e-5 * jtv_echos

            # 3.  极度舒适的微调重组
            # 因为 loss_ssim_val 大概是 0.15 左右，而我们希望它的贡献和 MAE(0.004) 差不多
            # 所以给 SSIM 乘上 0.05 的权重，使其贡献达到约 0.0075

            # 使用对数后的 K 空间 Loss 数值会变大，所以把权重降为 0.1
            loss = (
                1.0 * loss_img_mae
                + 0.1 * loss_ksp_mae
                + 0.05 * loss_ssim_val
                + 0.1 * loss_tv
            )

            # # 2. 对每个回波单独计算 Loss 并加权累加
            # for e in range(nEcho):
            #     # 图像域 MAE (强调视觉结构和对比度)
            #     img_mae_e = F.l1_loss(recon[:, :, e, ...], label[:, :, e, ...])
            #     loss_img_mae += echo_weights[e] * img_mae_e

            #     # k空间 MAE (强调物理数据保真度)
            #     ksp_mae_e = F.l1_loss(recon_ksp[:, :, :, e, ...], k0[:, :, :, e, ...])
            #     loss_ksp_mae += echo_weights[e] * ksp_mae_e

            # # 取平均
            # loss_img_mae = loss_img_mae / nEcho
            # loss_ksp_mae = loss_ksp_mae / nEcho

            # # 3. 组合最终的 Loss
            # # 这里给了图像域稍大一点的比重，同时保留了 k 空间和之前的 TV loss
            # loss = 1.0 * loss_img_mae + 0.5 * loss_ksp_mae + 0.1 * loss2
            # loss = 1.0 * loss_img_mae  + 0.1 * loss2

            # loss =  loss1 + 0.1 * loss2
            train_loss_epoch.append(loss.item())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(DSNMs.parameters(), max_norm=0.5)

            optim.step()
            if (step + 1) % 20 == 0:
                print(
                    "Epoch",
                    epoch + 1,
                    "/",
                    epoches,
                    "Step",
                    step + 1,
                    "/",
                    each_epoch_steps,
                    "loss =",
                    loss.item(),
                    "loss_mae = ",
                    loss_mae.item(),
                    "MSE_image = ",
                    loss_mse.item(),
                    "jtv_coils = ",
                    jtv_coils.item(),
                    "jtv_echos = ",
                    jtv_echos.item(),
                    "time",
                    time.time() - t0,
                )

        scheduler.step()
        avgLoss = np.mean(train_loss_epoch)
        print("epoch", epoch + 1, "trnLoss:", avgLoss)

        total_loss.append(avgLoss)
        lr = optim.param_groups[0]["lr"]
        with open(results_file, "a") as f:
            train_info = (
                f"[epoch: {epoch}]\n"
                f"train_loss: {avgLoss:.4f}\n"
                f"learning rate: {lr:.4f}\n"
            )
            f.write(train_info + "\n\n")

        # save model
        if (epoch + 1) % 1 == 0:
            torch.save(
                DSNMs.state_dict(), "%s/net_params_%d.pkl" % (ckpt_path, epoch + 1)
            )  # save only the parameters
            # show recon picture
            # image = recon[1].squeeze(dim=0).cpu().detach().numpy()
            # plt.imshow(image, cmap='gray')
            # plt.show()

    # draw loss curve
    epochs = range(len(total_loss))
    plt.plot(epochs, total_loss, "b")
    plt.title("loss")
    # plt.figure()
    plt.show()
