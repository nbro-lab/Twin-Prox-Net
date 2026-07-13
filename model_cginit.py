import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mdi_function import calculate_delta_S_echo, reconstruct_v
from torch.nn import init
from torch.utils.checkpoint import checkpoint
from utils import c2r, fft1c, fft2c, ifft1c, ifft2c, r2c, readcfl, writecfl


def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if gpu_ids:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ConvGRUCell(nn.Module):
    """轻量级卷积 GRU 单元"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding
        )
        self.conv_can = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv_gates(combined)
        r, z = torch.split(gates, gates.size(1) // 2, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_can(combined_reset))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class EchoConvGRU(nn.Module):
    """专门处理多回波序列的 ConvGRU"""

    def __init__(self, input_dim=2, hidden_dim=2):
        super(EchoConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvGRUCell(input_dim, hidden_dim)

    def forward(self, x_seq):
        B, C, E, Y, Z = x_seq.size()
        h = torch.zeros(B, self.hidden_dim, Y, Z, device=x_seq.device)
        out_seq = []
        for t in range(E):
            h = self.cell(x_seq[:, :, t, :, :], h)
            out_seq.append(h)
        return torch.stack(out_seq, dim=2)


# ==========================================
#  软阈值模块
# ==========================================
class SoftThreshold2D(nn.Module):
    """可学习的软阈值模块 (2D版)"""

    def __init__(self, channels):
        super(SoftThreshold2D, self).__init__()
        self.lambd = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.01)

    def forward(self, x):
        lambd = F.relu(self.lambd)
        if torch.rand(1).item() < 0.01:
            print(
                f"[Debug] 特征 x 平均幅值: {x.abs().mean().item():.6f}, 实际空间阈值平均: {lambd.mean().item():.6f}"
            )
        return torch.sign(x) * F.relu(torch.abs(x) - lambd)


class MultiDict_Im_Layer_CDL_2D(nn.Module):
    # 默认针对 12 个回波
    def __init__(self, num_echoes=12, filter_base=32):
        super(MultiDict_Im_Layer_CDL_2D, self).__init__()

        self.num_echoes = num_echoes

        #  修改：因为输入是 SENSE 合并后的单线圈，所以只有 12 个回波
        # 实部 12 + 虚部 12 = 24 通道 (不再有 c 维度)
        in_ch = 2 * num_echoes
        self.c1 = filter_base  # 32

        #  时序记忆网络 (处理纯粹 T2* 衰减)
        self.echo_temporal_gru = EchoConvGRU(input_dim=2, hidden_dim=2)

        # 1. 初始特征提取
        self.enc_in = nn.Conv2d(in_ch, self.c1, kernel_size=3, padding=1, bias=False)

        # 2. 连续的深层特征提取
        self.block1 = self._res_block_2d(self.c1, dilation=1)
        self.block2 = self._res_block_2d(self.c1, dilation=2)
        self.block3 = self._res_block_2d(self.c1, dilation=1)

        # 3. 物理软阈值去噪
        self.st1 = SoftThreshold2D(self.c1)
        self.st2 = SoftThreshold2D(self.c1)
        self.st3 = SoftThreshold2D(self.c1)

        # 4. 输出投影回 24 通道
        self.dec_out = nn.Conv2d(self.c1, in_ch, kernel_size=3, padding=1, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _res_block_2d(self, channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1, bias=False),
        )

    def forward(self, inputs):
        #  修改：inputs 已经是单通道合并图，形状为 [b, e, y, z]
        nb = inputs.shape[0]

        #  修改：直接分离实虚部，生成 [b, 2, e, y, z]，不再有 bc 的纠缠
        x_c = torch.stack([torch.real(inputs), torch.imag(inputs)], dim=1)

        # ==========================================
        #  物理孪生分离：FA4 和 FA16 独立过 GRU (此时速度极快，因为没有线圈维度)
        # ==========================================
        if self.num_echoes == 12:
            x_fa4, x_fa16 = torch.split(x_c, 6, dim=2)
            feat_fa4 = self.echo_temporal_gru(x_fa4) + x_fa4
            feat_fa16 = self.echo_temporal_gru(x_fa16) + x_fa16
            x_c_processed = torch.cat([feat_fa4, feat_fa16], dim=2)
        else:
            x_c_processed = self.echo_temporal_gru(x_c) + x_c

        #  修改：拍扁成 2e 个通道，送进全分辨率 2D 网络
        x_real = rearrange(x_c_processed, "b reim e y z -> b (reim e) y z")

        # --- 全分辨率空间去噪 ---
        feat = self.enc_in(x_real)

        x1 = self.block1(feat) + feat
        x1 = self.st1(x1)

        x2 = self.block2(x1) + x1
        x2 = self.st2(x2)

        x3 = self.block3(x2) + x2
        x3 = self.st3(x3)

        # 解码并施加残差连接
        res = self.dec_out(x3) * self.alpha
        out_real = x_real + res

        #  修改：直接恢复成 [b, e, y, z] 的复数
        re, im = torch.chunk(out_real, 2, dim=1)
        output = torch.complex(re, im)

        return output


"---------------------------------Deep SNMs 基础迭代块--------------------------------"


class Deepsnms_block_ns_image(nn.Module):
    def __init__(self, num_echoes=1):
        super(Deepsnms_block_ns_image, self).__init__()
        self.rho = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.ImageLayer = MultiDict_Im_Layer_CDL_2D(
            num_echoes=num_echoes, filter_base=16
        )

    def forward(self, data, x, Aobj, csm):
        [kData, mask] = data

        r = x - self.rho * Aobj.Ax_b(x)
        # print('rho',self.rho)
        r_ = r + kData
        im_coils = ifft2c(r_)  # [nb, nc, ne, ny, nz]

        # ==========================================
        #  1. 严格能量守恒的 SENSE 合并 (Roemer Optimal)
        # ==========================================
        csm_exp = csm.unsqueeze(2)  # [nb, nc, 1, ny, nz]

        # 计算线圈能量平方和，加上 1e-8 防止除以 0
        csm_mag_sq = torch.sum(csm_exp.abs() ** 2, dim=1, keepdim=True) + 1e-8

        # SENSE 合并，并严格除以能量项
        im_combined = (
            torch.sum(im_coils * torch.conj(csm_exp), dim=1, keepdim=True) / csm_mag_sq
        )
        im_combined = im_combined.squeeze(1)  # [nb, ne, ny, nz]

        # ==========================================
        # 2. 极速去噪 (处理单线圈)
        # ==========================================
        im_denoised_combined = checkpoint(
            self.ImageLayer, im_combined, use_reentrant=False
        )

        # ==========================================
        # 3. SENSE 展开 (因为之前除过了能量，这里展开绝对安全)
        # ==========================================
        im_denoised_coils = (
            im_denoised_combined.unsqueeze(1) * csm_exp
        )  # [nb, nc, ne, ny, nz]

        # 投影回 k 空间
        x_new = fft2c(im_denoised_coils)
        x_new = x_new * (1 - mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        x = x_new
        im_res = ifft2c(x + kData)

        return data, x, im_res


"---------------------------------Deep SNMs Model--------------------------------"


class Deep_snms_echos(nn.Module):
    def __init__(self, niter, num_echoes=12):
        super(Deep_snms_echos, self).__init__()
        self.niter = niter
        self.blocklist = nn.ModuleList()
        for i in range(self.niter):
            self.blocklist.append(Deepsnms_block_ns_image(num_echoes=num_echoes))

    #  修改：接收 csm 并在迭代中传递
    def forward(self, kData, Null_2d, csm, mask, x0):
        # 1. 实例化向量化 A 算子
        Aobj_vec = Aclass_Vectorized(kData, Null_2d, mask)

        # 2. 目标 b 为 -Aobj.b
        b_target = -Aobj_vec.b

        # 3. 执行全并行 CG
        cg_iters = random.randint(12, 20)
        x_init = CG_Vectorized(x0, Aobj_vec, b_target, 30)

        x_cginit = ifft2c(x_init)

        # 4. 提取非采样区作为后续重建的起始点
        x = x_init * Aobj_vec.mask_nacq

        # 开始 Deep SNMs 迭代展开
        data = [kData, mask]
        for i in range(self.niter):
            #  传入 csm
            data, x, im_new = self.blocklist[i](data, x, Aobj_vec, csm)

        ksp = x + kData
        res = ifft2c(ksp)

        return ksp, res, x_cginit


class Deep_snms_echos_mslice(nn.Module):
    def __init__(self, in_channel, niter, num_echoes=12):
        super(Deep_snms_echos_mslice, self).__init__()
        self.in_channel = in_channel
        self.niter = niter

        self.unrolling_net = Deep_snms_echos(self.niter, num_echoes=num_echoes)
        self.Im_conv3d = Im_cnn3d_layer_mslice_enhanced(in_channel=self.in_channel)

    def forward(self, kData, Null_2d, csm, mask, x0):
        # 维度准备
        nb, ns, nc, ne, ny, nz = kData.size()
        x = x0

        # 1. 展开 Batch 和 Slice
        x0_ = rearrange(x, "b s c e y z -> (b s) c e y z")
        kData_ = rearrange(kData, "b s c e y z -> (b s) c e y z")
        Null_2d_ = rearrange(Null_2d, "b s c2 c1 y z -> (b s) c2 c1 y z")
        csm_ = rearrange(csm, "b s c y z -> (b s) c y z")

        # 2. 运行 2D 迭代展开网络
        ksp_new, im_new, x_cginit = self.unrolling_net(
            kData_, Null_2d_, csm_, mask, x0_
        )
        # ksp_new, im_new = self.unrolling_net(kData_, Null_2d_, csm_, mask, x0_)

        # ==========================================
        #  3. 核心修复：SENSE 合并后再进入 3D CNN
        # ==========================================
        csm_exp = csm_.unsqueeze(2)
        csm_mag_sq = torch.sum(csm_exp.abs() ** 2, dim=1, keepdim=True) + 1e-8

        # 将 2D 网络的输出合并为纯净单通道
        im_combined = (
            torch.sum(im_new * torch.conj(csm_exp), dim=1, keepdim=True) / csm_mag_sq
        )
        im_combined = im_combined.squeeze(1)  # 得到 [(b s), e, ny, nz] 的单线圈解剖图

        # 送入 3D CNN (此时它只看解剖结构，不受线圈干扰！且计算量降 12 倍)
        im_3d_in = rearrange(im_combined, "(b s) e y z -> (b e) s y z", b=nb, s=ns)
        input = c2r(im_3d_in)
        output = self.Im_conv3d(input)
        im_3d_out = r2c(output)

        # ==========================================
        #  4. SENSE 再次展开回多线圈，准备最后的数据一致性
        # ==========================================
        im_combined_new = rearrange(im_3d_out, "(b e) s y z -> (b s) e y z", b=nb, s=ns)
        im_new_final = im_combined_new.unsqueeze(1) * csm_exp  # [(b s), c, e, y, z]

        # 恢复成原始的 6D
        im_new_6d = rearrange(
            im_new_final, "(b s) c e y z -> b s c e y z", b=nb, c=nc, e=ne
        )

        # 5. 最后一次严格数据一致性 (DC)
        x_out = fft2c(im_new_6d)

        # 注意匹配 6D mask 维度
        x_out = x_out * (1 - mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        ksp = x_out + kData
        res = ifft2c(ksp)
        x_cginit = x_cginit.unsqueeze(0)
        print("shape of res:", res.shape)
        print("shape of x_cginit:", x_cginit.shape)

        return ksp, res, x_cginit


"---------------------------------A function--------------------------------"
"---------------------------------A function (100% 向量化)--------------------------------"


class Aclass_Vectorized:
    def __init__(self, kData, Null_2d, mask):
        # 此时 kData shape: [nb, nc, ne, ny, nz]
        # Null_2d shape: [nb, c2, c1, ny, nz]
        self.kData = kData
        self.Null_2d = Null_2d

        # 适应 5D 张量的 mask，假设原 mask 为 2D (ny, nz)
        self.mask_nacq = (1 - mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # ifft2c 作用在最后两维，yy shape: [nb, nc, ne, ny, nz]
        yy = ifft2c(self.kData).unsqueeze(1)  # [nb, 1, nc, ne, ny, nz]
        Null_2d_expanded = self.Null_2d.unsqueeze(3)  # [nb, c2, c1, 1, ny, nz]

        # 利用广播机制相乘并在通道 c1 (dim=2) 上求和，得到 [nb, c2, ne, ny, nz]
        yy_Null = torch.sum(Null_2d_expanded * yy, dim=2)
        ksp_ = fft2c(yy_Null)
        self.b = ksp_ * self.mask_nacq

    def A(self, x):
        # x shape: [nb, nc, ne, ny, nz]
        ksp_nacq = x * self.mask_nacq
        im = ifft2c(ksp_nacq).unsqueeze(1)  # [nb, 1, nc, ne, ny, nz]
        Null_2d_expanded = self.Null_2d.unsqueeze(3)  # [nb, c2, c1, 1, ny, nz]
        im_Null = torch.sum(Null_2d_expanded * im, dim=2)  # [nb, c2, ne, ny, nz]
        ksp_ = fft2c(im_Null)
        res = ksp_ * self.mask_nacq
        return res

    def AH(self, y):
        return self.A(y)

    def AHAx_AHb(self, x):
        y = self.Ax_b(x)
        return self.AH(y)

    def Ax_b(self, x):
        return self.A(x) + self.b


"---------------------------------保留的原有辅助层--------------------------------"


class Conv2plus1d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: [int, int, int],
        padding: [int, int, int],
        stride: [int, int, int],
        **kwargs,
    ):
        super(Conv2plus1d, self).__init__()
        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)
        temporal_stride = (stride[0], 1, 1)
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_padding = (0, padding[1], padding[2])
        spatial_stride = (1, stride[1], stride[2])
        mid_channels = max(in_channel, out_channel)

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=mid_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_padding,
                stride=temporal_stride,
                **kwargs,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=mid_channels,
                out_channels=out_channel,
                kernel_size=spatial_kernel_size,
                padding=spatial_padding,
                stride=spatial_stride,
                **kwargs,
            ),
        )

    def forward(self, x):
        return self.conv(x)


class Im_cnn3d_layer_mslice_enhanced(nn.Module):
    def __init__(self, in_channel, filter=32, slice_kernel=7):
        super(Im_cnn3d_layer_mslice_enhanced, self).__init__()
        # 计算 Padding，保持图像尺寸不变
        pad_s = slice_kernel // 2

        # 直接使用 3D 卷积，暴力提取层间与空间的联合特征
        self.conv1 = nn.Conv3d(
            in_channel,
            filter,
            kernel_size=(slice_kernel, 3, 3),
            padding=(pad_s, 1, 1),
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=filter)

        self.conv2 = nn.Conv3d(
            filter,
            filter,
            kernel_size=(slice_kernel, 3, 3),
            padding=(pad_s, 1, 1),
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=filter)

        self.conv3 = nn.Conv3d(
            filter,
            in_channel,
            kernel_size=(slice_kernel, 3, 3),
            padding=(pad_s, 1, 1),
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1)  # 3D Dropout 强化跨层泛化

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        return x + inputs


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: [int, int, int],
        stride: [int, int, int],
        padding: [int, int, int],
        **kwargs,
    ):
        super(Conv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **kwargs,
            )
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        return self.relu(x2 + x)


def CG_Vectorized(x, Aobj, b, max_iter=20, eps=1e-5):
    """
    全张量并行的共轭梯度法 (Conjugate Gradient)
    自动适应任意回波数量 (12或6均可)，消灭 for 循环
    """
    r = b - Aobj.A(x)
    p = r

    # 沿着除 Batch 以外的所有维度求点积 (dim_sum = [1, 2, 3, 4])
    dim_sum = list(range(1, len(x.shape)))
    rTr = torch.sum(torch.conj(r) * r, dim=dim_sum, keepdim=True).real

    for i in range(max_iter):
        if rTr.abs().max() < eps:
            break
        Ap = Aobj.A(p)
        alpha = rTr / torch.sum(torch.conj(p) * Ap, dim=dim_sum, keepdim=True).real
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(torch.conj(r) * r, dim=dim_sum, keepdim=True).real
        beta = rTrNew / rTr
        p = r + beta * p
        rTr = rTrNew

    return x
