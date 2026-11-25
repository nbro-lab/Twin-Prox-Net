"""Exact pairwise-ratio architecture for mapping_params_100_jointloss.pkl.

This historical checkpoint is not the integrated two-channel architecture.
"""
import torch
from torch import nn

def calculate_delta_S_ratio(s2, echo, eps=1e-8):
    # signal ratio between neighboring echoes
    # s2 : [nb nc na n2 ny nz]

    a = torch.sum(torch.conj(s2[:, :, :, echo, :, :]) * s2[:, :, :, echo + 1, :, :], dim=(1,2))
    b = torch.sum(torch.abs(s2[:,:,:,echo,:,:]) ** 2, dim=(1,2))

    # a = a.to(dtype=torch.complex128)
    # b = b.to(dtype=torch.complex128)

    delta_S = a / (b + eps)

    return delta_S

def calculate_R2_ratio(s2, echo, eps=1e-8):
    # signal ratio between each flip-angle
    # s2 : [nb nc na n2 ny nz]

    a = torch.sum(torch.conj(s2[:, :, 1, echo, :, :]) * s2[:, :, 0, echo, :, :], dim=1)
    b = torch.sum(torch.abs(s2[:, :, 1, echo, :, :]) ** 2, dim=1)

    R2 = a / (b + eps)

    return R2

class Learned_S2(nn.Module):
    def __init__(self):
        super(Learned_S2, self).__init__()
        self.ImageLayer_S = ImageLayer_S(in_channel=8, out_channel=2)
        self.ConvLayer = ImageLayer_S2(in_channel=8, out_channel=8)

    def forward(self, s2, mask_brain):

        delta_S = []
        for iter_ne in range(4):
            delta_s = calculate_delta_S_ratio(s2, echo=iter_ne)  # [nb ny nz]
            delta_S.append(delta_s)
        delta_S = torch.stack(delta_S, dim=1)  # [nb 4 ny nz]
        input = torch.cat([torch.real(delta_S), torch.imag(delta_S)], dim=1)  # [nb 2*4 ny nz]
        
        deltaS_correct = self.ConvLayer(input)
        output = self.ImageLayer_S(deltaS_correct)
        re_, im_ = torch.chunk(output, 2, 1)
        delta_S_new = torch.complex(re_, im_).squeeze(1)  # [nb ny nz]
        # delta_S_new = delta_S_new * mask_brain

        return  delta_S_new

class Learned_R2(nn.Module):
    def __init__(self):
        super(Learned_R2, self).__init__()
        self.ImageLayer_R = ImageLayer_S(in_channel=10, out_channel=2)
        self.ConvLayer = ImageLayer_S2(in_channel=10, out_channel=10)

    def forward(self, s2, mask_brain):

        R2 = []
        for iter_ne in range(5):
            r2 = calculate_R2_ratio(s2, echo=iter_ne)  # [nb ny nz]
            R2.append(r2)

        R2 = torch.stack(R2, dim=1)  # [nb 5 ny nz]
        input = torch.cat([torch.real(R2), torch.imag(R2)], dim=1)  # [nb 5*2 ny nz]
        R2_correct = self.ConvLayer(input) # [nb 5*2 ny nz]

        output = self.ImageLayer_R(R2_correct)
        # output = self.ConvLayer(output)
        re_, im_ = torch.chunk(output, 2, 1)
        R2_new = torch.complex(re_, im_).squeeze(1)  # [nb ny nz]
        # R2_new = output.squeeze(1)

        return  R2_new

class ImageLayer_S(nn.Module):
    def __init__(self, in_channel, out_channel, filter=64):
        super(ImageLayer_S, self).__init__()
        conf = dict(in_ch=in_channel, out_ch=out_channel, filter=filter, ksize=1, pad=0, st=1, bias=True)

        self.mylayers = []

        self.mylayers.append(nn.Conv2d(in_channels=conf["in_ch"], out_channels=conf["filter"],
                                       kernel_size=conf["ksize"], stride=conf["st"], padding=conf["pad"], bias=conf["bias"]))
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(nn.Conv2d(in_channels=conf["filter"], out_channels=conf["filter"],
                                       kernel_size=conf["ksize"], stride=conf["st"], padding=conf["pad"], bias=conf["bias"]))
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(nn.Conv2d(in_channels=conf["filter"], out_channels=conf["out_ch"],
                                       kernel_size=conf["ksize"], stride=conf["st"], padding=conf["pad"], bias=conf["bias"]))

        self.seq = nn.Sequential(*self.mylayers)

    def forward(self, inputs):
        res = self.seq(inputs)
        # res = inputs + res
        return res

class ImageLayer_S2(nn.Module):
    def __init__(self, in_channel, out_channel, filter=64):
        super(ImageLayer_S2, self).__init__()
        conf = dict(in_ch=in_channel, out_ch=out_channel, filter=filter, ksize=3, pad=1, st=1, bias=True)

        self.mylayers = []

        self.mylayers.append(nn.Conv2d(in_channels=conf["in_ch"], out_channels=conf["filter"],
                                       kernel_size=conf["ksize"], stride=conf["st"], padding=conf["pad"], bias=conf["bias"]))
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(nn.Conv2d(in_channels=conf["filter"], out_channels=conf["filter"],
                                       kernel_size=conf["ksize"], stride=conf["st"], padding=conf["pad"], bias=conf["bias"]))
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(nn.Conv2d(in_channels=conf["filter"], out_channels=conf["out_ch"],
                                       kernel_size=conf["ksize"], stride=conf["st"], padding=conf["pad"], bias=conf["bias"]))

        self.seq = nn.Sequential(*self.mylayers)

    def forward(self, inputs):
        im = self.seq(inputs)
        res = inputs + im
        return res

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
        recon_FA4 = recon_im[:, :, 0:6, :, :]   # 前 6 个
        recon_FA16 = recon_im[:, :, 6:12, :, :] # 后 6 个
        
        # 堆叠成 MDI 物理公式需要的结构: [nb, nc, na(2), ne(6), ny, nz]
        im_stacked = torch.stack([recon_FA4, recon_FA16], dim=2)
        
        # 提取用于计算的后 5 个回波 (摒弃第1个回波避免非稳态效应)
        s2 = im_stacked[:, :, :, 1:, :, :] # 形状: [nb, nc, 2, 5, ny, nz]
        
        # 施加大脑掩膜
        s2_masked = s2 * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        # 送入学习网络
        delta_S = self.learned_S(s2_masked, mask_brain)
        R2 = self.learned_R(s2_masked, mask_brain)
        
        return delta_S, R2
