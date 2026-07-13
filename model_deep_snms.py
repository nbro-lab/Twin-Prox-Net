import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mdi_function import (
    calculate_delta_S_ratio,
    calculate_R2,
    calculate_R2_ratio,
    reconstruct_v_deltaS,
    reconstruct_v_R2,
    systhesis_v,
    systhesis_v2,
    systhesis_v3,
    systhesis_v_deltaS,
    systhesis_v_echo,
    systhesis_v_R,
    systhesis_v_R2,
)
from torch.nn import init
from utils import c2r, fft2c, ifft2c, r2c


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
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
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


"---------------------------------Deep SNMs Block with image output--------------------------------"


class Deepsnms_block_image(nn.Module):
    def __init__(self, ImageLayer, rho, nGrad):
        super(Deepsnms_block_image, self).__init__()
        self.ImageLayer = ImageLayer
        self.rho = rho
        self.nGrad = nGrad

    def forward(self, data, x, Aobj):
        [kData, mask] = data

        # gradient descent
        # Aobj = Aclass(kData, Null_2d, mask)
        for i in range(self.nGrad):
            r = x - self.rho * Aobj.Ax_b(x)  # r is non-acquired data
            x = r

        r_ = r + kData  # full kspace data

        im = ifft2c(r_)  # [nb nc ny nz]

        # CNN
        nb, nc, ny, nz = im.size()
        im_ = torch.reshape(im, (nb * nc, ny, nz))
        input = c2r(im_)  # [nb*nc, 2, ny, nz]
        output = self.ImageLayer(input)  # [nb*nc, 2, ny, nz]
        im_new = r2c(output)  # [nb*nc, ny, nz]
        im_new = torch.reshape(im_new, (nb, nc, ny, nz))
        x_new = fft2c(im_new)
        x_new = x_new * (1 - mask).unsqueeze(0).unsqueeze(
            0
        )  # Dc   x_new : [nb nc ny nz]

        # update x
        x = x_new
        im_res = ifft2c(x + kData)  # [nb nc ny nz]

        return x, im_res


class Deepsnms_block_ns_image(nn.Module):
    def __init__(self, in_channel):
        super(Deepsnms_block_ns_image, self).__init__()
        self.rho = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.ImageLayer = Im_cnn_layer(in_channel=in_channel)

    def forward(self, data, x, Aobjs):
        [kData, mask] = data
        nb, nc, ne, ny, nz = kData.size()
        # gradient descent
        # Aobj = Aclass(kData, Null_2d, mask)
        r = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for i in range(ne):
            Aobj = Aobjs[i]
            r[:, :, i, :, :] = x[:, :, i, :, :] - self.rho * Aobj.Ax_b(
                x[:, :, i, :, :]
            )  # r is non-acquired data

        r_ = r + kData  # full kspace data

        im = ifft2c(r_)  # [nb nc ne ny nz]
        # CNN
        im_new = self.ImageLayer(im)
        x_new = fft2c(im_new)  # [nb nc ne ny nz]
        x_new = x_new * (1 - mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Update x
        x = x_new
        im_res = ifft2c(x + kData)

        return data, x, im_res


"---------------------------------Deep SNMs Model not-share parameter with image output--------------------------------"


class Deep_snms_echos(nn.Module):
    def __init__(self, in_channel, niter):
        super(Deep_snms_echos, self).__init__()
        self.in_channel = in_channel
        self.niter = niter
        self.blocklist = nn.ModuleList()
        # self.rho = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        for i in range(self.niter):
            self.blocklist.append(Deepsnms_block_ns_image(self.in_channel))

    def forward(self, kData, Null_2d, mask, x0):
        """
        DeepSNMs for sigle-slice multi-echo ksp data
        Args:
            kData: [nb nc ne ny nz]
            Null_2d: [nb nc2 nc1 ks ks], which is same for each echo
            mask: [ny nz]
            x0: [nb nc ne ny nz]

        Returns:
            ksp_new: [nb nc ne ny nz]
            im_new: [nb nc ne ny nz]
        """
        nb, nc, ne, ny, nz = kData.size()
        # initialization
        # x = torch.zeros_like(kData)
        x = x0  # [nb nc ne ny nz]

        data = [kData, mask]
        Aobjs = []
        for i in range(ne):
            kdata = kData[:, :, i, :, :]
            # mask_ = mask[i, :, :] # for complementary kspace
            Aobj = Aclass(kdata, Null_2d, mask)
            Aobjs.append(Aobj)

        # List_image = []
        for i in range(self.niter):
            data, x, im_new = self.blocklist[i](data, x, Aobjs)
            # List_image.append(im_new)

        ksp = x + kData
        res = ifft2c(ksp)

        return ksp, res


# class Learned_S2(nn.Module):
#     def __init__(self):
#         super(Learned_S2, self).__init__()
#         self.ImageLayer_S = ImageLayer_S(in_channel=8, out_channel=2)
#
#     def forward(self, im_cnn, mask_brain):
#         # 2D CNN for delta_S
#         im = im_cnn.unsqueeze(2)  # [nb nc 1 ne ny nz]
#         s2 = im[:, :, :, 1:, :, :]  # [nb nc 1 ne-1 ny nz]
#         s2 = s2 #* mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)
#
#         delta_S = []
#         for iter_ne in range(4):
#             delta_s = calculate_delta_S_ratio(s2, echo=iter_ne)  # [nb ny nz]
#             delta_S.append(delta_s)
#         delta_S = torch.stack(delta_S, dim=1)  # [nb 4 ny nz]
#         input = torch.cat([torch.real(delta_S), torch.imag(delta_S)], dim=1)  # [nb 2*4 ny nz]
#         output = self.ImageLayer_S(input)
#         re_, im_ = torch.chunk(output, 2, 1)
#         delta_S_new = torch.complex(re_, im_).squeeze(1)  # [nb ny nz]
#         # delta_S_new = delta_S_new * mask_brain
#
#         # Integration
#         v11 = im_cnn[:, :, 0, :, :] #* mask_brain.unsqueeze(1)  # [nb nc ny nz] na is temporarily 1
#         v12 = im_cnn[:, :, 1, :, :] #* mask_brain.unsqueeze(1)
#         im_new = reconstruct_v_deltaS(v11, v12, delta_S_new, s2_shape=s2.size())  # only barin
#         im_new = im_new.squeeze(2)  # [nb nc ne ny nz]
#
#         return im_new, delta_S_new


class Joint_echos_maps(nn.Module):
    def __init__(self, in_channel, niter):
        super(Joint_echos_maps, self).__init__()
        self.in_channel = in_channel
        self.niter = niter
        self.recon_net = Deep_snms_echos(self.in_channel, self.niter)
        self.mapping = Learned_S2()

    def forward(self, kData, Null_2d, mask, mask_brain, x0):
        # with torch.no_grad:
        ksp, res = self.recon_net(kData, Null_2d, mask, x0)

        im_syn, delta_S = self.mapping(res, mask_brain)

        return ksp, res, im_syn, delta_S


"--------------------------------------------------MDI CNN mapping-----------------------------------------------------"


class MDI_mapping(nn.Module):
    def __init__(self):
        super(MDI_mapping, self).__init__()
        self.learned_S = Learned_S()
        self.learned_R = Learned_R()

    def forward(self, recon_ksp, mask_brain):
        # recon_ksp: [nb nc ne ny nz]
        # mask_brain: [nb ny nz]

        recon_coils = ifft2c(recon_ksp)  # [nb nc ne ny nz]

        im = torch.chunk(recon_coils, 2, dim=2)
        im = torch.stack(im, dim=2)  # [nb nc na n2+1 ny nz]

        s2 = im[:, :, :, 1:, :, :]  # [nb nc na n2 ny nz]
        s2 = s2 * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        delta_S = self.learned_S(s2, mask_brain)
        R2 = self.learned_R(s2, mask_brain)

        # v11 = im[:, :, 0, 0, :, :] * mask_brain.unsqueeze(1)
        # v12 = im[:, :, 0, 1, :, :] * mask_brain.unsqueeze(1)
        # v21 = im[:, :, 1, 0, :, :] * mask_brain.unsqueeze(1)
        # im_new = systhesis_v(v11, v12, v21, delta_S, R2, s2.size())

        im_ = im * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        im_new = systhesis_v3(im_, delta_S, R2, s2.size())

        im_new = im_new + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        im_new = torch.chunk(im_new, 2, dim=2)
        im_new = torch.cat(im_new, dim=3).squeeze(2)

        return recon_coils, im_new, delta_S, R2


class Learned_S(nn.Module):
    def __init__(self):
        super(Learned_S, self).__init__()
        self.ImageLayer_S = ImageLayer_S(in_channel=8, out_channel=2)
        # self.ConvLayer = ImageLayer_S2(in_channel=8, out_channel=8)

    def forward(self, s2, mask_brain):

        delta_S = []
        for iter_ne in range(4):
            delta_s = calculate_delta_S_ratio(s2, echo=iter_ne)  # [nb ny nz]
            delta_S.append(delta_s)
        delta_S = torch.stack(delta_S, dim=1)  # [nb 4 ny nz]
        input = torch.cat(
            [torch.real(delta_S), torch.imag(delta_S)], dim=1
        )  # [nb 2*4 ny nz]
        output = self.ImageLayer_S(input)
        # output = self.ConvLayer(deltaS_hat)
        re_, im_ = torch.chunk(output, 2, 1)
        delta_S_new = torch.complex(re_, im_).squeeze(1)  # [nb ny nz]
        # delta_S_new = delta_S_new * mask_brain

        return delta_S_new


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
        input = torch.cat(
            [torch.real(delta_S), torch.imag(delta_S)], dim=1
        )  # [nb 2*4 ny nz]

        deltaS_correct = self.ConvLayer(input)
        output = self.ImageLayer_S(deltaS_correct)
        re_, im_ = torch.chunk(output, 2, 1)
        delta_S_new = torch.complex(re_, im_).squeeze(1)  # [nb ny nz]
        # delta_S_new = delta_S_new * mask_brain

        return delta_S_new


class Learned_R(nn.Module):
    def __init__(self):
        super(Learned_R, self).__init__()
        self.ImageLayer_R = ImageLayer_S(in_channel=10, out_channel=2)
        # self.ConvLayer = ImageLayer_S2(in_channel=2, out_channel=2)

    def forward(self, s2, mask_brain):

        R2 = []
        for iter_ne in range(5):
            r2 = calculate_R2_ratio(s2, echo=iter_ne)  # [nb ny nz]
            R2.append(r2)
        R2 = torch.stack(R2, dim=1)  # [nb 5 ny nz]
        # input = torch.abs(R2)  # [nb 5 ny nz]
        input = torch.cat([torch.real(R2), torch.imag(R2)], dim=1)  # [nb 2*5 ny nz]
        output = self.ImageLayer_R(input)
        # output = self.ConvLayer(output)
        re_, im_ = torch.chunk(output, 2, 1)
        R2_new = torch.complex(re_, im_).squeeze(1)  # [nb ny nz]
        # R2_new = output.squeeze(1)

        return R2_new


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
        R2_correct = self.ConvLayer(input)  # [nb 5*2 ny nz]

        output = self.ImageLayer_R(R2_correct)
        # output = self.ConvLayer(output)
        re_, im_ = torch.chunk(output, 2, 1)
        R2_new = torch.complex(re_, im_).squeeze(1)  # [nb ny nz]
        # R2_new = output.squeeze(1)

        return R2_new


class MDI_mapping2(nn.Module):
    def __init__(self):
        super(MDI_mapping2, self).__init__()
        self.ImageLayer_S = ImageLayer_S2(in_channel=8, out_channel=8)
        self.ImageLayer_R = ImageLayer_S2(in_channel=10, out_channel=10)

    def forward(self, recon_ksp, mask_brain):
        # recon_ksp: [nb nc ne ny nz]
        # mask_brain: [nb ny nz]

        recon_coils = ifft2c(recon_ksp)  # [nb nc ne ny nz]

        im = torch.chunk(recon_coils, 2, dim=2)
        im = torch.stack(im, dim=2)  # [nb nc na n2+1 ny nz]

        s2 = im[:, :, :, 1:, :, :]  # [nb nc na n2 ny nz]
        s2 = s2 * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # delta_S
        delta_S = []
        for iter_ne in range(4):
            delta_s = calculate_delta_S_ratio(s2, echo=iter_ne)  # [nb ny nz]
            delta_S.append(delta_s)
        delta_S = torch.stack(delta_S, dim=1)  # [nb 4 ny nz]
        input = torch.cat(
            [torch.real(delta_S), torch.imag(delta_S)], dim=1
        )  # [nb 2*4 ny nz]
        output = self.ImageLayer_S(input)
        re_, im_ = torch.chunk(output, 2, 1)
        delta_S_new = torch.complex(re_, im_).squeeze(1)  # [nb 4 ny nz]

        # R2
        R2 = []
        for iter_ne in range(5):
            r2 = calculate_R2_ratio(s2, echo=iter_ne)  # [nb ny nz]
            R2.append(r2)
        R2 = torch.stack(R2, dim=1)  # [nb 4 ny nz]
        input = torch.cat([torch.real(R2), torch.imag(R2)], dim=1)  # [nb 2*5 ny nz]
        output = self.ImageLayer_R(input)
        re_, im_ = torch.chunk(output, 2, 1)
        R2_new = torch.complex(re_, im_).squeeze(1)  # [nb 5 ny nz]

        v11 = im[:, :, 0, 0, :, :] * mask_brain.unsqueeze(1)
        v12 = im[:, :, 0, 1, :, :] * mask_brain.unsqueeze(1)
        v21 = im[:, :, 1, 0, :, :] * mask_brain.unsqueeze(1)

        im_new = systhesis_v_echo(v11, v12, v21, delta_S_new, R2_new, s2.size())

        im_new = im_new + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        im_new = torch.chunk(im_new, 2, dim=2)
        im_new = torch.cat(im_new, dim=3).squeeze(2)

        return recon_coils, im_new, delta_S_new, R2_new


class MDI_mapping3(nn.Module):
    def __init__(self):
        super(MDI_mapping3, self).__init__()
        self.learned_S = Learned_S()
        self.learned_R = Learned_R()

    def forward(self, recon_ksp, mask_brain):
        # recon_ksp: [nb nc ne ny nz]
        # mask_brain: [nb ny nz]

        recon_coils = ifft2c(recon_ksp)  # [nb nc ne ny nz]

        im = torch.chunk(recon_coils, 2, dim=2)
        im = torch.stack(im, dim=2)  # [nb nc na n2+1 ny nz]

        s2 = im[:, :, :, 1:, :, :]  # [nb nc na n2 ny nz]
        s2 = s2 * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        delta_S = self.learned_S(s2, mask_brain)
        R2 = self.learned_R(s2, mask_brain)

        # v11 = im[:, :, 0, 0, :, :] * mask_brain.unsqueeze(1)
        # v12 = im[:, :, 0, 1, :, :] * mask_brain.unsqueeze(1)
        # v21 = im[:, :, 1, 0, :, :] * mask_brain.unsqueeze(1)
        # im_new = systhesis_v(v11, v12, v21, delta_S, R2, s2.size())

        im_ = im * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        im_new1 = systhesis_v_deltaS(im_, delta_S, s2.size())
        im_new1 = im_new1 + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        im_new1 = torch.chunk(im_new1, 2, dim=2)
        im_new1 = torch.cat(im_new1, dim=3).squeeze(2)

        im_new2 = systhesis_v_R(im_, R2, s2.size())
        im_new2 = im_new2 + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        im_new2 = torch.chunk(im_new2, 2, dim=2)
        im_new2 = torch.cat(im_new2, dim=3).squeeze(2)

        return recon_coils, im_new1, im_new2, delta_S, R2


class MDI_mapping_deltaS(nn.Module):
    def __init__(self):
        super(MDI_mapping_deltaS, self).__init__()
        self.learned_S = Learned_S2()
        # self.learned_R = Learned_R()

    def forward(self, recon_ksp, mask_brain):
        # recon_ksp: [nb nc ne ny nz]
        # mask_brain: [nb ny nz]

        recon_coils = ifft2c(recon_ksp)  # [nb nc ne ny nz]

        im = torch.chunk(recon_coils, 2, dim=2)
        im = torch.stack(im, dim=2)  # [nb nc na n2+1 ny nz]

        s2 = im[:, :, :, 1:, :, :]  # [nb nc na n2 ny nz]
        s2 = s2 * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        delta_S = self.learned_S(s2, mask_brain)
        # R2 = self.learned_R(s2, mask_brain)

        # v11 = im[:, :, 0, 0, :, :] * mask_brain.unsqueeze(1)
        # v12 = im[:, :, 0, 1, :, :] * mask_brain.unsqueeze(1)
        # v21 = im[:, :, 1, 0, :, :] * mask_brain.unsqueeze(1)
        # im_new = systhesis_v(v11, v12, v21, delta_S, R2, s2.size())

        im_ = im * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        im_new1 = systhesis_v_deltaS(im_, delta_S, s2.size())
        im_new1 = im_new1 + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        im_new1 = torch.chunk(im_new1, 2, dim=2)
        im_new1 = torch.cat(im_new1, dim=3).squeeze(2)

        # im_new2 = systhesis_v_R(im_, delta_S, s2.size())
        # im_new2 = im_new2 + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # im_new2 = torch.chunk(im_new2, 2, dim=2)
        # im_new2 = torch.cat(im_new2, dim=3).squeeze(2)

        # return recon_coils, im_new1, im_new2, delta_S, R2
        return recon_coils, im_new1, delta_S


class MDI_mapping_R(nn.Module):
    def __init__(self):
        super(MDI_mapping_R, self).__init__()
        # self.learned_S = Learned_S()
        self.learned_R = Learned_R2()

    def forward(self, recon_ksp, mask_brain):
        # recon_ksp: [nb nc ne ny nz]
        # mask_brain: [nb ny nz]

        recon_coils = ifft2c(recon_ksp)  # [nb nc ne ny nz]

        im = torch.chunk(recon_coils, 2, dim=2)
        im = torch.stack(im, dim=2)  # [nb nc na n2+1 ny nz]

        s2 = im[:, :, :, 1:, :, :]  # [nb nc na n2 ny nz]
        # s2_ = im # [nb nc na n2+1 ny nz]
        # s2_ = s2_ * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # delta_S = self.learned_S(s2, mask_brain)
        R2 = self.learned_R(s2, mask_brain)
        # R2_ = calculate_R2(s2, mask_brain)

        # v11 = im[:, :, 0, 0, :, :] * mask_brain.unsqueeze(1)
        # v12 = im[:, :, 0, 1, :, :] * mask_brain.unsqueeze(1)
        # v21 = im[:, :, 1, 0, :, :] * mask_brain.unsqueeze(1)
        # im_new = systhesis_v(v11, v12, v21, delta_S, R2, s2.size())

        im_ = im * mask_brain.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # im_new1 = systhesis_v_deltaS(im_, delta_S, s2.size())
        # im_new1 = im_new1 + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # im_new1 = torch.chunk(im_new1, 2, dim=2)
        # im_new1 = torch.cat(im_new1, dim=3).squeeze(2)

        im_new2 = systhesis_v_R(im_, R2, s2.size())
        im_new2 = im_new2 + im * (1 - mask_brain).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        im_new2 = torch.chunk(im_new2, 2, dim=2)
        im_new2 = torch.cat(im_new2, dim=3).squeeze(2)

        # return recon_coils, im_new1, im_new2, delta_S, R2
        return recon_coils, im_new2, R2


"---------------------------------Deep SNMs Model share parameter with find fixed point training--------------------------------"


class Deep_snms_JBF(nn.Module):
    def __init__(self, in_channel, niter, nGrad, max_iter=20, tol=1e-4):
        super(Deep_snms_JBF, self).__init__()
        self.in_channel = in_channel
        self.niter = niter
        self.nGrad = nGrad
        self.blocklist = []
        # self.imageconv = Unet(in_channels=in_channel, out_channels=in_channel, bilinear=True, base_c=32)
        self.imageconv = ImageLayer(in_channel=self.in_channel)
        self.rho = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.net_block = Deepsnms_block_image(self.imageconv, self.rho, self.nGrad)

        self.max_iter = max_iter
        self.tol = tol

    def forward(self, kData, Null_2d, mask, x0):
        x = x0

        data = [kData, mask]

        Aobj = Aclass(kData, Null_2d, mask)

        with torch.no_grad():
            x = self.find_fixed_point(data, x, Aobj)

        x, im_res = self.net_block(data, x, Aobj)

        res = x + kData

        return res

    def find_fixed_point(self, data, x, Aobj):
        with torch.no_grad():
            x0 = x
            temp = x0
            for i in range(self.max_iter - 1):
                x, _ = self.net_block(data, temp, Aobj)
                if self._converge(x, temp):
                    return x
                else:
                    temp = x

        return temp

    def _converge(self, x1, x2):

        res = (x1 - x2).norm().item() / (1e-5 + x2.norm().item())

        if res < self.tol:
            return True
        else:
            return False


"---------------------------------A function--------------------------------"


# Ax = Dc*F*N*F^H*Dc^H*x
# forward = ||Dc*F*N*F^H*Dc^H*x + Dc*F*N*F^H*D^H*y_acq||^2
# x : [nb, nc, nx, ny]
# Null_2d : [nb, nc, nc, nx, ny]
# mask : [nx, ny]
class Aclass:
    def __init__(self, kData, Null_2d, mask):
        self.kData = kData
        self.Null_2d = Null_2d
        mask_acq = mask
        self.mask_nacq = (1 - mask_acq).unsqueeze(0).unsqueeze(0)  # [1, 1, nx, ny]

        # calculate b
        yy = ifft2c(self.kData).unsqueeze(dim=1)  # [nb 1 nc ny nz]
        yy_Null = torch.sum(
            self.Null_2d * yy, dim=2
        )  # [nb nc nc ny nz] .* [nb 1 nc ny nz] -> [nb nc ny nz]
        ksp_ = fft2c(yy_Null)
        self.b = ksp_ * self.mask_nacq

    def A(self, x):
        ksp_nacq = x * self.mask_nacq
        im = ifft2c(ksp_nacq).unsqueeze(dim=1)  # [nb 1 nc ny nz]
        im_Null = torch.sum(
            self.Null_2d * im, dim=2
        )  # [nb nc nc ny nz] .* [nb 1 nc ny nz] -> [nb nc ny nz]
        ksp_ = fft2c(im_Null)
        res = ksp_ * self.mask_nacq
        return res

    def AH(self, y):
        res = self.A(y)  # A is symmetric positive
        return res

    def AHAx_AHb(self, x):
        y = self.Ax_b(x)
        res = self.AH(y)
        return res

    def Ax_b(self, x):
        res = self.A(x) + self.b
        return res


"---------------------------------CNN Layer--------------------------------"


class ImageLayer(nn.Module):
    def __init__(self, in_channel, filter=32):
        super(ImageLayer, self).__init__()

        self.mylayers = []

        self.mylayers.append(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=filter,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=filter,
                out_channels=filter,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        self.mylayers.append(nn.ReLU(True))

        # self.mylayers.append(nn.Conv2d(in_channels=filter, out_channels=filter, kernel_size=3, stride=1, padding=1, bias=False))
        # self.mylayers.append(nn.ReLU(True))
        #
        # self.mylayers.append(nn.Conv2d(in_channels=filter, out_channels=filter, kernel_size=3, stride=1, padding=1, bias=False))
        # self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=filter,
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )

        self.seq = nn.Sequential(*self.mylayers)

    def forward(self, inputs):
        im = self.seq(inputs)
        res = inputs + im
        return res


class Im_cnn_layer(nn.Module):
    def __init__(self, in_channel, filter=64):
        super(Im_cnn_layer, self).__init__()
        conf = dict(
            in_ch=in_channel,
            filter1=32,
            filter2=filter,
            ksize=3,
            pad=1,
            st=1,
            bias=True,
        )

        self.conv1 = Conv2plus1d(
            conf["in_ch"],
            conf["filter1"],
            kernel_size=(conf["ksize"], 3, 3),
            stride=(conf["pad"], 1, 1),
            padding=(conf["st"], 1, 1),
            bias=conf["bias"],
        )
        self.conv2 = Conv2plus1d(
            conf["filter1"],
            conf["filter2"],
            kernel_size=(conf["ksize"], 3, 3),
            stride=(conf["pad"], 1, 1),
            padding=(conf["st"], 1, 1),
            bias=conf["bias"],
        )
        self.conv3 = Conv2plus1d(
            conf["filter2"],
            conf["filter1"],
            kernel_size=(conf["ksize"], 3, 3),
            stride=(conf["pad"], 1, 1),
            padding=(conf["st"], 1, 1),
            bias=conf["bias"],
        )
        self.conv4 = Conv2plus1d(
            conf["filter1"],
            conf["in_ch"],
            kernel_size=(conf["ksize"], 3, 3),
            stride=(conf["pad"], 1, 1),
            padding=(conf["st"], 1, 1),
            bias=conf["bias"],
        )
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        # inputs: [nb, nc, ne, ny, nz] complex
        nb, nc = inputs.shape[0:2]
        x = rearrange(inputs, "b c e y z -> (b c) e y z")  # [nb*nc ne ny nz]
        x = torch.stack([torch.real(x), torch.imag(x)], dim=1)  # [nb*nc 2 ne ny nz]
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)

        x4 = self.conv4(x3) + x

        re, im = torch.chunk(x4, 2, 1)
        output = torch.complex(re.squeeze(1), im.squeeze(1))  # [nb*nc ne ny nz]
        output = rearrange(output, "(b c) e y z -> b c e y z", b=nb, c=nc)

        return output


class Conv2plus1d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: [int, int, int],
        padding: [int, int, int],
        stride: [int, int, int],
        **kwargs
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
                **kwargs
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=mid_channels,
                out_channels=out_channel,
                kernel_size=spatial_kernel_size,
                padding=spatial_padding,
                stride=spatial_stride,
                **kwargs
            ),
        )

    def forward(self, x):

        output = self.conv(x)

        return output


class Im_cnn3d_layer(nn.Module):
    def __init__(self, in_channel, filter=32):
        super(Im_cnn3d_layer, self).__init__()
        conf = dict(in_ch=in_channel, filter=filter, ksize=3, pad=1, st=1, bias=True)

        self.conv1 = Conv3d(
            conf["in_ch"],
            conf["filter"],
            kernel_size=(conf["ksize"], 3, 3),
            stride=(conf["pad"], 1, 1),
            padding=(conf["st"], 1, 1),
            bias=conf["bias"],
        )
        self.conv2 = Conv3d(
            conf["filter"],
            conf["filter"],
            kernel_size=(conf["ksize"], 3, 3),
            stride=(conf["pad"], 1, 1),
            padding=(conf["st"], 1, 1),
            bias=conf["bias"],
        )
        # self.conv3 = Conv2plus1d(conf["filter"], conf["filter"], kernel_size=(conf["ksize"], 3, 3),
        #                          stride=(conf["pad"], 1, 1), padding=(conf["st"], 1, 1), bias=conf["bias"])
        self.conv3 = Conv3d(
            conf["filter"],
            conf["in_ch"],
            kernel_size=(conf["ksize"], 3, 3),
            stride=(conf["pad"], 1, 1),
            padding=(conf["st"], 1, 1),
            bias=conf["bias"],
        )
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        # inputs: [nb, nc, ne, ny, nz] complex
        nb, nc = inputs.shape[0:2]
        x = rearrange(inputs, "b c e y z -> (b c) e y z")  # [nb*nc ne ny nz]
        x = torch.stack([torch.real(x), torch.imag(x)], dim=1)  # [nb*nc 2 ne ny nz]
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)

        x3 = self.conv3(x2) + x
        # x3 = self.relu(x3)
        #
        # x4 = self.conv4(x3) + x

        re, im = torch.chunk(x3, 2, 1)
        output = torch.complex(re.squeeze(1), im.squeeze(1))  # [nb*nc ne ny nz]
        output = rearrange(output, "(b c) e y z -> b c e y z", b=nb, c=nc)

        return output


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: [int, int, int],
        stride: [int, int, int],
        padding: [int, int, int],
        **kwargs
    ):
        super(Conv3d, self).__init__()

        kernel_size_3d = kernel_size
        padding_3d = padding
        stride_3d = stride

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size_3d,
                stride=stride_3d,
                padding=padding_3d,
                **kwargs
            )
        )

    def forward(self, x):

        output = self.conv(x)

        return output


"---------------------------------CNN Layer for mdi mapping--------------------------------"


class ImageLayer_S(nn.Module):
    def __init__(self, in_channel, out_channel, filter=64):
        super(ImageLayer_S, self).__init__()
        conf = dict(
            in_ch=in_channel,
            out_ch=out_channel,
            filter=filter,
            ksize=1,
            pad=0,
            st=1,
            bias=True,
        )

        self.mylayers = []

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["in_ch"],
                out_channels=conf["filter"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["filter"],
                out_channels=conf["filter"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["filter"],
                out_channels=conf["out_ch"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )

        self.seq = nn.Sequential(*self.mylayers)

    def forward(self, inputs):
        res = self.seq(inputs)
        # res = inputs + res
        return res


class ImageLayer_S2(nn.Module):
    def __init__(self, in_channel, out_channel, filter=64):
        super(ImageLayer_S2, self).__init__()
        conf = dict(
            in_ch=in_channel,
            out_ch=out_channel,
            filter=filter,
            ksize=3,
            pad=1,
            st=1,
            bias=True,
        )

        self.mylayers = []

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["in_ch"],
                out_channels=conf["filter"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["filter"],
                out_channels=conf["filter"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["filter"],
                out_channels=conf["out_ch"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )

        self.seq = nn.Sequential(*self.mylayers)

    def forward(self, inputs):
        im = self.seq(inputs)
        res = inputs + im
        return res


class ImageLayer_S3(nn.Module):
    def __init__(self, in_channel, out_channel, filter=64):
        super(ImageLayer_S3, self).__init__()
        conf = dict(
            in_ch=in_channel,
            out_ch=out_channel,
            filter1=32,
            filter2=filter,
            ksize=1,
            pad=0,
            st=1,
            bias=True,
        )

        self.mylayers = []

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["in_ch"],
                out_channels=conf["filter1"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["filter1"],
                out_channels=conf["filter2"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["filter2"],
                out_channels=conf["filter1"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )
        self.mylayers.append(nn.ReLU(True))

        self.mylayers.append(
            nn.Conv2d(
                in_channels=conf["filter1"],
                out_channels=conf["out_ch"],
                kernel_size=conf["ksize"],
                stride=conf["st"],
                padding=conf["pad"],
                bias=conf["bias"],
            )
        )

        self.seq = nn.Sequential(*self.mylayers)

    def forward(self, inputs):
        res = self.seq(inputs)
        # res = inputs + res
        return res
