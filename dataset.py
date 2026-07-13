import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import fft1c, ifft1c, readcfl, writecfl

"------------------------------------------------MTP multi-slice--------------------------------------------------"


class MTP_dataset_mslice(Dataset):
    def __init__(
        self,
        root: str,
        flag: str,
        vol: str,
        echo: str,
        acc: str,
        nslice: int,
        stride: int,
    ):
        super(MTP_dataset_mslice, self).__init__()
        data_root = os.path.join(root, flag, vol, echo)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

        self.ksp = readcfl(os.path.join(data_root, "ksp"))
        self.cg = readcfl(os.path.join(data_root, "ksp_grappa_" + acc))
        self.Null_2d = readcfl(os.path.join(data_root, "Null_2d"))

        self.ksp = np.transpose(
            self.ksp, (0, 3, 1, 2)
        )  # [nb ny nz nc] -> [nb nc ny nz]
        self.cg = np.transpose(self.cg, (0, 3, 1, 2))
        self.Null_2d = np.transpose(
            self.Null_2d, (0, 4, 3, 1, 2)
        )  # [nb ks ks nc1 nc2] -> [nb nc2 nc1 ks ks]

        self.nx, self.nc, self.ny, self.nz = self.ksp.shape
        self.kernel_size = self.Null_2d.shape[3]

        # multi-slice
        self.nslice = nslice
        self.stride = stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.nslice
        # print(f"start_idx: {start_idx}, end_idx: {end_idx}")

        ksp = self.ksp[start_idx:end_idx]
        cg = self.cg[start_idx:end_idx]
        Null_2d = []
        for i in range(start_idx, end_idx):
            null_2d = self.Null_2d[i]
            null_2d = torch.tensor(null_2d, dtype=torch.complex64)
            Null_2d.append(self.Null_to_image(null_2d))
        Null_2d = torch.stack(Null_2d, dim=0)

        ksp = torch.tensor(ksp, dtype=torch.complex64)
        cg = torch.tensor(cg, dtype=torch.complex64)

        sample = (ksp, cg, Null_2d)

        return sample

    def __len__(self):
        return (self.nx - self.nslice) // self.stride + 1

    def Null_to_image(self, Null_2d):
        # Null2d from kspace to image domain
        # Null2d_image size: [nb nc2 nc1 ny nz]
        padding_y1 = int((self.ny - self.kernel_size) / 2 + 0.5)
        padding_y2 = self.ny - self.kernel_size - padding_y1
        padding_z1 = int((self.nz - self.kernel_size) / 2 + 0.5)
        padding_z2 = self.nz - self.kernel_size - padding_z1
        padding = (padding_z1, padding_z2, padding_y1, padding_y2)

        Null_2d = torch.nn.functional.pad(Null_2d, padding) * torch.sqrt(
            torch.prod(torch.tensor([self.ny, self.nz]))
        )
        Null_2d = ifft1c(ifft1c(Null_2d, 2), 3)

        return Null_2d


class MTP_dataset_test_mslice(Dataset):
    def __init__(
        self, root: str, flag: str, vol: str, echo: str, acc: str, nslice: int
    ):
        super(MTP_dataset_test_mslice, self).__init__()
        data_root = os.path.join(root, flag, vol, echo)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

        self.ksp = readcfl(os.path.join(data_root, "ksp"))
        self.cg = readcfl(os.path.join(data_root, "ksp_grappa_" + acc))
        self.Null_2d = readcfl(os.path.join(data_root, "Null_2d"))

        self.ksp = np.transpose(
            self.ksp, (0, 3, 1, 2)
        )  # [nb ny nz nc] -> [nb nc ny nz]
        self.cg = np.transpose(self.cg, (0, 3, 1, 2))
        self.Null_2d = np.transpose(
            self.Null_2d, (0, 4, 3, 1, 2)
        )  # [nb ks ks nc1 nc2] -> [nb nc2 nc1 ks ks]

        self.nx, self.nc, self.ny, self.nz = self.ksp.shape
        self.kernel_size = self.Null_2d.shape[3]

        # multi-slice
        self.nslice = nslice
        self.stride = nslice  # especially for test data

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.nslice
        if end_idx > self.nx:
            end_idx = self.nx
        print(f"start_idx: {start_idx}, end_idx: {end_idx}")

        ksp = self.ksp[start_idx:end_idx]
        cg = self.cg[start_idx:end_idx]
        Null_2d = []
        for i in range(start_idx, end_idx):
            null_2d = self.Null_2d[i]
            null_2d = torch.tensor(null_2d, dtype=torch.complex64)
            Null_2d.append(self.Null_to_image(null_2d))
        Null_2d = torch.stack(Null_2d, dim=0)

        ksp = torch.tensor(ksp, dtype=torch.complex64)
        cg = torch.tensor(cg, dtype=torch.complex64)

        sample = (ksp, cg, Null_2d)

        return sample

    def __len__(self):
        return (self.nx - self.nslice + self.stride) // self.stride + 1

    def Null_to_image(self, Null_2d):
        # Null2d from kspace to image domain
        # Null2d_image size: [nb nc2 nc1 ny nz]
        padding_y1 = int((self.ny - self.kernel_size) / 2 + 0.5)
        padding_y2 = self.ny - self.kernel_size - padding_y1
        padding_z1 = int((self.nz - self.kernel_size) / 2 + 0.5)
        padding_z2 = self.nz - self.kernel_size - padding_z1
        padding = (padding_z1, padding_z2, padding_y1, padding_y2)

        Null_2d = torch.nn.functional.pad(Null_2d, padding) * torch.sqrt(
            torch.prod(torch.tensor([self.ny, self.nz]))
        )
        Null_2d = ifft1c(ifft1c(Null_2d, 2), 3)

        return Null_2d


"--------------------------------------------MTP multi-slice multi-echos-----------------------------------------------"


class MTP_dataset_echos_mslice_g(Dataset):
    def __init__(
        self,
        root: str,
        flag: str,
        vol: str,
        echo: int,
        acc: str,
        nslice: int,
        stride: int,
    ):
        super(MTP_dataset_echos_mslice_g, self).__init__()
        self.flag = flag
        self.ksp = []
        self.ksp_init = []

        # : 用于存储 g-factor 的列表
        self.gfactor = []

        for i in range(echo):
            echo_name = "Echo_" + str(i + 7)
            data_root = os.path.join(root, flag, vol, echo_name)
            assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

            ksp = readcfl(os.path.join(data_root, "ksp"))
            ksp_init = readcfl(os.path.join(data_root, "ksp_grappa_" + acc))

            # : 读取 gfactor 图 (注意这里利用了 acc 变量来匹配 'gfactor_9x')
            g_map = readcfl(os.path.join(data_root, "gfactor_" + acc))
            # g-factor 物理意义上是实数且 >= 1，这里取模以防底层库以复数格式存储
            g_map = np.abs(g_map)

            ksp = np.transpose(ksp, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]
            ksp_init = np.transpose(ksp_init, (0, 3, 1, 2))

            # : 处理 g-factor 的维度
            # ksp 的通道数 nc 通常是多线圈，但 g-factor 是一张空间图 (通道为 1)
            if len(g_map.shape) == 3:
                # 如果底层读出来是 [nb, ny, nz]，手动增加 channel 维度
                g_map = np.expand_dims(g_map, axis=1)  # -> [nb, 1, ny, nz]
            elif len(g_map.shape) == 4:
                # 如果底层读出来是 [nb, ny, nz, 1]，进行 transpose
                g_map = np.transpose(g_map, (0, 3, 1, 2))  # -> [nb, 1, ny, nz]

            self.ksp.append(ksp)
            self.ksp_init.append(ksp_init)
            self.gfactor.append(g_map)  #

        self.ksp = np.stack(self.ksp, axis=2)  # [nx, nc, ne, ny, nz]
        self.ksp_init = np.stack(self.ksp_init, axis=2)

        # : stack g-factor 列表
        self.gfactor = np.stack(self.gfactor, axis=2)  # [nx, 1, ne, ny, nz]

        echo_name = "Echo_1"
        data_root = os.path.join(root, flag, vol, echo_name)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist. "
        self.Null_2d = readcfl(os.path.join(data_root, "Null_2d"))
        self.Null_2d = np.transpose(
            self.Null_2d, (0, 4, 3, 1, 2)
        )  # [nb ks ks nc1 nc2] -> [nb nc2 nc1 ks ks]

        self.nx, self.nc, self.ne, self.ny, self.nz = self.ksp.shape
        self.kernel_size = self.Null_2d.shape[3]

        # multi-slice
        self.nslice = nslice
        self.stride = stride
        if self.flag == "test" or self.flag == "valid":
            assert (
                self.nslice == self.stride
            ), f"nslice{nslice} should be equal to stride{stride}"

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.nslice
        if end_idx > self.nx:
            end_idx = self.nx

        if self.flag == "test":
            print(f"start_idx: {start_idx}, end_idx: {end_idx}")

        ksp = self.ksp[start_idx:end_idx]  # [ns nc ne ny nz]
        ksp_init = self.ksp_init[start_idx:end_idx]  # [ns nc ne ny nz]

        # : 切片提取 g-factor
        g_map = self.gfactor[start_idx:end_idx]  # [ns 1 ne ny nz]

        Null_2d = []
        for i in range(start_idx, end_idx):
            null_2d = self.Null_2d[i]
            null_2d = torch.tensor(null_2d, dtype=torch.complex64)
            # 假设你的 self.Null_to_image 是在这个类外或者你隐去的方法
            Null_2d.append(self.Null_to_image(null_2d))
        Null_2d = torch.stack(Null_2d, dim=0)  # [ns nc nc ny nz]

        ksp = torch.tensor(ksp, dtype=torch.complex64)
        ksp_init = torch.tensor(ksp_init, dtype=torch.complex64)

        # : 转换为 float32 Tensor (因为它是实数空间的条件先验)
        g_map = torch.tensor(g_map, dtype=torch.float32)

        # : 在 sample 中返回 g_map
        sample = (ksp, ksp_init, Null_2d, g_map)

        return sample

    def __len__(self):
        if self.flag == "train":
            len = (self.nx - self.nslice) // self.stride + 1
        elif self.flag == "valid" or self.flag == "test":
            len = (self.nx - self.nslice + self.stride - 1) // self.stride + 1
            # len = (self.nx - self.nslice + self.stride) // self.stride + 1
        return len

    def Null_to_image(self, Null_2d):
        # Null2d from kspace to image domain
        # Null2d_image size: [nb nc2 nc1 ny nz]
        padding_y1 = int((self.ny - self.kernel_size) / 2 + 0.5)
        padding_y2 = self.ny - self.kernel_size - padding_y1
        padding_z1 = int((self.nz - self.kernel_size) / 2 + 0.5)
        padding_z2 = self.nz - self.kernel_size - padding_z1
        padding = (padding_z1, padding_z2, padding_y1, padding_y2)

        Null_2d = torch.nn.functional.pad(Null_2d, padding) * torch.sqrt(
            torch.prod(torch.tensor([self.ny, self.nz]))
        )
        Null_2d = ifft1c(ifft1c(Null_2d, 2), 3)

        return Null_2d


class MTP_perspective_dataset(Dataset):
    def __init__(
        self, root: str, vol: str, echo: int, acc: str, nslice: int, stride: int
    ):
        super(MTP_perspective_dataset, self).__init__()
        self.ksp = []
        self.ksp_init = []

        vol_name = vol + "_" + acc

        for i in range(echo):
            echo_name = "Echo_" + str(i + 1)
            data_root = os.path.join(root, vol_name, echo_name)
            assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

            ksp = readcfl(os.path.join(data_root, "ksp"))

            ksp = np.transpose(ksp, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]

            self.ksp.append(ksp)

        self.ksp = np.stack(self.ksp, axis=2)  # [nx, nc, ne, ny, nz]

        echo_name = "Echo_1"
        data_root = os.path.join(root, vol_name, echo_name)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist. "
        self.Null_2d = readcfl(os.path.join(data_root, "Null_2d"))
        self.Null_2d = np.transpose(
            self.Null_2d, (0, 4, 3, 1, 2)
        )  # [nb ks ks nc1 nc2] -> [nb nc2 nc1 ks ks]

        self.nx, self.nc, self.ne, self.ny, self.nz = self.ksp.shape
        self.kernel_size = self.Null_2d.shape[3]

        # multi-slice
        self.nslice = nslice
        self.stride = stride
        assert (
            self.nslice == self.stride
        ), f"nslice{nslice} should be equal to stride{stride}"

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.nslice
        if end_idx > self.nx:
            end_idx = self.nx

        print(f"start_idx: {start_idx}, end_idx: {end_idx}")

        ksp = self.ksp[start_idx:end_idx]  # [ns nc ne ny nz]
        Null_2d = []
        for i in range(start_idx, end_idx):
            null_2d = self.Null_2d[i]
            null_2d = torch.tensor(null_2d, dtype=torch.complex64)
            Null_2d.append(self.Null_to_image(null_2d))
        Null_2d = torch.stack(Null_2d, dim=0)  # [ns nc nc ny nz]

        ksp = torch.tensor(ksp, dtype=torch.complex64)

        # print('ksp: ', ksp.size())
        # print('ksp_init: ', ksp_init.size())
        # print('Null_2d: ', Null_2d.size())

        sample = (ksp, Null_2d)

        return sample

    def __len__(self):
        len = (self.nx - self.nslice + self.stride - 1) // self.stride + 1
        # len = (self.nx - self.nslice + self.stride) // self.stride + 1
        return len

    def Null_to_image(self, Null_2d):
        # Null2d from kspace to image domain
        # Null2d_image size: [nb nc2 nc1 ny nz]
        padding_y1 = int((self.ny - self.kernel_size) / 2 + 0.5)
        padding_y2 = self.ny - self.kernel_size - padding_y1
        padding_z1 = int((self.nz - self.kernel_size) / 2 + 0.5)
        padding_z2 = self.nz - self.kernel_size - padding_z1
        padding = (padding_z1, padding_z2, padding_y1, padding_y2)

        Null_2d = torch.nn.functional.pad(Null_2d, padding) * torch.sqrt(
            torch.prod(torch.tensor([self.ny, self.nz]))
        )
        Null_2d = ifft1c(ifft1c(Null_2d, 2), 3)

        return Null_2d


class MTP_dataset_echos_mslice(Dataset):
    def __init__(
        self,
        root: str,
        flag: str,
        vol: str,
        echo: int,
        acc: str,
        nslice: int,
        stride: int,
    ):
        super(MTP_dataset_echos_mslice, self).__init__()
        self.flag = flag
        self.ksp = []

        # 1. 遍历加载所有回波的 ksp
        for i in range(echo):
            echo_name = "Echo_" + str(i + 1)
            data_root = os.path.join(root, flag, vol, echo_name)
            assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

            ksp = readcfl(os.path.join(data_root, "ksp"))
            ksp = np.transpose(ksp, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]
            self.ksp.append(ksp)

        self.ksp = np.stack(self.ksp, axis=2)  # [nx, nc, ne, ny, nz]

        # 2. 加载 Echo_1 下的 Null_2d
        echo_name = "Echo_1"
        data_root = os.path.join(root, flag, vol, echo_name)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist. "
        self.Null_2d = readcfl(os.path.join(data_root, "Null_2d"))
        self.Null_2d = np.transpose(
            self.Null_2d, (0, 4, 3, 1, 2)
        )  # [nb ks ks nc1 nc2] -> [nb nc2 nc1 ks ks]

        # ==========================================
        #  新增：3. 加载 Volume 根目录下的 csm
        # ==========================================
        csm_root = os.path.join(root, flag, vol)
        assert os.path.exists(
            os.path.join(csm_root, "csm.hdr")
        ), f"csm file does not exist in {csm_root}"
        csm = readcfl(os.path.join(csm_root, "csm"))
        # 假设 CSM 的原始维度和 KSP 一样也是 [nb, ny, nz, nc]
        self.csm = np.transpose(csm, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]

        self.nx, self.nc, self.ne, self.ny, self.nz = self.ksp.shape
        self.kernel_size = self.Null_2d.shape[3]

        # multi-slice
        self.nslice = nslice
        self.stride = stride
        if self.flag == "test" or self.flag == "valid":
            assert (
                self.nslice == self.stride
            ), f"nslice{nslice} should be equal to stride{stride}"

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.nslice
        if end_idx > self.nx:
            end_idx = self.nx

        if self.flag == "test":
            print(f"start_idx: {start_idx}, end_idx: {end_idx}")

        ksp = self.ksp[start_idx:end_idx]  # [ns nc ne ny nz]

        Null_2d = []
        for i in range(start_idx, end_idx):
            null_2d = self.Null_2d[i]
            null_2d = torch.tensor(null_2d, dtype=torch.complex64)
            Null_2d.append(self.Null_to_image(null_2d))
        Null_2d = torch.stack(Null_2d, dim=0)  # [ns nc nc ny nz]

        ksp = torch.tensor(ksp, dtype=torch.complex64)

        # ==========================================
        #  新增：切片获取 CSM 并进行严格的能量归一化
        # ==========================================
        csm_slice = self.csm[start_idx:end_idx]  # [ns, nc, ny, nz]
        csm_slice = torch.tensor(csm_slice, dtype=torch.complex64)

        # 强制能量归一化：保证各个线圈平方和为 1
        sum_squares = torch.sum(
            torch.abs(csm_slice) ** 2, dim=1, keepdim=True
        )  # dim=1 是线圈维度 nc
        csm_slice = csm_slice / (torch.sqrt(sum_squares) + 1e-8)  # +1e-8 防止除以 0

        # 返回时加入 csm_slice
        sample = (ksp, Null_2d, csm_slice)

        return sample

    def __len__(self):
        if self.flag == "train":
            length = (self.nx - self.nslice) // self.stride + 1
        elif self.flag == "valid" or self.flag == "test":
            length = (self.nx - self.nslice + self.stride - 1) // self.stride + 1
        return length


# class MTP_dataset_echos_mslice(Dataset):
#     def __init__(self, root: str, flag: str, vol: str, echo: int, acc: str, nslice: int, stride: int):
#         super(MTP_dataset_echos_mslice, self).__init__()
#         self.flag = flag
#         self.ksp = []

#         # 1. 遍历加载所有回波的 ksp
#         for i in range(echo):
#             echo_name = 'Echo_' + str(i + 1)
#             data_root = os.path.join(root, flag, vol, echo_name)
#             assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

#             ksp = readcfl(os.path.join(data_root, 'ksp'))
#             ksp = np.transpose(ksp, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]
#             self.ksp.append(ksp)

#         self.ksp = np.stack(self.ksp, axis=2)  # [nx, nc, ne, ny, nz]

#         # 2. 加载 Echo_1 下的 Null_2d
#         echo_name = 'Echo_1'
#         data_root = os.path.join(root, flag, vol, echo_name)
#         assert os.path.exists(data_root), f"path '{data_root}' does not exist. "
#         self.Null_2d = readcfl(os.path.join(data_root, 'Null_2d'))
#         self.Null_2d = np.transpose(self.Null_2d, (0, 4, 3, 1, 2))  # [nb ks ks nc1 nc2] -> [nb nc2 nc1 ks ks]

#         # 3. 加载 Volume 根目录下的 csm
#         csm_root = os.path.join(root, flag, vol)
#         assert os.path.exists(os.path.join(csm_root, 'csm.hdr')), f"csm file does not exist in {csm_root}"
#         csm = readcfl(os.path.join(csm_root, 'csm'))
#         self.csm = np.transpose(csm, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]

#         self.nx, self.nc, self.ne, self.ny, self.nz = self.ksp.shape
#         self.kernel_size = self.Null_2d.shape[3]

#         # multi-slice 设置
#         self.nslice = nslice

#         # =========================================================================
#         #  核心修改：强制重置训练时的 Stride，完美适配 train.py 的 3 层截取法
#         # =========================================================================
#         if self.flag == 'train':
#             # 外部传进来的可能是 stride=1，我们强制覆盖它！
#             # 对于 nslice=7，每次滑动 3 层，恰好能覆盖所有我们要监督的“中间 3 层”
#             self.stride = 3
#         elif self.flag == 'valid' or self.flag == 'test':
#             # 测试集保持互不重叠，或者你也可以在这里用重叠推理 (看你 eval.py 怎么写)
#             self.stride = self.nslice
#             assert self.nslice == self.stride, f"nslice{nslice} should be equal to stride{stride}"

#     # __getitem__ 保持原样完全不动
#     def __getitem__(self, idx):
#         start_idx = idx * self.stride
#         end_idx = start_idx + self.nslice

#         # 处理边界溢出
#         if end_idx > self.nx:
#             end_idx = self.nx

#         if self.flag == 'test':
#             # print(f"start_idx: {start_idx}, end_idx: {end_idx}")
#             pass

#         # 1. 提取 ksp
#         ksp = self.ksp[start_idx:end_idx] # [ns nc ne ny nz]

#         # 2. 提取并转换 Null_2d
#         Null_2d = []
#         for i in range(start_idx, end_idx):
#             null_2d = self.Null_2d[i]
#             null_2d = torch.tensor(null_2d, dtype=torch.complex64)
#             Null_2d.append(self.Null_to_image(null_2d))
#         Null_2d = torch.stack(Null_2d, dim=0) # [ns nc nc ny nz]

#         ksp = torch.tensor(ksp, dtype=torch.complex64)

#         # ==========================================
#         #  3. CSM 切片获取与严格能量归一化
#         # ==========================================
#         csm_slice = self.csm[start_idx:end_idx] # [ns, nc, ny, nz]
#         csm_slice = torch.tensor(csm_slice, dtype=torch.complex64)

#         # 强制能量归一化：保证各个线圈平方和为 1
#         sum_squares = torch.sum(torch.abs(csm_slice)**2, dim=1, keepdim=True) # dim=1 是线圈维度 nc
#         csm_slice = csm_slice / (torch.sqrt(sum_squares) + 1e-8)

#         # 4. 打包返回
#         sample = (ksp, Null_2d, csm_slice)

#         return sample

#     def __len__(self):
#         #  修正长度计算公式，确保最后不够 nslice 的切片也能被取到（即使要补 0 或扔掉一部分）
#         if self.flag == 'train':
#             # 如果是重叠滑窗，计算能滑动多少次
#             length = (self.nx - self.nslice) // self.stride + 1
#         elif self.flag == 'valid' or self.flag == 'test':
#             length = (self.nx - self.nslice + self.stride - 1) // self.stride + 1
#         return length


def Null_to_image(self, Null_2d):
    padding_y1 = int((self.ny - self.kernel_size) / 2 + 0.5)
    padding_y2 = self.ny - self.kernel_size - padding_y1
    padding_z1 = int((self.nz - self.kernel_size) / 2 + 0.5)
    padding_z2 = self.nz - self.kernel_size - padding_z1
    padding = (padding_z1, padding_z2, padding_y1, padding_y2)

    Null_2d = torch.nn.functional.pad(Null_2d, padding) * torch.sqrt(
        torch.prod(torch.tensor([self.ny, self.nz]))
    )
    Null_2d = ifft1c(ifft1c(Null_2d, 2), 3)

    return Null_2d


def collate_fn_keep_12echo(batch):
    ksp_list = []
    Null_2d_list = []
    csm_list = []

    for unit in batch:
        ksp = unit[0]  # [ns, nc, 12, ny, nz]
        Null_2d = unit[1]  # [ns, nc2, nc1, ny, nz]
        csm = unit[2]  # [ns, nc, ny, nz]  <-  接收 csm

        ksp_list.append(ksp)
        Null_2d_list.append(Null_2d)
        csm_list.append(csm)

    # 堆叠 Batch 维度
    ksps = torch.stack(ksp_list, dim=0)  # [batch, ns, nc, 12, ny, nz]
    Null_2ds = torch.stack(Null_2d_list, dim=0)  # [batch, ns, nc2, nc1, ny, nz]
    csms = torch.stack(csm_list, dim=0)  # [batch, ns, nc, ny, nz] <-  堆叠 csm

    return {"ksp": ksps, "Null_2d": Null_2ds, "csm": csms}
