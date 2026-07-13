import os

import numpy as np
import torch
from utils import ifft2c, readcfl, writecfl


def calculate_delta_S(s2, eps=1e-8):

    # nb, nc, na, n2, ny, nz = s2.size() # [nb, nc, (na,) ne-1, ny, nz]

    # a = torch.zeros([nb, ny, nz], dtype=s2.dtype, device=s2.device)
    # b = torch.zeros([nb, ny, nz], dtype=s2.dtype, device=s2.device)

    # for iter_coil in range(nc):
    #     for iter_fa in range(na):
    #         for iter_echo in range(n2 - 1):
    #             a += (torch.conj(s2[:, iter_coil, iter_fa, iter_echo, :, :])
    #                               * s2[:, iter_coil, iter_fa, iter_echo + 1, :, :])
    #             b += torch.abs(s2[:, iter_coil, iter_fa, iter_echo, :, :]) ** 2
    a = torch.sum(
        torch.conj(s2[:, :, :, :-1, :, :]) * s2[:, :, :, 1:, :, :], dim=(1, 2, 3)
    )
    b = torch.sum(torch.abs(s2[:, :, :, :-1, :, :]) ** 2, dim=(1, 2, 3))

    delta_S = a / (b + eps)

    return delta_S


def calculate_delta_S_echo(s2, echo, eps=1e-8):
    # signal ratio between neighboring echoes

    a = torch.sum(
        torch.conj(s2[:, :, :, echo, :, :]) * s2[:, :, :, echo + 1, :, :], dim=(1, 2)
    )
    b = torch.sum(torch.abs(s2[:, :, :, echo, :, :]) ** 2, dim=(1, 2))

    # a = a.to(dtype=torch.complex128)
    # b = b.to(dtype=torch.complex128)

    delta_S = a / (b + eps)

    return delta_S


# def reconstruct_v(v11, v12, delta_s, s2_shape):
#     # v11 [nb nc ny nz]
#     # delta_S [nb ny nz]
#
#     _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]
#
#     v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
#
#     if na == 1:
#         v[:, :, 0, 0, :, :] = v12
#         delta_s = delta_s.unsqueeze(1) # [nb 1 ny nz]
#         for iter_echo in range(n2 - 1):
#             v[:, :, 0, iter_echo + 1, :, :] = v[:, :, 0, iter_echo, :, :].clone() * delta_s
#
#         v = torch.cat((v11.unsqueeze(2).unsqueeze(2), v), dim=3)
#
#         return v


def reconstruct_v(v11, v12, delta_S, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)

    if na == 1:
        v[:, :, 0, 0, :, :] = v12
        delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
        for iter_echo in range(n2 - 1):
            v[:, :, 0, iter_echo + 1, :, :] = (
                v[:, :, 0, iter_echo, :, :].clone() * delta_s
            )  # [:,:,iter_echo,:,:]

        v = torch.cat((v11.unsqueeze(2).unsqueeze(2), v), dim=3)

        return v


# def reconstruct_v_echo(v, delta_s, s2_shape):
#     # v11 [nb nc (na), ne, ny nz]
#     # delta_S [nb ny nz
#     _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]
#
#     v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
#
#     if na == 1:
#         v[:, :, 0, 0, :, :] = v12
#         delta_s = delta_s.unsqueeze(1)  # [nb 1 ny nz]
#         for iter_echo in range(n2 - 1):
#             v[:, :, 0, iter_echo + 1, :, :] = v[:, :, 0, iter_echo, :, :].clone() * delta_s
#
#         v = torch.cat((v11.unsqueeze(2).unsqueeze(2), v), dim=3)
#
#         return v


if __name__ == "__main__":
    echo = 6
    data_path = "/data0/zhaojiaying/MTP_datasets/train/vol1"
    # assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

    # delta_S = readcfl(os.path.join(data_root, 'delta_S'))  # [nx ny nz]
    # delta_S = torch.tensor(delta_S, dtype=torch.complex64)  # [nx ny nz]

    ksp_echos = []

    for i in range(echo):
        echo_name = "Echo_" + str(i + 1)
        data_root = os.path.join(data_path, echo_name)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

        ksp = readcfl(os.path.join(data_root, "ksp"))

        ksp = np.transpose(ksp, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]

        ksp_echos.append(ksp)

    ksp = np.stack(ksp_echos, axis=2)  # [nx, nc, ne, ny, nz]

    ksp = torch.tensor(ksp, dtype=torch.complex64)  # [nx, nc, ne, ny, nz]
    image = torch.zeros_like(ksp, dtype=torch.complex64)

    nx = ksp.shape[0]

    for i in range(nx):
        ksp_ = ksp[i, :, :, :, :]
        image[i, :, :, :, :] = ifft2c(ksp[i, :, :, :, :])

    image = image.unsqueeze(2)  # [nx nc ne ny nz] -> # [nx nc 1 ne ny nz]
    s2 = image[:, :, :, 1:, :, :]  # [nx nc 1 ne-1 ny nz]
    delta_S = calculate_delta_S(s2, eps=1e-8)
    writecfl("delta_S", delta_S.numpy())

    v11 = image[:, :, 0, 0, :, :]  # [nx nc ny nz]
    image_ = reconstruct_v(v11, delta_S, s2.shape)
    writecfl("image_", image_.numpy())


import os

import numpy as np
import torch
from utils import ifft2c, readcfl, writecfl


def calculate_delta_S_ratio(s2, echo, eps=1e-8):
    # signal ratio between neighboring echoes
    # s2 : [nb nc na n2 ny nz]

    a = torch.sum(
        torch.conj(s2[:, :, :, echo, :, :]) * s2[:, :, :, echo + 1, :, :], dim=(1, 2)
    )
    b = torch.sum(torch.abs(s2[:, :, :, echo, :, :]) ** 2, dim=(1, 2))

    # a = a.to(dtype=torch.complex128)
    # b = b.to(dtype=torch.complex128)

    delta_S = a / (b + eps)

    return delta_S


def calculate_delta_S(s2, eps=1e-8):
    # s2: [nb, nc, na, ne, ny, nz]

    # 最优匹配滤波复数内积 (包含幅值衰减与相位累积)
    a = torch.sum(
        torch.conj(s2[:, :, :, :-1, :, :]) * s2[:, :, :, 1:, :, :], dim=(1, 2, 3)
    )
    b = torch.sum(torch.abs(s2[:, :, :, :-1, :, :]) ** 2, dim=(1, 2, 3))

    delta_S_complex = a / (b + eps)

    # ：提取复数的幅值，这才是纯粹的 T2* 信号衰减比例！
    delta_S_mag = torch.abs(delta_S_complex)

    # 物理约束：衰减比例理论上在 0~1 之间，clamp 截断可以有效屏蔽纯背景空气中的极端除法噪声
    delta_S_mag = torch.clamp(delta_S_mag, 0.0, 1.2)

    return delta_S_mag


def calculate_R2(s2, eps=1e-8):
    # s2: [nb, nc, na, ne, ny, nz]

    a = torch.sum(torch.conj(s2[:, :, 1, :, :, :]) * s2[:, :, 0, :, :, :], dim=(1, 2))
    b = torch.sum(torch.abs(s2[:, :, 1, :, :, :]) ** 2, dim=(1, 2))

    R2_complex = a / (b + eps)

    # 尽管相位基本抵消，但也应取幅值，确保输出为绝对正实数
    R2_mag = torch.abs(R2_complex)

    # 物理约束：R 通常在 0.5 到 3 之间
    R2_mag = torch.clamp(R2_mag, 0.0, 5.0)

    return R2_mag


def calculate_R2_ratio(s2, echo, eps=1e-8):
    # signal ratio between each flip-angle
    # s2 : [nb nc na n2 ny nz]

    a = torch.sum(torch.conj(s2[:, :, 1, echo, :, :]) * s2[:, :, 0, echo, :, :], dim=1)
    b = torch.sum(torch.abs(s2[:, :, 1, echo, :, :]) ** 2, dim=1)

    R2 = a / (b + eps)

    return R2


def reconstruct_v_deltaS(v11, v12, delta_S, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)

    if na == 1:
        v[:, :, 0, 0, :, :] = v12
        delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
        for iter_echo in range(n2 - 1):
            v[:, :, 0, iter_echo + 1, :, :] = (
                v[:, :, 0, iter_echo, :, :].clone() * delta_s
            )  # [:,:,iter_echo,:,:]

        v = torch.cat((v11.unsqueeze(2).unsqueeze(2), v), dim=3)

        return v


def reconstruct_v_echo_deltaS(v11, v12, delta_s, s2_shape):
    # v11 [nb nc (na), ne, ny nz]
    # delta_S [nb 4 ny nz]
    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)

    if na == 1:
        v[:, :, 0, 0, :, :] = v12
        delta_s = delta_s.unsqueeze(1)  # [nb 1 ny nz]
        for iter_echo in range(n2 - 1):
            v[:, :, 0, iter_echo + 1, :, :] = (
                v[:, :, 0, iter_echo, :, :].clone() * delta_s[:, :, iter_echo, :, :]
            )

        v = torch.cat((v11.unsqueeze(2).unsqueeze(2), v), dim=3)

        return v


def reconstruct_v_R2(v1, v21, R2, s2_shape):
    # v21 [nb nc ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, na, ne-1, ny, nz]
    v = torch.zeros(s2_shape, dtype=v21.dtype, device=v21.device)

    R2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 0, iter_echo, :, :] = v1[:, :, 0, iter_echo, :, :] * R2

    v = torch.cat((v21.unsqueeze(2).unsqueeze(2), v), dim=3)  # [nb, nc, na, ne, ny, nz]

    return v


def reconstruct_v_echo_R2(v1, v21, R2, s2_shape):
    # v21 [nb nc ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, na, ne-1, ny, nz]
    v = torch.zeros(s2_shape, dtype=v21.dtype, device=v21.device)

    R2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 0, iter_echo, :, :] = (
            v1[:, :, 0, iter_echo, :, :] * R2[:, :, iter_echo, :, :]
        )

    v = torch.cat((v21.unsqueeze(2).unsqueeze(2), v), dim=3)  # [nb, nc, na, ne, ny, nz]

    return v


def systhesis_v(v11, v12, v21, delta_S, R2, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
    v1 = torch.stack((v11, v21), dim=2)  # [nb nc na ny nz]

    # recon FA1 with delta_S
    v[:, :, 0, 0, :, :] = v12
    delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
    for iter_echo in range(n2 - 1):
        v[:, :, 0, iter_echo + 1, :, :] = (
            v[:, :, 0, iter_echo, :, :].clone() * delta_s
        )  # [:,:,iter_echo,:,:]

    # recon FA2 with R2
    R2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 1, iter_echo, :, :] = v[:, :, 0, iter_echo, :, :].clone() * R2

    v = torch.cat((v1.unsqueeze(3), v), dim=3)

    return v


def systhesis_v2(im, delta_S, R2, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v11 = im[:, :, 0, 0, :, :]
    v22 = im[:, :, 1, 1, :, :]
    v21 = im[:, :, 1, 0, :, :]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
    v1 = torch.stack((v11, v21), dim=2)  # [nb nc na ny nz]

    # recon FA2 with delta_S
    v[:, :, 1, 0, :, :] = v22
    delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
    for iter_echo in range(n2 - 1):
        v[:, :, 1, iter_echo + 1, :, :] = (
            v[:, :, 1, iter_echo, :, :].clone() * delta_s
        )  # [:,:,iter_echo,:,:]

    # recon FA1 with R2
    R2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 0, iter_echo, :, :] = v[:, :, 1, iter_echo, :, :].clone() * R2

    v = torch.cat((v1.unsqueeze(3), v), dim=3)

    return v


def systhesis_v3(im, delta_S, R2, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v11 = im[:, :, 0, 0, :, :]
    v12 = im[:, :, 0, 1, :, :]
    v21 = im[:, :, 1, 0, :, :]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
    v1 = torch.stack((v11, v21), dim=2)  # [nb nc na ny nz]

    # recon FA1 with delta_S
    v[:, :, 0, 0, :, :] = v12
    delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
    for iter_echo in range(n2 - 1):
        v[:, :, 0, iter_echo + 1, :, :] = (
            v[:, :, 0, iter_echo, :, :].clone() * delta_s
        )  # [:,:,iter_echo,:,:]

    # recon FA2 with R2
    R2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 1, iter_echo, :, :] = v[:, :, 0, iter_echo, :, :].clone() / (R2 + 1e-8)

    v = torch.cat((v1.unsqueeze(3), v), dim=3)

    return v


def systhesis_v_deltaS(im, delta_S, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v11 = im[:, :, 0, 0, :, :]
    v12 = im[:, :, 0, 1, :, :]
    v21 = im[:, :, 1, 0, :, :]
    v22 = im[:, :, 1, 1, :, :]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
    v1 = torch.stack((v11, v21), dim=2)  # [nb nc na ny nz]

    # recon FA1 with delta_S
    v[:, :, 0, 0, :, :] = v12
    delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
    for iter_echo in range(n2 - 1):
        v[:, :, 0, iter_echo + 1, :, :] = (
            v[:, :, 0, iter_echo, :, :].clone() * delta_s
        )  # [:,:,iter_echo,:,:]

    # recon FA2 with delta_S
    v[:, :, 1, 0, :, :] = v22
    delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
    for iter_echo in range(n2 - 1):
        v[:, :, 1, iter_echo + 1, :, :] = (
            v[:, :, 1, iter_echo, :, :].clone() * delta_s
        )  # [:,:,iter_echo,:,:]

    v = torch.cat((v1.unsqueeze(3), v), dim=3)

    return v


def systhesis_v_R(im, R2, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v11 = im[:, :, 0, 0, :, :]
    v12 = im[:, :, 0, 1, :, :]
    v21 = im[:, :, 1, 0, :, :]
    v22 = im[:, :, 1, 1, :, :]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
    v1 = torch.stack((v11, v21), dim=2)  # [nb nc na ny nz]

    # recon FA2 with im
    v[:, :, 1, 0, :, :] = v22
    for iter_echo in range(n2 - 1):
        v[:, :, 1, iter_echo + 1, :, :] = im[:, :, 1, iter_echo + 2, :, :]

    # recon FA1 with R2
    R2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 0, iter_echo, :, :] = v[:, :, 1, iter_echo, :, :].clone() * R2

    # # recon FA1 with im
    # v[:, :, 0, 0, :, :] = v12
    # for iter_echo in range(n2 - 1):
    #     v[:, :, 0, iter_echo + 1, :, :] = im[:, :, 0, iter_echo + 2, :, :]
    #
    # # recon FA2 with R2
    # R2 = R2.unsqueeze(1) # [nb 1 4 ny nz]
    # for iter_echo in range(n2):
    #     v[:, :, 1, iter_echo, :, :] = v[:, :, 0, iter_echo, :, :].clone() / (R2 + 1e-8)

    v = torch.cat((v1.unsqueeze(3), v), dim=3)

    return v


def systhesis_v_R2(im, R2, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v11 = im[:, :, 1, 0, :, :] * R2.unsqueeze(1)
    v12 = im[:, :, 0, 1, :, :]
    # v21 = im[:, :, 1, 0, :, :]
    v21 = im[:, :, 1, 0, :, :]
    v22 = im[:, :, 1, 1, :, :]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
    v1 = torch.stack((v11, v21), dim=2)  # [nb nc na ny nz]

    # recon FA2 with im
    v[:, :, 1, 0, :, :] = v22
    for iter_echo in range(n2 - 1):
        v[:, :, 1, iter_echo + 1, :, :] = im[:, :, 1, iter_echo + 2, :, :]

    # recon FA1 with R2
    R2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 0, iter_echo, :, :] = v[:, :, 1, iter_echo, :, :].clone() * R2

    v = torch.cat((v1.unsqueeze(3), v), dim=3)

    return v


def systhesis_v_echo(v11, v12, v21, delta_S, R2, s2_shape):
    # v11 [nb nc ny nz]
    # delta_S [nb 4 ny nz]
    # R2 [nb 5 ny nz]

    _, nc, na, n2, _, _ = s2_shape  # [nb, nc, (na,) ne-1, ny, nz]

    v = torch.zeros(s2_shape, dtype=v11.dtype, device=v11.device)
    v1 = torch.stack((v11, v21), dim=2)  # [nb nc na ny nz]

    # recon FA1 with delta_S
    v[:, :, 0, 0, :, :] = v12
    delta_s = delta_S.unsqueeze(1)  # [nb 1 ny nz, 4]
    for iter_echo in range(n2 - 1):
        v[:, :, 0, iter_echo + 1, :, :] = (
            v[:, :, 0, iter_echo, :, :].clone() * delta_s[:, :, iter_echo, :, :]
        )

    # recon FA2 with R2
    r2 = R2.unsqueeze(1)  # [nb 1 4 ny nz]
    for iter_echo in range(n2):
        v[:, :, 1, iter_echo, :, :] = v[:, :, 0, iter_echo, :, :].clone() / (
            r2[:, :, iter_echo, :, :] + 1e-8
        )

    v = torch.cat((v1.unsqueeze(3), v), dim=3)

    return v


if __name__ == "__main__":
    echo = 12
    data_path = "/data0/zhaojiaying/MTP_datasets/train/vol1"
    # assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

    # delta_S = readcfl(os.path.join(data_root, 'delta_S'))  # [nx ny nz]
    # delta_S = torch.tensor(delta_S, dtype=torch.complex64)  # [nx ny nz]

    ksp_echos = []

    for i in range(echo):
        echo_name = "Echo_" + str(i + 1)
        data_root = os.path.join(data_path, echo_name)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist. "

        ksp = readcfl(os.path.join(data_root, "ksp"))

        ksp = np.transpose(ksp, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]

        ksp_echos.append(ksp)

    ksp = np.stack(ksp_echos, axis=2)  # [nx, nc, ne, ny, nz]

    ksp = torch.tensor(ksp, dtype=torch.complex64)  # [nx, nc, ne, ny, nz]
    image = torch.zeros_like(ksp, dtype=torch.complex64)

    nx = ksp.shape[0]

    for i in range(nx):
        ksp_ = ksp[i, :, :, :, :]
        image[i, :, :, :, :] = ifft2c(ksp[i, :, :, :, :])

    # image = image.unsqueeze(2) # [nx nc ne ny nz] -> # [nx nc 1 ne ny nz]
    im = torch.chunk(image, 2, dim=2)
    im = torch.stack(im, dim=2)  # [nb nc na n2+1 ny nz]
    s2 = image[:, :, :, 1:, :, :]  # [nx nc 1 ne-1 ny nz]
    delta_S = calculate_delta_S(s2, eps=1e-8)
    writecfl("delta_S", delta_S.numpy())

    R2 = calculate_R2(s2, eps=1e-8)
    writecfl("R2", R2.numpy())

    # v11 = image[:,:,0,0,:,:] # [nx nc ny nz]
    # image_ = reconstruct_v(v11, delta_S, s2.shape)
    # writecfl('image_', image_.numpy())
