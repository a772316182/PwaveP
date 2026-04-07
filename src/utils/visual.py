import matplotlib.pyplot as plt
import numpy
import torch
from pygsp2.graphs import Graph

from src.utils.save_utils import pre_check_path


def view_wavelet_coffs(ori_wavelet_coffs, adv_wavelet_coffs, filename="demo.png"):
    if isinstance(ori_wavelet_coffs, torch.Tensor):
        ori_wavelet_coffs = ori_wavelet_coffs.clone().detach().cpu().numpy()
        adv_wavelet_coffs = adv_wavelet_coffs.clone().detach().cpu().numpy()

    assert isinstance(ori_wavelet_coffs, numpy.ndarray)
    assert isinstance(adv_wavelet_coffs, numpy.ndarray)

    fig, axs = plt.subplots(6, 6, figsize=(30, 20))
    for i in range(6):
        for j in range(3):
            ax = axs[i, j]
            data = ori_wavelet_coffs[:, j, i]  # 获取对应的数据
            ax.plot(data, color="green")
            ax.set_title(f"clean WAVELET {i + 1} on AXIS {j + 1}")
        for j in range(3):
            ax = axs[i, j + 3]
            data = adv_wavelet_coffs[:, j, i]  # 获取对应的数据
            ax.plot(data, color="red")
            ax.set_title(f"adv WAVELET {i + 1} on AXIS {j + 1}")

    # 设置相同的轴范围以便于比较
    min_limit = min(ori_wavelet_coffs.min(), adv_wavelet_coffs.min())
    max_limit = max(ori_wavelet_coffs.max(), adv_wavelet_coffs.max())
    for ax in axs.flatten():
        ax.set_ylim(min_limit, max_limit)

    plt.tight_layout()
    plt.savefig(pre_check_path(filename), dpi=300)


def view_wavelet(Nf_size, filtered_signal, graph: Graph, title: str = ""):
    fig = plt.figure(figsize=(10, 70))
    for q in range(Nf_size):
        ax = fig.add_subplot(Nf_size, 1, q + 1, projection="3d")
        filtered_j_ = filtered_signal[:, q]
        _ = graph.plot(
            filtered_j_,
            ax=ax,
            title=f"{title} > WAVELET {q}",
            alphav=0.2,
            alphan=0.2,
        )
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def view_signals(
    ori_signals,
    adv_signals,
    filename="demo.png",
    sort_signal=False,
    titles: numpy.ndarray = None,
    title: str = None,
):
    if isinstance(ori_signals, torch.Tensor):
        ori_signals = ori_signals.clone().detach().cpu().numpy()
        adv_signals = adv_signals.clone().detach().cpu().numpy()

    assert isinstance(ori_signals, numpy.ndarray)
    assert isinstance(adv_signals, numpy.ndarray)

    rows = ori_signals.shape[0]

    assert rows < 100, "too many rows"

    fig = plt.figure(figsize=(10, 3 * rows))  # 调整整个图表的大小以适应多个子图
    if title:
        fig.suptitle(title)

    for i in range(rows):
        ax1 = fig.add_subplot(rows, 2, i * 2 + 1)
        ax2 = fig.add_subplot(rows, 2, i * 2 + 2)
        ori_signals_i_ = ori_signals[i, :]
        adv_signals_i_ = adv_signals[i, :]

        if sort_signal:
            ori_signals_i_.sort()
            adv_signals_i_.sort()

        ax1.plot(
            numpy.arange(ori_signals.shape[1]), ori_signals_i_, label="Original", c="b"
        )
        ax2.plot(
            numpy.arange(adv_signals.shape[1]),
            adv_signals_i_,
            label="Adversarial",
            c="r",
        )
        if titles:
            assert len(titles) == 2
            ax1.set_title(f"row {i}: " + str(titles[0]))
            ax2.set_title(f"row {i}: " + str(titles[1]))
        else:
            ax1.set_title(f"row {i}: " + "Original")
            ax2.set_title(f"row {i}: " + "Adversarial")
        # 设置相同的轴范围以便于比较
        min_limit = min(ori_signals.min(), adv_signals.min())
        max_limit = max(ori_signals.max(), adv_signals.max())
        for ax in [ax1, ax2]:
            ax.set_ylim(min_limit, max_limit)

    plt.tight_layout()
    plt.savefig(pre_check_path(filename), dpi=300)
    plt.close()


def view_point_clouds(ori_point_clouds, adv_point_clouds, filename="demo.png"):
    if isinstance(ori_point_clouds, torch.Tensor):
        ori_point_clouds = ori_point_clouds.clone().detach().cpu().numpy()
        adv_point_clouds = adv_point_clouds.clone().detach().cpu().numpy()
    elif isinstance(ori_point_clouds, list):
        ori_point_clouds = torch.stack(ori_point_clouds, dim=0).numpy()
        adv_point_clouds = torch.stack(adv_point_clouds, dim=0).numpy()

    assert isinstance(ori_point_clouds, numpy.ndarray)
    assert isinstance(adv_point_clouds, numpy.ndarray)

    # 计算行数，每行显示两个点云图
    rows = ori_point_clouds.shape[0]

    fig = plt.figure(figsize=(15, 5 * rows))  # 调整整个图表的大小以适应多个子图

    for i in range(rows):
        ax1 = fig.add_subplot(rows, 2, i * 2 + 1, projection="3d")
        ax2 = fig.add_subplot(rows, 2, i * 2 + 2, projection="3d")

        # 绘制原始点云
        ax1.scatter(
            ori_point_clouds[i, :, 0],
            ori_point_clouds[i, :, 1],
            ori_point_clouds[i, :, 2],
            color="b",
            alpha=0.3,
        )
        # 绘制对抗点云
        ax2.scatter(
            adv_point_clouds[i, :, 0],
            adv_point_clouds[i, :, 1],
            adv_point_clouds[i, :, 2],
            color="r",
            alpha=0.3,
        )

        # 设置相同的轴范围以便于比较
        min_limit = min(ori_point_clouds.min(), adv_point_clouds.min())
        max_limit = max(ori_point_clouds.max(), adv_point_clouds.max())
        for ax in [ax1, ax2]:
            ax.set_xlim(min_limit, max_limit)
            ax.set_ylim(min_limit, max_limit)
            ax.set_zlim(min_limit, max_limit)

    plt.tight_layout()
    plt.savefig(pre_check_path(filename), dpi=300)
    plt.close()
