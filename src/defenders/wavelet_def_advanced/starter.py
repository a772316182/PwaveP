import torch
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.defenders.wavelet_def.local_risk import calculate_local_risk
from src.utils.model_eval import PerfTrackVal
from .WaveletTrans import WaveletTransformUtil


def start_wavelet_gard_wrt_coff_energy(
        loader: DataLoader, device: str, model: Module,
        filter_name: str = "mexicanhat",
        k_neighbors: int = 20,
        num_wavelets: int = 12,
        k_bands_to_filter: int = 1,
        attenuation_factor: float = 0.,
        drop_rate: float = 0.01,
        filter_rate: float = 0.09,
        geo_risk_weight: float = 1.0,
        spectral_risk_weight: float = 1.0,
        grad_smooth_step: int = 5,
        grad_smooth_eps: float = 0.05
):
    """
    最终版防御流程，使用封装好的、带平滑和组合梯度的风险计算方法。
    """
    logger.info("Starting FINALIZED wavelet gard with SMOOTHED COMBINED GRADIENT")

    all_defend_pc = []
    for batch_idx, data_item in enumerate(tqdm(loader)):
        clean_input_item = {}

        batched_adv_pc = data_item["attacked_data"].to(device).float()
        # 在新的逻辑中，标签在风险计算函数内部处理（使用模型预测），
        # 但我们仍需一个标签来满足函数接口，这里使用真实标签
        clean_label = torch.squeeze(data_item["real_label"]).to(device)

        batch_size = batched_adv_pc.shape[0]
        num_points = batched_adv_pc.shape[1]
        num_to_remove = int(num_points * drop_rate)
        num_to_repair = int(num_points * filter_rate)

        # 步骤 1: 初始化小波变换工具
        wavelet = WaveletTransformUtil(
            batched_pc=batched_adv_pc,
            filter_name=filter_name,
            num_wavelets=num_wavelets,
            k_neighbors=k_neighbors
        )

        model_risk, wavelet_coeffs, grad_wrt_wavelet_coeffs = wavelet.get_coeffs_and_grads_advanced(
            target_model=model, signal_batch=batched_adv_pc.detach().clone(),
            label=clean_label,
            step=grad_smooth_step, eps=grad_smooth_eps
        )

        # 步骤 2: 结合几何风险，得到最终风险
        geo_risk = calculate_local_risk(batched_adv_pc, k=k_neighbors, mode="norm").to(device)
        risk = spectral_risk_weight * model_risk + geo_risk_weight * geo_risk

        # 步骤 3: 排序和分层
        sorted_indices = torch.argsort(risk, dim=1, descending=True)
        high_risk_indices = sorted_indices[:, :num_to_remove]
        mid_risk_indices = sorted_indices[:, num_to_remove: num_to_remove + num_to_repair]

        # 步骤 4: 滤波 (Purification Step 1) - 使用梯度引导的智能滤波
        purified_coeffs = wavelet_coeffs.clone()
        if num_to_repair > 0:
            # 计算每个频段的全局梯度能量
            grad_energy_per_band = torch.norm(grad_wrt_wavelet_coeffs, p=2, dim=(0, 1, 2))
            # 找到能量最高（最可疑）的k个频段的索引
            _, high_energy_bands_indices = torch.topk(grad_energy_per_band, k=k_bands_to_filter)

            logger.info(f"Batch {batch_idx}, filtering bands: {high_energy_bands_indices.cpu().numpy()}")

            # 只对中风险点在这些高能量频段上的系数进行衰减
            for i in range(batch_size):
                points_to_filter = mid_risk_indices[i]
                purified_coeffs[i, points_to_filter, :, high_energy_bands_indices] *= attenuation_factor

        filtered_adv_pc = wavelet.inverse_transform(purified_coeffs)

        # 步骤 6: 删点 (Purification Step 2)
        mask = torch.ones_like(risk, dtype=torch.bool, device='cpu')
        mask.scatter_(1, high_risk_indices.cpu(), False)
        res = filtered_adv_pc[mask].view(batch_size, -1, 3)

        clean_input_item["res"] = res.detach().clone().cpu().float()
        # 注意评估时使用的是真实标签
        clean_input_item["label"] = clean_label.detach().clone().cpu()
        all_defend_pc.append(clean_input_item)

    # 步骤 7: 评估防御效果
    perf = PerfTrackVal()
    for j, data_batch in enumerate(all_defend_pc):
        purified_pc = data_batch["res"].to(device)
        label = data_batch["label"].to(device)
        # 确保模型在前向传播时不会计算梯度
        with torch.no_grad():
            perf.update(model.forward(purified_pc)["logit"], label)
    res = perf.agg()
    logger.info("Defenced accuracy:")
    logger.info(res)
    return res
