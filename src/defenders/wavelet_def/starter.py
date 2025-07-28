import torch
import tqdm
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cvar import advanced_cvar
from src.utils.model_eval import PerfTrackVal
from .WaveletTrans import WaveletTransformUtil
from .WaveletTransCheb import WaveletTransformUtilCheb


def start_wavelet_gard_wrt_coff_energy(
        loader: DataLoader, device: str, model: Module,
        filter_name: str = "mexicanhat",
        k_neighbors: int = 20,
        num_wavelets: int = 12,
        k_bands_to_filter: int = 1,
        attenuation_factor: float = 0.1,
        drop_rate: float = 0.01,
        filter_rate: float = 0.09,
        key_clean_pc: str = "real_data",
        key_adv_pc: str = "attacked_data",
        key_clean_label: str = "real_label",
        key_adv_label: str = "target_label",
        break_on: int = 10000,
        using_chebyshev: bool = False
):
    logger.info("Starting wavelet_gard_wrt_coff_energy")

    all_defend_pc = []
    for batch_idx, data_item in enumerate(tqdm(loader)):
        if batch_idx > break_on:
            break
        clean_input_item = {}

        batched_adv_pc = data_item[key_adv_pc].to(device).float()
        batched_clean_pc = data_item[key_clean_pc].to(device).float()
        adv_label = torch.squeeze(data_item[key_adv_label]).to(device)
        clean_label = torch.squeeze(data_item[key_clean_label]).to(device)

        batch_size = batched_adv_pc.shape[0]
        num_points = batched_adv_pc.shape[1]
        num_to_remove = int(num_points * drop_rate)
        num_to_repair = int(num_points * filter_rate)

        cvar_res = advanced_cvar(data_batch={
            "pc": batched_adv_pc.detach().clone(),
            "label": adv_label
        }, model=model, drop_rate=drop_rate)

        extracted_risk = torch.from_numpy(cvar_res["risk"])
        sorted_indices = torch.argsort(extracted_risk, dim=1, descending=True)

        high_risk_indices = sorted_indices[:, :num_to_remove]
        mid_risk_indices = sorted_indices[:, num_to_remove: num_to_remove + num_to_repair]

        if using_chebyshev:
            wavelet = WaveletTransformUtilCheb(
                batched_pc=batched_adv_pc,
                filter_name=filter_name,
                num_wavelets=num_wavelets,
                k_neighbors=k_neighbors
            )
        else:
            wavelet = WaveletTransformUtil(
                batched_pc=batched_adv_pc,
                filter_name=filter_name,
                num_wavelets=num_wavelets,
            )


        grad_wrt_wavelet_coeffs, wavelet_coeffs = wavelet.extract_grad_wrt_coffs(model, batched_adv_pc.detach().clone().double())

        # 滤波
        purified_coeffs = wavelet_coeffs.clone()
        if num_to_repair > 0:
            for i in range(batched_adv_pc.shape[0]):
                indices = mid_risk_indices[i]
                purified_coeffs[i, indices, :, -k_bands_to_filter:] *= attenuation_factor

        filtered_adv_pc =wavelet.inverse_transform(purified_coeffs)

        # 删点
        res = filtered_adv_pc[torch.ones_like(extracted_risk, dtype=torch.bool).cpu().scatter_(1, high_risk_indices, False)].view(batch_size, -1, 3)

        clean_input_item["res"] = res.detach().clone().cpu().float()
        clean_input_item["label"] = clean_label.detach().clone().cpu()
        all_defend_pc.append(clean_input_item)

    perf = PerfTrackVal()
    for j, data_batch in enumerate(all_defend_pc):
        purified_pc = data_batch["res"].to(device)
        label = data_batch["label"].to(device)
        perf.update(model.forward(purified_pc)["logit"].detach().clone(), label)
    res = perf.agg()
    logger.info("defenced acc:")
    logger.info(res)
    return res
