"""Load-test every experiment checkpoint against the current slim model package."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, r"G:\RealSR")
from model.model import build_model
from model.swinir_baseline import SwinIRTrainWrapper, build_swinir

PATHS = {
    "A0": r"G:\RealSR\experiments\ablation_x2\A0_l1_only\ckpt_best.pt",
    "A1": r"G:\RealSR\experiments\ablation_x2\A1_l1_ema\ckpt_best.pt",
    "A2": r"G:\RealSR\experiments\ablation_x2\A2_l1_ema_patchamp\ckpt_best.pt",
    "A3": r"G:\RealSR\experiments\ablation_x2\A3_l1_ema_hfconf\ckpt_best.pt",
    "A4": r"G:\RealSR\experiments\ablation_x2\A4_full\ckpt_best.pt",
    "E7": r"G:\RealSR\experiments\improve\E7_mod_v2_l1_amp_ema\ckpt_best.pt",
    "E8": r"G:\RealSR\experiments\improve\E8_v2_kpn\ckpt_best.pt",
    "E6b": r"G:\RealSR\experiments\improve\E6b_mod_base_l1_ema_ft\ckpt_best.pt",
    "E10": r"G:\RealSR\experiments\improve\E10_realsr_loss_15k\ckpt_best.pt",
    "E12": r"G:\RealSR\experiments\improve\E12_align_lpkpn\ckpt_best.pt",
    "E13": r"G:\RealSR\experiments\improve\E13_align_wiener\ckpt_best.pt",
    "E14": r"G:\RealSR\experiments\improve\E14_align_ampphase\ckpt_best.pt",
    "E15": r"G:\RealSR\experiments\improve\E15_align_radialpsf\ckpt_best.pt",
    "E6": r"G:\RealSR\experiments\improve\E6_mod_base_l1_ema\ckpt_best.pt",
    "E4": r"G:\RealSR\experiments\matrix_x2\E4_mod_large\ckpt_best.pt",
    "mod_base_early": r"G:\RealSR\experiments\mod_swinir_x2_base\ckpt_best.pt",
    "E11": r"G:\RealSR\experiments\improve\E11_align_loss\ckpt_best.pt",
    "E1_swinir": r"G:\RealSR\experiments\matrix_x2\E1_swinir_light\ckpt_best.pt",
    "E2_swinir": r"G:\RealSR\experiments\matrix_x2\E2_swinir_capmatch\ckpt_best.pt",
    "E5_swinir": r"G:\RealSR\experiments\matrix_x2\E5_swinir_classical\ckpt_best.pt",
}


def main() -> None:
    ok, fail = [], []
    for name, path in PATHS.items():
        p = Path(path)
        if not p.exists():
            fail.append((name, "missing file"))
            print(f"FAIL {name:16s} missing file")
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            a = ck.get("args", {})
            arch = a.get("arch", ck.get("meta", {}).get("arch", "mod"))
            size = a.get("model_size", "base")
            scale = a.get("scale", 2)
            if arch == "swinir":
                model = SwinIRTrainWrapper(build_swinir(size=size, upscale=scale))
            else:
                model = build_model(upscale=scale, size=size)
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            nmiss = len(missing)
            nextra = len(unexpected)
            # size-mismatch keys land in missing when shapes differ? actually strict=False still errors on shape
            print(
                f"OK   {name:16s} arch={arch:6s} size={size:6s} "
                f"missing={nmiss} extra={nextra} step={ck.get('step')}"
            )
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:140]}"
            print(f"FAIL {name:16s} {msg}")
            fail.append((name, msg))
    print(f"\nOK={len(ok)} FAIL={len(fail)}")
    for name, msg in fail:
        print(f"  - {name}: {msg}")


if __name__ == "__main__":
    main()
