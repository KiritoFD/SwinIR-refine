"""Aggregate all official full-set eval.json into one comparison table.

Searches known batch folders + ablation_full_* and prefers n>=100 results.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"G:\RealSR")
EVAL_ROOT = ROOT / "experiments" / "eval_official"

# canonical id -> preferred relative eval.json locations (first hit with max n)
PREFERRED = {
    "E11_align": ["batch_20260914/E11_align_best"],
    "E9_no_align": ["batch_20260914/E9_long12k_best"],
    "A0_no_ema": ["batch_20260914/A0_l1_only_best"],
    "A1_ema": ["batch_20260915/A1_l1_ema", "batch_20260914/A1_l1_ema"],
    "A2_patchamp": ["batch_20260915/A2_l1_ema_patchamp"],
    "A3_hfconf": ["batch_20260915/A3_l1_ema_hfconf"],
    "A4_full_aux": ["batch_20260915/A4_full", "batch_20260914/A4_full"],
    "E10_realsr_loss": ["batch_20260915/E10_realsr_loss", "batch_20260914/E10_realsr_loss"],
    "E12_lpkpn": ["batch_20260915/E12_lpkpn", "batch_20260914/E12_lpkpn"],
    "E13_wiener": ["batch_20260915/E13_wiener", "batch_20260914/E13_wiener"],
    "E14_ampphase": ["batch_20260915/E14_ampphase", "batch_20260914/E14_ampphase"],
    "E15_radialpsf": ["batch_20260915/E15_radialpsf", "batch_20260914/E15_radialpsf"],
    "E7_v2_l1_amp_ema": ["batch_20260915/E7_v2_l1_amp_ema", "ablation_full_20260915/E7_v2_l1_amp_ema"],
    "E8_v2_kpn": ["batch_20260915/E8_v2_kpn", "ablation_full_20260915/E8_v2_kpn"],
    "E1_swinir_light": ["batch_20260914/E1_swinir_light"],
    "E2_swinir_capmatch": ["batch_20260914/E2_swinir_capmatch"],
    "E5_swinir_classical": ["batch_20260914/E5_swinir_classical"],
    "E11c_muon": ["ablation_full_20260915/E11c_muon", "batch_20260915/E11c_muon"],
}

ORDER = [
    "E11_align", "E9_no_align", "E11c_muon",
    "A0_no_ema", "A1_ema", "A2_patchamp", "A3_hfconf", "A4_full_aux",
    "E10_realsr_loss", "E12_lpkpn",
    "E13_wiener", "E14_ampphase", "E15_radialpsf",
    "E7_v2_l1_amp_ema", "E8_v2_kpn",
    "E2_swinir_capmatch", "E1_swinir_light", "E5_swinir_classical",
]


def load_best(paths: list[str]) -> dict | None:
    best = None
    for rel in paths:
        p = EVAL_ROOT / rel / "eval.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        n = int(data.get("n") or 0)
        if n <= 0:
            continue
        if best is None or n > int(best.get("n") or 0):
            best = data
            best["_path"] = str(p)
    return best


def main() -> None:
    rows = []
    for rid in ORDER:
        data = load_best(PREFERRED.get(rid, []))
        if data is None:
            rows.append({"id": rid, "n": 0, "Y": None, "SSIM_Y": None, "RGB": None, "path": ""})
            continue
        rows.append({
            "id": rid,
            "n": data.get("n"),
            "Y": data.get("psnr_y"),
            "SSIM_Y": data.get("ssim_y"),
            "RGB": data.get("psnr_rgb"),
            "path": data.get("_path", ""),
        })

    out_json = EVAL_ROOT / "OFFICIAL_FULL_SUMMARY.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    base = next((r for r in rows if r["id"] == "E11_align" and r["Y"]), None)
    lines = [
        "# 官方全量评估汇总（RealSR V3 Test.m, n=100）",
        "",
        "生成：`python scripts/aggregate_full_results.py`",
        "",
        "| 模型 | n | Y-PSNR | Y-SSIM | RGB | ΔY vs E11 |",
        "|------|---|--------|--------|-----|-----------|",
    ]
    for r in rows:
        if r["Y"] is None:
            lines.append(f"| {r['id']} | {r['n'] or 0} | — | — | — | — |")
            continue
        d = f"{r['Y']-base['Y']:+.4f}" if base and r["id"] != "E11_align" else "—"
        lines.append(
            f"| {r['id']} | {r['n']} | {r['Y']:.4f} | {r['SSIM_Y']:.4f} | "
            f"{r['RGB']:.4f} | {d} |"
        )
    lines += [
        "",
        "## 关键对比",
        "",
    ]
    by = {r["id"]: r for r in rows if r.get("Y") is not None}

    def delta(a: str, b: str) -> str:
        if a not in by or b not in by:
            return f"{a} vs {b}: 缺结果"
        return f"- **{a} − {b}** = {by[a]['Y']-by[b]['Y']:+.4f} dB Y"

    for pair in [
        ("E11_align", "E9_no_align"),
        ("A1_ema", "A0_no_ema"),
        ("E11c_muon", "E11_align"),
        ("E12_lpkpn", "E11_align"),
        ("E11_align", "E2_swinir_capmatch"),
        ("E10_realsr_loss", "E11_align"),
    ]:
        lines.append(delta(*pair))

    out_md = EVAL_ROOT / "OFFICIAL_FULL_SUMMARY.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"\nwrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
