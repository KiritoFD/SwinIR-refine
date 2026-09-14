# Official RealSR Test.m protocol eval batch
$py = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\RealSR\experiments\eval_official\batch_20260914"
New-Item -ItemType Directory -Force -Path $root | Out-Null
Set-Location G:\RealSR

# name, ckpt, max_pairs (0=full 100)
$jobs = @(
  @{name="E11_align_best";      ckpt="G:\RealSR\experiments\improve\E11_align_loss\ckpt_best.pt";           pairs=0},
  @{name="A0_l1_only_best";     ckpt="G:\RealSR\experiments\ablation_x2\A0_l1_only\ckpt_best.pt";          pairs=0},
  @{name="E9_long12k_best";     ckpt="G:\RealSR\experiments\improve\E9_v2_long12k\ckpt_best.pt";           pairs=0},
  @{name="E2_swinir_capmatch";  ckpt="G:\RealSR\experiments\matrix_x2\E2_swinir_capmatch\ckpt_best.pt";    pairs=0},
  @{name="E5_swinir_classical"; ckpt="G:\RealSR\experiments\matrix_x2\E5_swinir_classical\ckpt_best.pt";   pairs=0},
  @{name="E1_swinir_light";     ckpt="G:\RealSR\experiments\matrix_x2\E1_swinir_light\ckpt_best.pt";       pairs=0},
  @{name="E6_l1_ema";           ckpt="G:\RealSR\experiments\improve\E6_mod_base_l1_ema\ckpt_best.pt";      pairs=30},
  @{name="E10_realsr_loss";     ckpt="G:\RealSR\experiments\improve\E10_realsr_loss_15k\ckpt_best.pt";     pairs=30},
  @{name="A4_full";             ckpt="G:\RealSR\experiments\ablation_x2\A4_full\ckpt_best.pt";             pairs=30},
  @{name="A1_l1_ema";           ckpt="G:\RealSR\experiments\ablation_x2\A1_l1_ema\ckpt_best.pt";           pairs=30},
  @{name="E12_lpkpn";           ckpt="G:\RealSR\experiments\improve\E12_align_lpkpn\ckpt_best.pt";         pairs=20},
  @{name="E13_wiener";          ckpt="G:\RealSR\experiments\improve\E13_align_wiener\ckpt_best.pt";        pairs=20},
  @{name="E14_ampphase";        ckpt="G:\RealSR\experiments\improve\E14_align_ampphase\ckpt_best.pt";      pairs=20},
  @{name="E15_radialpsf";       ckpt="G:\RealSR\experiments\improve\E15_align_radialpsf\ckpt_best.pt";     pairs=20},
  @{name="E4_mod_large";        ckpt="G:\RealSR\experiments\matrix_x2\E4_mod_large\ckpt_best.pt";          pairs=20},
  @{name="mod_base_early";      ckpt="G:\RealSR\experiments\mod_swinir_x2_base\ckpt_best.pt";              pairs=20}
)

$summary = @()
foreach ($j in $jobs) {
  $out = Join-Path $root $j.name
  Write-Host "==== $($j.name) max_pairs=$($j.pairs) ====" -ForegroundColor Cyan
  & $py -m model.eval --ckpt $j.ckpt --data-root "G:\RealSR\data\RealSR(V3)" --scale 2 --max-pairs $j.pairs --out $out --tile 192
  if (Test-Path (Join-Path $out "eval.json")) {
    $ej = Get-Content (Join-Path $out "eval.json") -Raw | ConvertFrom-Json
    $summary += [pscustomobject]@{
      name=$j.name; n=$ej.n; Y=$ej.psnr_y; SSIM_Y=$ej.ssim_y; RGB=$ej.psnr_rgb
    }
    Write-Host ("  Y={0:N2} SSIM={1:N4} RGB={2:N2} n={3}" -f $ej.psnr_y,$ej.ssim_y,$ej.psnr_rgb,$ej.n)
  } else {
    Write-Host "  FAILED" -ForegroundColor Red
  }
}

$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content (Join-Path $root "summary.json") -Encoding utf8
$summary | Export-Csv (Join-Path $root "summary.csv") -NoTypeInformation
Write-Host "DONE $root"
