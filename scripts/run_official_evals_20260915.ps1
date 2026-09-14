# Official full-set eval for checkpoints missing from batch_20260914
$py = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\RealSR\experiments\eval_official\batch_20260915"
New-Item -ItemType Directory -Force -Path $root | Out-Null
Set-Location G:\RealSR

# name, ckpt — all full 100 pairs
$jobs = @(
  @{name="E10_realsr_loss";   ckpt="G:\RealSR\experiments\improve\E10_realsr_loss_15k\ckpt_best.pt"},
  @{name="A1_l1_ema";         ckpt="G:\RealSR\experiments\ablation_x2\A1_l1_ema\ckpt_best.pt"},
  @{name="A2_l1_ema_patchamp";ckpt="G:\RealSR\experiments\ablation_x2\A2_l1_ema_patchamp\ckpt_best.pt"},
  @{name="A3_l1_ema_hfconf";  ckpt="G:\RealSR\experiments\ablation_x2\A3_l1_ema_hfconf\ckpt_best.pt"},
  @{name="A4_full";           ckpt="G:\RealSR\experiments\ablation_x2\A4_full\ckpt_best.pt"},
  @{name="E7_v2_l1_amp_ema";  ckpt="G:\RealSR\experiments\improve\E7_mod_v2_l1_amp_ema\ckpt_best.pt"},
  @{name="E8_v2_kpn";         ckpt="G:\RealSR\experiments\improve\E8_v2_kpn\ckpt_best.pt"},
  @{name="E12_lpkpn";         ckpt="G:\RealSR\experiments\improve\E12_align_lpkpn\ckpt_best.pt"},
  @{name="E13_wiener";        ckpt="G:\RealSR\experiments\improve\E13_align_wiener\ckpt_best.pt"},
  @{name="E14_ampphase";      ckpt="G:\RealSR\experiments\improve\E14_align_ampphase\ckpt_best.pt"},
  @{name="E15_radialpsf";     ckpt="G:\RealSR\experiments\improve\E15_align_radialpsf\ckpt_best.pt"}
)

$summary = @()
foreach ($j in $jobs) {
  $out = Join-Path $root $j.name
  if (Test-Path (Join-Path $out "eval.json")) {
    Write-Host "==== SKIP $($j.name) (exists) ====" -ForegroundColor Yellow
  } else {
    Write-Host "==== $($j.name) max_pairs=0 ====" -ForegroundColor Cyan
    & $py -m model.eval --ckpt $j.ckpt --data-root "G:\RealSR\data\RealSR(V3)" --scale 2 --max-pairs 0 --out $out --tile 192
  }
  if (Test-Path (Join-Path $out "eval.json")) {
    $ej = Get-Content (Join-Path $out "eval.json") -Raw | ConvertFrom-Json
    $summary += [pscustomobject]@{
      name=$j.name; n=$ej.n; Y=$ej.psnr_y; SSIM_Y=$ej.ssim_y; RGB=$ej.psnr_rgb
    }
    Write-Host ("  Y={0:N4} SSIM={1:N4} RGB={2:N4} n={3}" -f $ej.psnr_y,$ej.ssim_y,$ej.psnr_rgb,$ej.n)
  } else {
    Write-Host "  FAILED" -ForegroundColor Red
    $summary += [pscustomobject]@{ name=$j.name; n=0; Y=$null; SSIM_Y=$null; RGB=$null }
  }
}

$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content (Join-Path $root "summary.json") -Encoding utf8
$summary | Export-Csv (Join-Path $root "summary.csv") -NoTypeInformation
Write-Host "DONE $root"
