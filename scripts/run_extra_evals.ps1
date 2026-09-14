$py = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
$root = "G:\RealSR\experiments\eval_official\batch_20260914"
Set-Location G:\RealSR
$jobs = @(
  @{name="E6_l1_ema";     ckpt="G:\RealSR\experiments\improve\E6_mod_base_l1_ema\ckpt_best.pt"; pairs=30},
  @{name="E4_mod_large";  ckpt="G:\RealSR\experiments\matrix_x2\E4_mod_large\ckpt_best.pt";     pairs=20},
  @{name="mod_base_early";ckpt="G:\RealSR\experiments\mod_swinir_x2_base\ckpt_best.pt";         pairs=20}
)
foreach ($j in $jobs) {
  $out = Join-Path $root $j.name
  Write-Host "==== $($j.name) ===="
  & $py -m model.eval --ckpt $j.ckpt --data-root "G:\RealSR\data\RealSR(V3)" --scale 2 --max-pairs $j.pairs --out $out --tile 192
  if (Test-Path (Join-Path $out "eval.json")) {
    $ej = Get-Content (Join-Path $out "eval.json") -Raw | ConvertFrom-Json
    Write-Host ("  Y={0:N2} n={1}" -f $ej.psnr_y,$ej.n)
  } else { Write-Host "  FAILED" -ForegroundColor Red }
}
Write-Host EXTRA_DONE
