# Background training via WSL tmux (survives tool timeouts)
# Usage: .\scripts\run_train_bg.ps1

$ErrorActionPreference = "Stop"
$root = "G:\RealSR"
$py = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
$exp = Join-Path $root "experiments\improve\E13_align_wiener"
New-Item -ItemType Directory -Force -Path $exp | Out-Null

$pyUnix = "/mnt/c/Users/xy/AppData/Local/Programs/Python/Python312/python.exe"
$rootUnix = "/mnt/g/RealSR"
$logUnix = "/mnt/g/RealSR/experiments/improve/E13_align_wiener/tmux_train.log"

$trainArgs = @(
  "-m", "mod_swinir.train",
  "--data-root", "G:\RealSR\data\RealSR(V3)",
  "--out", "G:\RealSR\experiments\improve\E13_align_wiener",
  "--scale", "2",
  "--arch", "mod",
  "--model-size", "base",
  "--batch-size", "4",
  "--grad-accum", "2",
  "--lr-patch", "64",
  "--steps", "12000",
  "--lr", "2e-4",
  "--warmup", "150",
  "--num-workers", "0",
  "--eval-every", "1500",
  "--eval-pairs", "6",
  "--save-every", "3000",
  "--cameras", "Canon,Nikon",
  "--amp", "--amp-dtype", "fp16",
  "--l1-only",
  "--align-loss",
  "--no-kpn",
  "--ema", "0.999"
)
$argStr = ($trainArgs | ForEach-Object { $_ }) -join " "
$cmd = "cd $rootUnix && $pyUnix $argStr > $logUnix 2>&1"

# kill old session if present
wsl bash -lc "tmux has-session -t e13 2>/dev/null && tmux kill-session -t e13; tmux new-session -d -s e13 '$cmd'"
Write-Host "launched tmux session e13"
wsl bash -lc "tmux ls"
