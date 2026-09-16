# Upload RealSR Train splits (needed for DiT training)
$ErrorActionPreference = "Continue"
$remote = "ds@10.222.120.101"
$base = "/home/ds/realsr/data/RealSR(V3)"
$log = "G:\RealSR\experiments\diffusion\server_sync\scp_train.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
"start $(Get-Date)" | Add-Content $log

foreach ($cam in @("Canon","Nikon")) {
  $src = "G:\RealSR\data\RealSR(V3)\$cam\Train"
  $dst = "${remote}:$base/$cam/"
  Write-Host "scp $src -> $dst"
  "scp $cam Train" | Add-Content $log
  scp -r $src $dst 2>&1 | Add-Content $log
  "scp $cam done rc=$LASTEXITCODE" | Add-Content $log
}

ssh $remote "find $base -name '*_LR2.png' | wc -l; du -sh $base" 2>&1 | Add-Content $log
"ALL_DONE $(Get-Date)" | Add-Content $log
Write-Host "done"
