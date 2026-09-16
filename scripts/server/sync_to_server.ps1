# Push RealSR diffusion package to ds@10.222.120.101
# Requires: ssh public key authorized on the server (see user pubkey).
param(
  [string]$Host = "ds@10.222.120.101",
  [string]$RemoteRoot = "/home/ds/realsr"
)
$ErrorActionPreference = "Stop"
$local = "G:\RealSR"

ssh $Host "mkdir -p $RemoteRoot/diffusion $RemoteRoot/scripts/server $RemoteRoot/model $RemoteRoot/experiments/diffusion"

# package code
scp -r "$local\diffusion"/* "${Host}:$RemoteRoot/diffusion/"
scp -r "$local\scripts\server"/* "${Host}:$RemoteRoot/scripts/server/"
# reuse official metrics from model/
scp "$local\model\metrics.py" "${Host}:$RemoteRoot/model/"
scp "$local\model\__init__.py" "${Host}:$RemoteRoot/model/" 2>$null
# empty package init if missing
ssh $Host "touch $RemoteRoot/model/__init__.py $RemoteRoot/diffusion/__init__.py"
ssh $Host "chmod +x $RemoteRoot/scripts/server/run_pipeline.sh"

Write-Host "SYNC OK → $Host:$RemoteRoot"
Write-Host "Next: put RealSR(V3) at $RemoteRoot/data/RealSR(V3) then:"
Write-Host "  ssh $Host 'cd $RemoteRoot && SMOKE=1 bash scripts/server/run_pipeline.sh flux1-dev'"
