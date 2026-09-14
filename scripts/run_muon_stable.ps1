# Stable Muon A/B vs E11 (AdamW Align-L1 EMA 12k)
# No AMP (Newton-Schulz bf16 + GradScaler caused cudaErrorUnknown).
# batch 4 x grad_accum 2 = effective 8, matching E11.
$py = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
$out = "G:\RealSR\experiments\improve\E11c_muon"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location G:\RealSR

$argList = @(
  "-m","model.train",
  "--data-root","G:\RealSR\data\RealSR(V3)",
  "--out",$out,
  "--scale","2","--model-size","base",
  "--batch-size","4","--grad-accum","2","--lr-patch","64",
  "--steps","12000","--lr","0.001","--warmup","150",
  "--num-workers","0",
  "--eval-every","2000","--eval-pairs","4","--eval-tile","96","--eval-border","0",
  "--save-every","1000",
  "--cameras","Canon,Nikon",
  "--align-loss","--ema","0.999",
  "--optimizer","muon","--seed","42"
)
$log = "$out\train_stable.out"
$err = "$out\train_stable.err"
Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory "G:\RealSR" `
  -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow
Start-Sleep -Seconds 25
Get-Content $log -Tail 20 -ErrorAction SilentlyContinue
Get-Content $err -Tail 15 -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
  Where-Object { $_.CommandLine -match 'E11c_muon' } |
  Select-Object ProcessId, CommandLine | Format-List
