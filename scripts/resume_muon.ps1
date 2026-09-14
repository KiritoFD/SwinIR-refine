Set-Location G:\RealSR
$py = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
$out = "G:\RealSR\experiments\improve\E11c_muon"
# safer eval: smaller tile, fewer pairs
$argList = @(
  "-m","model.train",
  "--data-root","G:\RealSR\data\RealSR(V3)",
  "--out",$out,
  "--scale","2","--model-size","base",
  "--batch-size","4","--lr-patch","64",
  "--steps","12000","--lr","0.001","--warmup","150",
  "--num-workers","0",
  "--eval-every","2000","--eval-pairs","4","--eval-tile","96","--eval-border","2",
  "--save-every","4000",
  "--cameras","Canon,Nikon",
  "--align-loss","--ema","0.999",
  "--optimizer","muon","--seed","42"
)
# append to same logs
$log = "$out\train.out"
$err = "$out\train.err"
Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory "G:\RealSR" -RedirectStandardOutput "$out\train2.out" -RedirectStandardError "$out\train2.err" -NoNewWindow
Start-Sleep -Seconds 20
Get-Content "$out\train2.out" -Tail 12 -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object { $_.CommandLine -match 'E11c_muon' } | Select-Object ProcessId | Format-Table
