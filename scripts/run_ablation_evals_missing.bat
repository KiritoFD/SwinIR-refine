@echo off
cd /d G:\RealSR
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe" scripts\run_ablation_evals.py --only E7_v2_l1_amp_ema,E8_v2_kpn
echo ABLATION_EVAL_EXIT_%ERRORLEVEL%
