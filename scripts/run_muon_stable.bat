@echo off
setlocal
cd /d G:\RealSR
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe" -m model.train --data-root "G:\RealSR\data\RealSR(V3)" --out "G:\RealSR\experiments\improve\E11c_muon" --scale 2 --model-size base --batch-size 4 --grad-accum 2 --lr-patch 64 --steps 12000 --lr 0.001 --warmup 150 --num-workers 0 --eval-every 2000 --eval-pairs 4 --eval-tile 96 --eval-border 0 --save-every 1000 --cameras Canon,Nikon --align-loss --ema 0.999 --optimizer muon --seed 42
echo MUON_TRAIN_EXIT_%ERRORLEVEL%
