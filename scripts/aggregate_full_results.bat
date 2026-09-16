@echo off
cd /d G:\RealSR
"C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe" scripts\aggregate_full_results.py
echo AGG_EXIT_%ERRORLEVEL%
