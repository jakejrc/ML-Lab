@echo off
chcp 65001 >nul
cd /d D:\WPSClaw\ML-Lab
python app.py --port 7860 --share 2>&1 | tee startup_v33_auto.log
pause