@echo off
chcp 65001 >nul
echo ===================================================
echo   ML-Lab v3.8.1 一键启动
echo ===================================================

cd /d D:\WPSClaw\ML-Lab

echo.
echo [1/2] 安装依赖（已安装会自动跳过）...
python -m pip install -r requirements.txt networkx pyvis -q 2>nul

echo.
echo [2/2] 启动 ML-Lab...
echo   地址: http://localhost:7860
echo   按 Ctrl+C 停止
echo ===================================================
echo.
python app.py
pause
