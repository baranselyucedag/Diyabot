@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Tip-2 Diyabet Chatbot - Baslatiliyor
echo ========================================
echo.

REM Backend (FastAPI) - yeni pencere
start "Diyabet-Backend" cmd /k "cd /d "%~dp0" && python -m src.api.app"

REM Frontend'in ayaga kalkmasi icin kisa bekleme
timeout /t 2 /nobreak >nul

REM Frontend (Vite) - yeni pencere
start "Diyabet-Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Backend  : http://127.0.0.1:8000
echo Frontend : http://localhost:5173  (veya 5174...)
echo.
echo Pencereleri kapatarak servisleri durdurabilirsiniz.
echo.
pause
