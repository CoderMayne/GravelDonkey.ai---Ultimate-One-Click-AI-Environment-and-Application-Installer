@echo off
echo =================================================================
echo  Launching AI Environment Installer inside WSL...
echo =================================================================
echo.
C:\Windows\System32\wsl.exe --cd "%~dp0" bash -c "source venv/Scripts/activate && python3 installer_app.py"
echo.
pause