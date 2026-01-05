@echo off
setlocal
chcp 65001 >nul

if "%~1"=="" (
  echo Please drag video files or folders onto this .bat to start.
  pause
  exit /b 1
)

set SCRIPT=%~dp0auto_subtitle.py
if not exist "%SCRIPT%" (
  echo 没找到 auto_subtitle.py ，请确保它和本bat在同一目录。
  pause
  exit /b 1
)

python "%SCRIPT%" %*
pause
