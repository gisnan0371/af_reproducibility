@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call python run_all.py %*
pause
