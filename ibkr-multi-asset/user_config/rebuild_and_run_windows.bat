@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

cd /d "%PROJECT_ROOT%"
python -m build
pip install dist\ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
cd /d "%SCRIPT_DIR%"
python main.py
