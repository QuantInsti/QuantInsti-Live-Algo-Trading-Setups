@echo off
setlocal

cd ..
python -m build
pip install dist\ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
cd user_config
python main.py
