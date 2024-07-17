conda create -n test_env python=3.10.13
conda activate test_env
@REM pip install -r requirements-dev.txt
pip install -r requirements.txt
pyinstaller app.spec