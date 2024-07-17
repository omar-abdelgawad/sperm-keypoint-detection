pip install virtualenv
virtualenv venv
.\venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -r requirements.txt
pyinstaller app.spec