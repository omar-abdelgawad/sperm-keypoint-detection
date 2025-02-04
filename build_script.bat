Building steps:
1- code normally on vscode.
2- open auto-py-to-exe and specify what you want.
3- don't forget to add model folder to app (do it manually if it is not working).
4- copy the command from the auto-py-to-exe gui.
5- use the  command to produce the app.spec file (it will also build the build folder and dist folder but they won't work).
6- delete the build and dist folder from the first pyinstaller command.
7- copy paste the code below in the .spec file and then run [ pyinstaller .\app.spec] to build two new dist and build folders.
8- the executable in the dist folder should work. 



from PyInstaller.utils.hooks import collect_data_files
import ultralytics
ultra_files = collect_data_files('ultralytics')

a = Analysis(
    ['app.py'], # <- Please change this into your python code name.
    pathex=[],
    binaries=[],
    datas=ultra_files, # <- This is for enabling the referencing of all data files in the Ultralytics library.
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


====================================================================================
conda create -n test_env python=3.10.13
conda activate test_env
@REM pip install -r requirements-dev.txt
pip install -r requirements.txt
pyinstaller app.spec

====================================================================================
for cloning the environment with conda:
conda create --name test_env --clone sperm_detection_env
once you update the ultralytics version and torch version you should be good to go.