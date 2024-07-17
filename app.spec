# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


from PyInstaller.utils.hooks import collect_data_files
import ultralytics
ultra_files = collect_data_files('ultralytics')

a = Analysis(
    ['app.py'], # <- Please change this into your python code name.
    pathex=[],
    binaries=[],
    datas=ultra_files + [('D:/my_work_and_projects/sperm-keypoint-detection-2/custom_track.yaml', '.'), ('D:/my_work_and_projects/sperm-keypoint-detection-2/model', 'model/')], # <- This is for enabling the referencing of all data files in the Ultralytics library.
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
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
