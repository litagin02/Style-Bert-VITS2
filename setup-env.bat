@echo off
echo ====================================
echo Style-Bert-VITS2 環境構築スクリプト
echo ====================================
echo.

REM Python 3.10.11の確認
python --version 2>nul | findstr /C:"3.10.11" >nul
if errorlevel 1 (
    echo [ERROR] Python 3.10.11が必要です。
    echo https://www.python.org/downloads/release/python-31011/ からインストールしてください。
    exit /b 1
)

echo [1/5] 既存の仮想環境を削除中...
if exist venv (
    rmdir /s /q venv
)

echo [2/5] 新しい仮想環境を作成中...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] 仮想環境の作成に失敗しました
    exit /b 1
)

echo [3/5] 仮想環境を有効化中...
call venv\Scripts\activate.bat

echo [4/5] pipをアップグレード中...
python -m pip install --upgrade pip

echo [5/5] 依存関係をインストール中...
pip install -r requirements.txt

echo.
echo ====================================
echo セットアップ完了！
echo ====================================
echo.
echo 仮想環境を有効化するには以下を実行してください:
echo   venv\Scripts\activate.bat
echo.
echo アプリケーションを起動するには:
echo   python app.py
echo.