@echo off
chcp 65001 > NUL
set CONDA_ALWAYS_YES=false

pushd %~dp0

echo anime-whisperを使用した文字起こしを開始します...

set /p MODEL_NAME="モデル名を入力してください: "

echo.
echo 設定内容:
echo モデル名: %MODEL_NAME%
echo.
echo 注意: anime-whisperは初期プロンプトを使用しません（相性が悪いため）

echo anime-whisperモデルのダウンロードを確認中...
call python download_anime_whisper.py

if %errorlevel% neq 0 (
    echo エラー: anime-whisperのダウンロードに失敗しました。
    pause
    exit /b 1
)

echo.
echo 文字起こしを開始します...
call python transcribe_anime.py --model_name "%MODEL_NAME%"

if %errorlevel% neq 0 (
    echo エラー: 文字起こしに失敗しました。
    pause
    exit /b 1
)

echo.
echo 文字起こしが完了しました！
echo 結果は Data\%MODEL_NAME%\esd.list に保存されています。

popd
pause