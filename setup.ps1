# setup.ps1
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Style-Bert-VITS2 環境構築" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Python確認
$pythonVersion = python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "エラー: Pythonが見つかりません。PATHを設定してください。" -ForegroundColor Red
    exit 1
}
Write-Host "Python: $pythonVersion" -ForegroundColor Green

# 既存のvenv削除
if (Test-Path venv) {
    Write-Host "既存の仮想環境を削除中..." -ForegroundColor Yellow
    Remove-Item venv -Recurse -Force
}

# uvインストール
Write-Host "`nuvをインストール中..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install uv

# 仮想環境作成
Write-Host "`n仮想環境を作成中..." -ForegroundColor Yellow
uv venv venv --python 3.10.11

# 仮想環境有効化
Write-Host "`n仮想環境を有効化中..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# PyTorchインストール（CUDA 11.8）
Write-Host "`nPyTorchをインストール中..." -ForegroundColor Yellow
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係
Write-Host "`nその他の依存関係をインストール中..." -ForegroundColor Yellow
uv pip install -r requirements.txt

# 確認
Write-Host "`n====================================" -ForegroundColor Cyan
Write-Host "インストール確認" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA利用可能: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 初期化
Write-Host "`n必要なモデルをダウンロード中..." -ForegroundColor Yellow
python initialize.py

Write-Host "`n====================================" -ForegroundColor Green
Write-Host "セットアップ完了！" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green