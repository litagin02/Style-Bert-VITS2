# Windows（日本語ユーザー名環境）向けセットアップ手順

このドキュメントは、**ユーザー名に日本語が含まれるWindows環境**（例：`C:\Users\川上貴大\`）での
Style-Bert-VITS2 の環境構築手順をまとめたものです。

## 動作確認済み環境

| 項目 | 内容 |
|------|------|
| OS | Windows 11 Pro |
| GPU | AMD Radeon（NVIDIA GPU なし → CPU推論） |
| Python | 3.10.11（`C:\Python310` にインストール） |
| PyTorch | 2.6.0+cpu |
| transformers | 4.57.x |

---

## クイックスタート（スクリプト実行）

PowerShell を開き、以下を実行するだけで環境構築が完了します。

```powershell
cd C:\Sbv2
PowerShell -ExecutionPolicy Bypass -File setup_windows_jp.ps1
```

デフォルト音声モデル（jvnv-F1-jp 等）のダウンロードをスキップする場合：

```powershell
PowerShell -ExecutionPolicy Bypass -File setup_windows_jp.ps1 -SkipModels
```

完了後は以下で Web UI を起動できます：

```powershell
.\venv\Scripts\python.exe app.py
```

ブラウザで `http://127.0.0.1:7860` を開いてください。

---

## 手動セットアップ手順

スクリプトを使わず手動で構築する場合の手順です。

### 1. Visual C++ 2015-2022 Redistributable のインストール

PyTorch が依存する `fbgemm.dll` を読み込むために必要です。

```powershell
winget install Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements
```

### 2. Python 3.10 を ASCII パスにインストール

**重要**: uv や pyenv が Python を自動インストールする場合、`C:\Users\川上貴大\AppData\Roaming\uv\python\...`
のような日本語を含むパスに配置されます。
このパスを venv の `pyvenv.cfg` に記録すると、一部の PowerShell 環境で Python が起動できなくなります。

日本語を含まない `C:\Python310` に直接インストールします。

```powershell
winget install Python.Python.3.10 --location C:\Python310 --accept-package-agreements --accept-source-agreements
```

インストール確認：

```powershell
C:\Python310\python.exe --version
# Python 3.10.11
```

### 3. uv のインストール

```powershell
C:\Python310\python.exe -m pip install uv
```

### 4. 仮想環境の作成

`C:\Python310` を使って venv を作成します（uv は使わず標準 venv モジュールを使用）。

```powershell
cd C:\Sbv2
C:\Python310\python.exe -m venv venv
```

`venv\pyvenv.cfg` を開き、`home = C:\Python310` になっていることを確認してください。

### 5. CPU 用 PyTorch のインストール

NVIDIA GPU がない環境では CPU 版 PyTorch を使用します。
AMD GPU でも CUDA は使えないため、CPU 推論になります（音声合成のみであれば実用的な速度で動作します）。

```powershell
uv pip install --python .\venv\Scripts\python.exe `
    "torch==2.6.0" "torchaudio==2.6.0" `
    --index-url https://download.pytorch.org/whl/cpu
```

### 6. 依存パッケージのインストール

`requirements.txt` をそのまま使用すると以下の問題が発生します。

| パッケージ | 問題 | 対処 |
|-----------|------|------|
| `faster-whisper==0.10.1` | `av` パッケージのビルドが Windows / Python 3.10 で失敗 | バージョン指定を外して最新版（1.2.1+）を使用 |
| `torch<2.4` / `torchaudio<2.4` | CUDA 版がデフォルトで選択される | 先に CPU 版をインストール済みのため除外 |

以下の手順でインストールします。

```powershell
# torch/torchaudio/faster-whisper を除いた requirements をインストール
(Get-Content requirements.txt | Where-Object { $_ -notmatch '^\s*(torch|torchaudio|faster-whisper)' }) `
    | Out-File requirements-custom.txt -Encoding utf8
uv pip install --python .\venv\Scripts\python.exe -r requirements-custom.txt

# faster-whisper は最新版（av のビルド不要）
uv pip install --python .\venv\Scripts\python.exe faster-whisper

# soxr: transformers 4.57+ が必要とするが requirements.txt に未記載
uv pip install --python .\venv\Scripts\python.exe soxr

# transformers: 4.x 系（torch 2.6 との互換性あり、CVE-2025-32434 セキュリティチェック対応済み）
uv pip install --python .\venv\Scripts\python.exe "transformers>=4.40,<5.0"
```

### 7. モデルのダウンロード

```powershell
# BERT モデル・事前学習モデル・デフォルト音声モデルをダウンロード
.\venv\Scripts\python.exe initialize.py

# デフォルト音声モデルをスキップする場合（ダウンロード容量を節約）
.\venv\Scripts\python.exe initialize.py --skip_default_models
```

---

## Web UI の起動

```powershell
cd C:\Sbv2
.\venv\Scripts\python.exe app.py
```

`http://127.0.0.1:7860` をブラウザで開いてください。

---

## よくあるエラーと対処法

### `No Python at 'C:\Users\????\AppData\...'`

**原因**: venv の `pyvenv.cfg` が日本語パスを参照しており、
ターミナルのコードページがそのパスを解釈できない。

**対処**: `venv` を削除し、手順 4 に従って `C:\Python310` を使って再作成する。

```powershell
Remove-Item -Recurse -Force .\venv
C:\Python310\python.exe -m venv venv
```

### `Error loading "fbgemm.dll" or one of its dependencies`

**原因**: Visual C++ 2015-2022 Redistributable が未インストール。

**対処**: 手順 1 を実施する。

### `[transformers] Disabling PyTorch because PyTorch >= 2.4 is required`

**原因**: transformers 5.x は torch 2.4+ を要求するが、torch 2.3.1 が入っている。

**対処**: transformers を 4.x 系に固定する。

```powershell
uv pip install --python .\venv\Scripts\python.exe "transformers>=4.40,<5.0"
```

### `ModuleNotFoundError: No module named 'soxr'`

**対処**:
```powershell
uv pip install --python .\venv\Scripts\python.exe soxr
```

### `モデルが見つかりませんでした。model_assets にモデルを置いてください`

**原因**: 音声モデルがダウンロードされていない（`--skip_default_models` を使用した場合など）。

**対処**: `initialize.py` を引数なしで実行してサンプルモデルをダウンロードするか、
自前のモデルを `model_assets/<モデル名>/` に配置する。

```powershell
.\venv\Scripts\python.exe initialize.py
```

---

## 補足：GPU について

このセットアップは **CPU 推論専用**です。

- NVIDIA GPU（CUDA 対応）がある場合は、`torch` を CUDA 版に置き換えてください。
  元の `requirements.txt` の手順（`cu118` インデックス URL）が使えます。
- AMD GPU では Windows 上での ROCm サポートが限定的なため、CPU 推論を推奨します。
  onnxruntime-directml（自動インストール済み）を使った ONNX 推論も選択肢の一つです。
