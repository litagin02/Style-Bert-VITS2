# Style-Bert-VITS2 環境構築ガイド

## 🚀 クイックスタート

### 自動セットアップ（推奨）

```powershell
# PowerShellで実行
.\setup.ps1
```

### 手動セットアップ

```powershell
# 1. 既存環境の削除と作成
Remove-Item venv -Recurse -Force -ErrorAction SilentlyContinue
python -m pip install uv
uv venv venv --python 3.10.11

# 2. 有効化
.\venv\Scripts\Activate.ps1

# 3. PyTorchインストール（CUDA 11.8）
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118

# 4. その他の依存関係
uv pip install -r requirements.txt

# 5. 初期化
python initialize.py
```

### Linux/macOS
```bash
# 既存の仮想環境がある場合は削除
rm -rf venv 2>/dev/null

# セットアップ
python3 -m pip install uv
uv venv --python 3.10.11
source venv/bin/activate
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
python initialize.py
```

### サーバー起動（2回目以降）
```bash
# Windows
venv\Scripts\activate.bat && python server_editor.py

# Linux/macOS
source venv/bin/activate && python server_editor.py
```

アクセス: http://localhost:8000

## 📋 動作環境

### 必須要件
- **Python**: 3.10.11（推奨）※3.9-3.11で動作可能
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS
- **GPU**: NVIDIA GPU推奨（CUDA対応）

### 重要なパッケージバージョン
| パッケージ | バージョン | 備考 |
|-----------|-----------|------|
| gradio | 4.36.1 | 固定必須 |
| fastapi | <0.113.0 | 0.113.0以降は非互換 |
| pydantic | >=2.0,<3.0 | v2必須 |
| numpy | 1.24.3 | 1.26.4はエラーの可能性 |
| protobuf | 4.25.0 | onnxとの互換性のため |
| onnx | 1.16.2 | protobuf 4.25.0と互換 |
| aivmlib | - | オプション（onnx>=1.17.0が必要） |
| torch | <2.4 | - |


## 🔧 トラブルシューティング

### よくあるエラーと対処法

#### 1. 白画面になる
```bash
# gradio-clientのバージョン不整合が原因
uv pip install gradio==4.36.1
```

#### 2. ONNX関連エラー (ImportError: cannot import name 'ONNX_ML')
```bash
# protobufバージョン競合の解決
uv pip uninstall onnx onnxsim protobuf
uv pip install protobuf==4.25.0
uv pip install onnx==1.16.2 onnxsim
```

#### 3. NumPy関連エラー (No module named 'numpy.core._multiarray_umath')
```bash
# NumPyの再インストール
uv pip uninstall numpy
uv pip install numpy==1.24.3
```

#### 4. ポート8000が既に使用されている
```bash
# Windows
netstat -ano | findstr :8000
taskkill //PID [PID番号] //F

# Linux/macOS
lsof -i :8000
kill -9 [PID番号]
```

#### 5. FastAPI関連エラー (PydanticSchemaGenerationError)
```bash
# FastAPIのバージョン制限
uv pip install "fastapi<0.113.0"
```

## 📦 ONNX変換機能

### 必要なファイル構成
モデルフォルダに以下のファイルが必要：
- `model.safetensors` - 変換元のモデルファイル
- `config.json` - モデル設定ファイル
- `style_vectors.npy` - スタイルベクターファイル

### 使用方法
```bash
# 全モデルを変換
python convert_onnx.py --model model_assets/

# 単一モデルを変換
python convert_onnx.py --model model_assets/model.safetensors

# 強制的に再変換
python convert_onnx.py --model model_assets/ --force-convert

# AIVM/AIVMX形式も生成（aivmlibが必要 - 別途インストール）
# uv pip install onnx>=1.17.0 aivmlib
python convert_onnx.py --model model_assets/ --aivm --aivmx
```

## ⚠️ 注意事項

1. **バージョンを勝手に更新しない**
   - 特にgradio、fastapi、pydanticは相互依存が複雑
   - 更新する場合は必ず動作確認を行う

2. **Python バージョン**
   - 3.10.11が最も安定
   - 3.12以降は未テスト

3. **仮想環境の使用**
   - 必ず仮想環境を使用すること
   - グローバル環境での実行は非推奨

## 環境確認

```powershell
# 環境チェック
.\check-env.ps1
```