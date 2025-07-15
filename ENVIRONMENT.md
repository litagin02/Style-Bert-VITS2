# Style-Bert-VITS2 動作環境

## 動作確認済み環境

### 必須要件
- **Python**: 3.10.11 (3.9-3.11で動作可能だが、3.10.11推奨)
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS
- **GPU**: NVIDIA GPU推奨（CUDA対応）

### 重要なバージョン固定

#### 1. Gradio関連
```
gradio==4.36.1
gradio-client==1.0.1  # 自動でインストールされる
```

#### 2. Pydantic関連
```
pydantic>=2.0,<3.0
pydantic-core>=2.20.1
```

#### 3. FastAPI関連（重要！）
```
fastapi<0.113.0  # 0.113.0以降はGradioと互換性問題あり
starlette<0.39.0
uvicorn>=0.30.1
```

#### 4. PyTorch関連
```
torch<2.4
torchaudio<2.4
```

## セットアップ方法

### 方法1: 通常のpip（安定）
```bash
# 既存環境のクリーンアップ
pip uninstall gradio gradio-client pydantic fastapi -y

# 新規インストール
python -m venv venv
venv\Scripts\activate.bat  # Windowsの場合
source venv/bin/activate   # Linux/macOSの場合

pip install -r requirements.txt
```

### 方法2: uv使用（高速）
```bash
# uvのインストール
pip install uv

# 環境構築
uv venv --python 3.10.11
venv\Scripts\activate.bat
uv pip install -r requirements.txt
```

### 方法3: 自動セットアップスクリプト
```bash
# Windows
setup-env.bat

# Windows (uv版・高速)
setup-env-uv.bat

# Linux/macOS
chmod +x setup-env.sh
./setup-env.sh
```

## トラブルシューティング

### 1. 白画面になる場合
- gradio-clientのバージョン不整合が原因
- `pip install gradio==4.36.1`で解決

### 2. TypeError: argument of type 'bool' is not iterable
- pydanticのバージョン問題
- pydantic v2を使用し、コードもv2対応済み

### 3. PydanticSchemaGenerationError (starlette.requests.Request)
- FastAPI 0.113.0以降の非互換性が原因
- `pip install "fastapi<0.113.0"`で解決

### 4. pyopenjtalk関連エラー
- 日本語処理用のワーカープロセスが必要
- 自動的に起動されるが、エラーが出る場合は再起動

## バージョン履歴

### 最新版
- gradio 4.36.1 + pydantic v2対応
- FastAPI < 0.113.0制限追加
- pydantic v1からv2への移行完了

## 注意事項

1. **バージョンを勝手に更新しない**
   - 特にgradio、fastapi、pydanticは相互依存が複雑
   - 更新する場合は必ず動作確認を行う

2. **Python バージョン**
   - 3.10.11が最も安定
   - 3.12以降は未テスト

3. **仮想環境の使用**
   - 必ず仮想環境を使用すること
   - グローバル環境での実行は推奨しない


● 使用方法

  必要なファイル構成

  モデルフォルダに以下のファイルが必要です：
  - model.safetensors - 変換元のモデルファイル
  - config.json - モデル設定ファイル
  - style_vectors.npy - スタイルベクターファイル


## ONNX変換方法

  1. 単一モデルの変換
  python convert_onnx.py --model
  model_assets/koharune-ami/koharune-ami.safetensors

  2. ディレクトリ内の全モデルを変換
  python convert_onnx.py --model model_assets/

  オプション

  - --force-convert - 既存のONNXファイルを上書き
  - --aivm - SafetensorsからAIVMファイルを生成
  - --aivmx - ONNXからAIVMXファイルを生成

  実行例

  # 強制的に再変換
  python convert_onnx.py --model model_assets/model.safetensors --force-convert        

  # AIVM/AIVMXファイルも生成
  python convert_onnx.py --model model_assets/model.safetensors --aivm --aivmx

  変換後、同じディレクトリにmodel.onnxが生成されます。