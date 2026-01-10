# Style-Bert-VITS2 訓練ワークフロー

esd.listからモデル訓練、スタイルベクトル作成、推論までの一連の流れ。

## 概要

```
esd.list → 前処理 → BERT生成 → スタイル生成 → 訓練 → model_assets作成 → 推論
```

---

## 1. データ準備

### 1.1 ディレクトリ構成

```
Data/{model_name}/
├── esd.list              # 書き起こしファイル
├── wavs/                 # 音声ファイル
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ...
└── config.json           # 訓練設定（後で生成）
```

### 1.2 esd.list フォーマット

```
ファイル名|スピーカー名|言語|テキスト
```

例:
```
audio001.wav|speaker1|JP|こんにちは、今日はいい天気ですね。
audio002.wav|speaker1|JP|明日も晴れるといいですね。
```

- **ファイル名**: `wavs/`ディレクトリからの相対パス
- **スピーカー名**: 任意の名前（1話者なら統一）
- **言語**: `JP`, `EN`, `ZH` のいずれか
- **テキスト**: 音声の書き起こし

---

## 2. 前処理

### 2.1 テキスト前処理

```bash
python preprocess_text.py \
  --transcription-path Data/{model_name}/esd.list \
  --train-path Data/{model_name}/train.list \
  --val-path Data/{model_name}/val.list \
  --config-path Data/{model_name}/config.json \
  --use_jp_extra \
  --correct_path \
  --yomi_error skip
```

**オプション説明:**
- `--correct_path`: ファイルパスを`Data/{model_name}/wavs/`に自動補正
- `--yomi_error skip`: 読み仮名エラーをスキップ（エラー行を除外して続行）

**処理内容:**
- テキストの正規化（数字→読み、記号処理など）
- 音素(phones)、アクセント(tones)、単語境界(word2ph)の抽出
- `train.list` / `val.list` の生成

**出力ファイル:**
```
Data/{model_name}/
├── train.list            # 訓練用リスト（前処理済み）
├── val.list              # 検証用リスト
└── preprocess_text_error.log  # エラーログ
```

**train.list フォーマット:**
```
wavs/audio001.wav|speaker1|JP|正規化テキスト|音素列|トーン列|word2ph列
```

### 2.2 BERT特徴量生成

```bash
python bert_gen.py -c Data/{model_name}/config.json
```

**処理内容:**
- 各音声ファイルに対応するBERT特徴量を抽出
- 日本語: `deberta-v2-large-japanese-char-wwm`

**出力ファイル:**
```
Data/{model_name}/wavs/
├── audio001.bert.pt      # BERT特徴量
├── audio002.bert.pt
└── ...
```

### 2.3 スタイルベクトル生成

```bash
python style_gen.py -c Data/{model_name}/config.json
```

**注意:**
- 既に`*.wav.npy`ファイルが存在する場合、この手順はスキップ可能
- pyannote-audioのバージョンによってはAudioDecoderエラーが発生する場合がある
- 既存ファイル確認: `ls Data/{model_name}/wavs/*.wav.npy | wc -l`

**処理内容:**
- pyannote/wespeaker-voxceleb-resnet34-LMで話者埋め込みを抽出
- 各音声ファイルに256次元のスタイルベクトルを保存

**出力ファイル:**
```
Data/{model_name}/wavs/
├── audio001.wav.npy      # スタイルベクトル (256,)
├── audio002.wav.npy
└── ...
```

---

## 3. config.json 作成

### 3.1 基本構成

```json
{
  "model_name": "{model_name}",
  "train": {
    "log_interval": 100,
    "eval_interval": 500,
    "save_every_steps": 1000,
    "seed": 42,
    "epochs": 50,
    "learning_rate": 1e-05,
    "betas": [0.8, 0.99],
    "eps": 1e-09,
    "batch_size": 4,
    "bf16_run": false,
    "fp16_run": false,
    "freeze_ZH_bert": true,
    "freeze_JP_bert": true,
    "freeze_EN_bert": true,
    "freeze_emo": false,
    "freeze_style": true,
    "freeze_decoder": false
  },
  "data": {
    "use_jp_extra": true,
    "training_files": "Data/{model_name}/train.list",
    "validation_files": "Data/{model_name}/val.list",
    "max_wav_value": 32768.0,
    "sampling_rate": 44100,
    "filter_length": 2048,
    "hop_length": 512,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 1,
    "cleaned_text": true,
    "spk2id": {
      "{speaker_name}": 0
    }
  },
  "model": {
    "use_spk_conditioned_encoder": true,
    "use_noise_scaled_mas": true,
    "use_mel_posterior_encoder": false,
    "use_duration_discriminator": false,
    "use_wavlm_discriminator": true,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "gin_channels": 512,
    "slm": {
      "model": "./slm/wavlm-base-plus",
      "sr": 16000,
      "hidden": 768,
      "nlayers": 13,
      "initial_channel": 64
    }
  },
  "version": "2.7.0-JP-Extra"
}
```

---

## 4. 訓練

### 4.1 事前学習モデルの配置

```bash
# model_assets から事前学習モデルをコピー
# 注意: 訓練スクリプトは {model_path}/models/G_0.safetensors を探す
mkdir -p Data/{model_name}/models/models
cp model_assets/jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors \
   Data/{model_name}/models/models/G_0.safetensors
```

### 4.2 既存チェックポイントの削除（新規訓練時）

再訓練時に前回のチェックポイントから再開してしまう場合、既存のチェックポイントを削除する必要がある。

```bash
# 既存チェックポイントを削除（G_0.safetensorsは保持）
rm -rf Data/{model_name}/models/models/
```

**注意:** `Data/{model_name}/models/models/`ディレクトリには訓練中のチェックポイント（G_*.pth, D_*.pth, WD_*.pth）が保存される。このディレクトリを削除することで、G_0.safetensorsから新規訓練を開始できる。

### 4.3 訓練実行

```bash
python train_ms_jp_extra.py \
  --config Data/{model_name}/config.json \
  --model Data/{model_name}/models
```

**バックグラウンド実行:**
```bash
nohup python train_ms_jp_extra.py \
  --config Data/{model_name}/config.json \
  --model Data/{model_name}/models \
  > {model_name}_train.log 2>&1 &
```

**出力ファイル:**
```
Data/{model_name}/models/
├── G_0.safetensors       # 初期モデル
├── G_1000.pth            # チェックポイント
├── G_2000.pth
├── D_1000.pth            # Discriminator
├── D_2000.pth
└── ...
```

---

## 5. model_assets 作成

訓練完了後、推論用にmodel_assetsディレクトリを構成する。

### 5.1 ディレクトリ作成

```bash
mkdir -p model_assets/{model_name}
```

### 5.2 モデル変換 (pth → safetensors)

```python
import torch
from safetensors.torch import save_file
from pathlib import Path

INPUT_PTH = Path("Data/{model_name}/models/G_{step}.pth")
OUTPUT_DIR = Path("model_assets/{model_name}")

checkpoint = torch.load(INPUT_PTH, map_location="cpu")
state_dict = checkpoint.get("model", checkpoint)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

save_file(new_state_dict, OUTPUT_DIR / "{model_name}_e{epoch}_s{step}.safetensors")
```

### 5.3 スタイルベクトル生成

```python
import numpy as np
from pathlib import Path

wavs_dir = Path("Data/{model_name}/wavs")
npy_files = list(wavs_dir.glob("*.wav.npy"))

# 全ベクトルを読み込み
vectors = [np.load(f) for f in npy_files]
x = np.array(vectors)

# 平均を計算 (Neutralスタイル)
mean = np.mean(x, axis=0)
style_vectors = mean.reshape(1, -1)

# 保存
np.save("model_assets/{model_name}/style_vectors.npy", style_vectors)
```

### 5.4 config.json 作成

```json
{
  "model_name": "{model_name}",
  "data": {
    "use_jp_extra": true,
    "spk2id": {"{speaker_name}": 0},
    "num_styles": 1,
    "style2id": {"Neutral": 0}
  },
  "model": {
    // 訓練時と同じmodel設定
  },
  "version": "2.0-JP-Extra"
}
```

### 5.5 最終構成

```
model_assets/{model_name}/
├── {model_name}_e{epoch}_s{step}.safetensors  # モデル
├── config.json                                  # 推論設定
└── style_vectors.npy                            # スタイルベクトル
```

---

## 6. 推論

### 6.1 基本推論

```python
from style_bert_vits2.tts_model import TTSModel
from pathlib import Path
import soundfile as sf

MODEL_DIR = Path("model_assets/{model_name}")

model = TTSModel(
    model_path=MODEL_DIR / "{model_name}_e{epoch}_s{step}.safetensors",
    config_path=MODEL_DIR / "config.json",
    style_vec_path=MODEL_DIR / "style_vectors.npy",
    device="cuda:0"
)

sr, audio = model.infer(
    text="こんにちは、今日はいい天気ですね。",
    language="JP",
    sdp_ratio=0.2,
    noise=0.6,
    noise_w=0.8,
    length=1.0,
    style="Neutral",
    style_weight=1.0
)

sf.write("output.wav", audio, sr)
```

### 6.2 推論パラメータ

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| `sdp_ratio` | Stochastic Duration Predictor比率 | 0.0-0.5 |
| `noise` | デコーダノイズ | 0.3-0.6 |
| `noise_w` | 持続時間ノイズ | 0.3-0.8 |
| `length` | 発話速度 (1.0=標準) | 0.8-1.2 |
| `style` | スタイル名 | "Neutral" |
| `style_weight` | スタイル強度 | 0.0-5.0 |

**ノイズが高い場合:** `noise=0.3, noise_w=0.3` を試す

---

## トラブルシューティング

### 音声がガビガビする

1. **スタイルベクトルの確認**
   ```python
   import numpy as np
   sv = np.load("model_assets/{model_name}/style_vectors.npy")
   print(f"Shape: {sv.shape}")  # (1, 256) であること
   print(f"Range: {sv.min():.3f} to {sv.max():.3f}")  # -0.5〜0.5程度
   ```

2. **ノイズパラメータを下げる**
   ```python
   sr, audio = model.infer(..., noise=0.3, noise_w=0.3)
   ```

3. **訓練エポック数を増やす**

### 訓練が進まない

1. **事前学習モデルの確認**
   - `G_0.safetensors` が正しく配置されているか

2. **GPUメモリ不足**
   - `batch_size` を下げる

---

## コマンドまとめ

```bash
# 1. 前処理
python preprocess_text.py -c Data/{model_name}/config.json

# 2. BERT生成
python bert_gen.py -c Data/{model_name}/config.json

# 3. スタイル生成
python style_gen.py -c Data/{model_name}/config.json

# 4. 訓練
python train_ms_jp_extra.py --config Data/{model_name}/config.json --model Data/{model_name}/models

# 5. 推論テスト
python infer_test.py
```
