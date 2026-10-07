# Style-Bert-VITS2（学習コア）

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) のフォークで、**日本語・JP-Extra 一本化の学習コア**です。前処理〜学習（CLI）、推論ライブラリ、学習済みモデルの試聴GUIを提供します（学習曲線は TensorBoard で確認）。
"お願いとデフォルトモデルの利用規約"もフォーク元Style-Bert-VITS2に準じます。

- データセット作成（スライス・ASR・リサンプリング等）・評価・アプリとしての推論（エディター・APIサーバー）は本リポジトリでは扱いません。`style-bert-vits2` をライブラリとして依存する分離先リポジトリで利用してください
- **言語は日本語のみ**です。英語はカタカナ入力が前提です（細かい読みは `dict_data/` のユーザー辞書で調整）

## クイックスタート

```bash
# 1. インストール（GPU環境では先に torch を入れてから uv sync でも可）
uv sync                      # BERT・JP-Extra事前学習モデル・SLM は前処理時に自動で取得されます（手動一括取得は `uv run -m train.initialize`）

# 2. Data/<モデル名>/ に esd.list + raw.zip を置く → 前処理 → 学習
uv run preprocess_all.py -m <モデル名>   # 前処理（データセット検証〜特徴抽出）
uv run train_model.py -m <モデル名>       # 学習（新規も再開もこのコマンド。再開時はスタイル生成が自動でスキップされます）

# 3. 試聴
uv run app.py                            # 試聴GUI（学習済みモデルを音声合成して確認）
```

詳しい手順（データセット契約・各ステップ・成果物・ライブラリ利用）は **[実操作ガイド docs/GUIDE.md](/docs/GUIDE.md)** を参照してください。

## ディレクトリ構成

```
style_bert_vits2/   ①コアライブラリ（推論: tts_model / nlp / models）
train/              ②学習（pipeline・データローダ・損失関数）
preprocess/         学習前処理チェーン（検証→テキスト→BERT特徴→スタイル特徴）
gradio_tabs/ + app.py  ③試聴GUI（音声合成タブのみ）
Data/               学習データセット（esd.list + raw.zip を置く場所）
pretrained/         事前学習モデル（bert / jp_extra / slm）
model_assets/       学習済みモデル資産（推論・共有用3点セット）
dict_data/          pyopenjtalk ユーザー辞書
```

## ドキュメント

- **[docs/GUIDE.md](/docs/GUIDE.md)** — 実操作ガイド（データセット→学習→成果物→試聴）
- [docs/FAQ.md](/docs/FAQ.md) — よくある質問
- [docs/SPLIT_PLAN.md](/docs/SPLIT_PLAN.md) — 本コアの分割設計方針
- `docs/upstream/` — フォーク元リポジトリのドキュメント（歴史・アーキテクチャ背景としての参照用）