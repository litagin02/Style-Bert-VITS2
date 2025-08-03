# wandb導入手順

## 1. wandbのインストール

`requirements.txt`に`wandb`を追加し、`uv`を使ってインストールします。

```bash
echo "wandb" >> requirements.txt
source venv/bin/activate
uv pip install -r requirements.txt
```

## 2. 学習スクリプトの修正

`train_ms_jp_extra.py`を修正し、`wandb`の初期化処理と、学習中の指標を`wandb`に送信する処理を追加します。

### 修正箇所

1.  **`import wandb`**: `wandb`ライブラリをインポートします。
2.  **`wandb.init()`**: `run()`関数の`SummaryWriter`を初期化している箇所の後で`wandb.init()`を呼び出し、`wandb`を初期化します。
3.  **`wandb.config`**: `wandb.init()`の引数に`config=hps`を渡すことで、ハイパーパラメータを`wandb`に保存します。
4.  **`wandb.log()`**: `train_and_evaluate`関数内の`utils.summarize`を呼び出している箇所で、`scalar_dict`の情報を`wandb.log()`を使って`wandb`に送信します。

## 3. wandbのAPIキー設定

環境変数`WANDB_API_KEY`にご自身のAPIキーを設定します。

```bash
export WANDB_API_KEY="YOUR_API_KEY"
```

## 4. 学習の実行

`train_ms_jp_extra.py`を実行して学習を開始します。

## 5. wandbのダッシュボードで確認

`wandb`のサイトで学習の進捗や指標を確認します。

## 6. Slack通知の設定

`wandb`のダッシュボードでSlackとの連携を設定し、学習の完了やエラーなどを通知するようにします。
