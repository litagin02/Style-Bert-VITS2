# Ver 3.0.0

- BERTモデルをローカルに持つのやめる
- 散らかったスクリプトをディレクトリ分けして整理
- `requirements.txt` でのインストールをやめて、uvでpyproject.tomlのdev dependenciesとしてすべて扱う
- ビルドはhatchからuvへ
- `faster_whisper` を廃止、TransforemrsライブラリのWhisperを常に使う
- `torch` 2.9系にする
- onnx関係は推論をサポートしない
- `pyopenjtalk-plus` を使う

TODO

- [ ] Colabチェック
- [ ] ffmpeg無くてもインストーラーがちゃんとffmpegをインストールしてくるかチェック
- [ ] CPUのやつでチェック
- [ ] 学習データの追跡性や署名的な何かを付けたい
- [ ] ライブラリとしての使用チェック
- [ ] テストアップデート
- [x] onnx変換がバグっているっぽいの直す
- [ ] pypiビルド
- [ ] フォークでの改善ロジックとか取り込む？
- [ ] etc
