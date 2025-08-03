## ffmpegによる音声ファイルの一括処理

### 1. 目的

指定されたフォルダ内にある全ての音声ファイル（`.wav`, `.mp3`）に対し、以下の処理を一括で適用します。

1.  **リバーブ除去:** オーディオゲート（`agate`）を使い、音声の無音部分や響きをカットしてリバーブを軽減します。
2.  **音量正規化:** ラウドネスノーマライゼーション（`loudnorm`）を使い、全てのファイルの音量を均一な聴感レベルに調整します。

処理後のファイルは、元ファイルを上書きせず、`processed`という名前の新しいフォルダに保存されます。

### 2. 要件

*   **ffmpegがインストールされていること:** コマンドプロンプト（Windows）やターミナル（Linux/macOS）から`ffmpeg`コマンドが実行できる状態である必要があります。

### 3. コマンド

#### **Windows (バッチファイル)**

以下のコードを、`process_audio.bat` のような名前でテキストファイルに保存し、処理したい音声ファイルがあるフォルダに置いて実行します。

```batch
@echo off
setlocal

:: --- 設定 ---
:: このバッチファイルを置いた場所を基準とします
set "INPUT_DIR=%~dp0"
set "OUTPUT_DIR=%INPUT_DIR%processed"
set "FFMPEG_FILTER=agate=threshold=-35dB,loudnorm=I=-16:LRA=11:TP=-1.5"

:: --- 処理 ---
echo 出力先フォルダを作成します: %OUTPUT_DIR%
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo 処理を開始します...

:: .wav ファイルを処理
for %%f in ("%INPUT_DIR%*.wav") do (
    echo %%~nxf を処理中...
    ffmpeg -i "%%f" -af "%FFMPEG_FILTER%" "%OUTPUT_DIR%\%%~nxf"
)

:: .mp3 ファイルを処理
for %%f in ("%INPUT_DIR%*.mp3") do (
    echo %%~nxf を処理中...
    ffmpeg -i "%%f" -af "%FFMPEG_FILTER%" "%OUTPUT_DIR%\%%~nxf"
)

echo 全ての処理が完了しました。
pause
```

#### **Linux / macOS (シェルスクリプト)**

以下のコードを、`process_audio.sh` のような名前でテキストファイルに保存し、実行権限を与えてから実行します。

```bash
#!/bin/bash

# --- 設定 ---
# スクリプトを置いたディレクトリを基準にするか、下の絶対パスを指定します
# INPUT_DIR=$(pwd)
INPUT_DIR="/path/to/your/audio/files" # <-- 必ず実際の音声フォルダのパスに変更してください
OUTPUT_DIR="$INPUT_DIR/processed"
FFMPEG_FILTER="agate=threshold=-35dB,loudnorm=I=-16:LRA=11:TP=-1.5"

# --- 処理 ---
echo "出力先フォルダを作成します: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "処理を開始します..."

# .wav と .mp3 ファイルを処理
for f in "$INPUT_DIR"/*.{wav,mp3}; do
    # ファイルが存在する場合のみ処理
    if [ -f "$f" ]; then
        filename=$(basename -- "$f")
        echo "$filename を処理中..."
        ffmpeg -i "$f" -af "$FFMPEG_FILTER" "$OUTPUT_DIR/$filename"
    fi
done

echo "全ての処理が完了しました。"
```

### 4. コマンドの解説

*   **`for %%f in (*.wav *.mp3) do ...` (Windows)**
*   **`for f in *.wav *.mp3; do ... done` (Linux)**
    *   指定したフォルダ内の `.wav` と `.mp3` ファイルを一つずつ取り出して、ループ処理を実行します。

*   **`ffmpeg -i "入力ファイル" -af "フィルター" "出力ファイル"`**
    *   `-i`: 入力ファイルを指定します。
    *   `-af`: オーディオフィルター（Audio Filter）を指定します。複数のフィルターはカンマ `,` で区切ります。
    *   **`agate=threshold=-35dB`**: リバーブ除去用のフィルターです。
        *   `threshold=-35dB`: 音量が-35dB以下の部分をカットします。これにより、音が減衰していくリバーブ成分が除去されやすくなります。
    *   **`loudnorm=I=-16:LRA=11:TP=-1.5`**: 音量正規化用のフィルターです。
        *   `I=-16`: 目標の統合ラウドネス値を-16 LUFSに設定します。YouTubeなど多くのプラットフォームで採用されている標準的な値です。
        *   `LRA=11`: ラウドネス・レンジ（音量のばらつき）の目標値を設定します。
        *   `TP=-1.5`: トゥルーピーク（音の最大値）が-1.5dBFSを超えないようにします。これにより音割れ（クリッピング）を確実に防ぎます。

### 5. 実行方法

#### **Windows**

1.  上記のバッチファイルコードをコピーし、メモ帳などに貼り付けます。
2.  `process_audio.bat` という名前で、音声ファイルのあるフォルダ内に保存します。
3.  保存した `process_audio.bat` ファイルをダブルクリックして実行します。

#### **Linux / macOS**

1.  上記のシェルスクリプトコードをコピーし、テキストエディタに貼り付けます。
2.  `INPUT_DIR` の行を、実際の音声ファイルがあるフォルダの絶対パスに書き換えます。
3.  `process_audio.sh` という名前で保存します。
4.  ターミナルを開き、スクリプトを保存した場所に移動します。
5.  `chmod +x process_audio.sh` コマンドでスクリプトに実行権限を与えます。
6.  `./process_audio.sh` コマンドでスクリプトを実行します。
