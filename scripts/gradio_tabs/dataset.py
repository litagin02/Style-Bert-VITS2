import gradio as gr

from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
):
    if model_name == "":
        return "Error: モデル名を入力してください。"
    logger.info("Start slicing...")
    cmd = [
        "scripts/slice.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    if time_suffix:
        cmd.append("--time_suffix")
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # onnxの警告が出るので無視する
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "音声のスライスが完了しました。"


def do_transcribe(
    model_name,
    model_id,
    language,
    initial_prompt,
    batch_size,
    num_beams,
):
    if model_name == "":
        return "Error: モデル名を入力してください。"
    if model_id == "litagin/anime-whisper":
        logger.info(
            "Since litagin/anime-whisper does not support initial prompt, it will be ignored."
        )
        initial_prompt = ""

    cmd = [
        "scripts/transcribe.py",
        "--model_name",
        model_name,
        "--model_id",
        model_id,
        "--language",
        language,
        "--initial_prompt",
        f'"{initial_prompt}"',
        "--num_beams",
        str(num_beams),
        "--batch_size",
        str(batch_size),
    ]
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}. エラーメッセージが空の場合、何も問題がない可能性があるので、書き起こしファイルをチェックして問題なければ無視してください。"
    return "音声の文字起こしが完了しました。"


how_to_md = """
Style-Bert-VITS2の学習用データセットを作成するためのツールです。以下の2つからなります。

- 与えられた音声からちょうどいい長さの発話区間を切り取りスライス
- 音声に対して文字起こし

このうち両方を使ってもよいし、スライスする必要がない場合は後者のみを使ってもよいです。**コーパス音源などすでに適度な長さの音声ファイルがある場合はスライスは不要**です。

## 必要なもの

学習したい音声が入った音声ファイルいくつか（形式はwav以外でもmp3, ogg, flac, opus, m4a等も可能）。
合計時間がある程度はあったほうがいいかも、10分とかでも大丈夫だったとの報告あり。単一ファイルでも良いし複数ファイルでもよい。

## スライス使い方
1. `inputs`フォルダに音声ファイルをすべて入れる（スタイル分けをしたい場合は、サブフォルダにスタイルごとに音声を分けて入れる）
2. `モデル名`を入力して、設定を必要なら調整して`音声のスライス`ボタンを押す
3. 出来上がった音声ファイルたちは`Data/{モデル名}/raw`に保存される

## 書き起こし使い方

1. `Data/{モデル名}/raw`に音声ファイルが入っていることを確認（直下でなくてもよい、wav以外の形式も対応）
2. 設定を必要なら調整してボタンを押す
3. 書き起こしファイルは`Data/{モデル名}/esd.list`に保存される

## 手動で書き起こしファイルを作成する場合

コーパス等で既に書き起こしがある場合は、手動で`esd.list`を作成できます。

1. `Data/{モデル名}/raw`に音声ファイルを配置（wav, mp3, ogg, flac等どの形式でも可）
2. `Data/{モデル名}/esd.list`を以下の形式で作成:
   ```
   サブフォルダ/ファイル名.ogg|話者名|JP|書き起こしテキスト
   ```
   **音声ファイルの実際の拡張子をそのまま記載してください**（例: `.ogg`, `.mp3`等）
3. 学習タブで前処理を実行すると、音声は自動的にwav形式に変換され、`esd.list`内のパスも自動で更新されます

## 注意

- 長すぎる音声があるとVRAM消費量が増えたり安定しなかったりするので、適度な長さ（2〜12秒程度）にスライスすることをおすすめします。
- 書き起こしの結果をどれだけ修正すればいいかはデータセットに依存しそうです。
"""


def create_dataset_app():
    with gr.Blocks() as app:
        gr.Markdown("## データセット作成")
        gr.Markdown("音声ファイル（1つ以上）から、学習用のデータセットを作成します。")
        with gr.Accordion("使い方", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="モデル名を入力してください（話者名としても使われます）。"
        )
        with gr.Accordion("音声のスライス"):
            gr.Markdown(
                "**すでに適度な長さの音声ファイルからなるデータがある場合は、その音声をData/{モデル名}/rawに入れれば、このステップは不要です。**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="元音声の入っているフォルダパス",
                        value="inputs",
                        info="下記フォルダにwavやmp3等のファイルを入れておいてください",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="この秒数未満は切り捨てる",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="この秒数以上は切り捨てる",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="無音とみなして区切る最小の無音の長さ（ms）",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="WAVファイル名の末尾に元ファイルの時間範囲を付与する",
                    )
                    slice_button = gr.Button("スライスを実行")
                result1 = gr.Textbox(label="結果")
        with gr.Row():
            with gr.Column():
                model_id = gr.Dropdown(
                    [
                        "openai/whisper-large-v3-turbo",
                        "openai/whisper-large-v3",
                        "openai/whisper-large-v2",
                        "kotoba-tech/kotoba-whisper-v2.1",
                        "litagin/anime-whisper",
                    ],
                    label="Whisperモデル (Hugging Face)",
                    value="openai/whisper-large-v3-turbo",
                    allow_custom_value=True,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    label="バッチサイズ",
                    info="大きくすると速度が速くなるがVRAMを多く使う",
                )
                # v3.0.0以降は日本語のみサポート
                language = gr.Dropdown(["ja"], value="ja", label="言語")
                initial_prompt = gr.Textbox(
                    label="初期プロンプト",
                    value="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
                    info="このように書き起こしてほしいという例文（句読点の入れ方・笑い方・固有名詞等）",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="ビームサーチのビーム数",
                    info="小さいほど速度が上がる（以前は5）",
                )
            transcribe_button = gr.Button("音声の文字起こし")
            result2 = gr.Textbox(label="結果")
        slice_button.click(
            do_slice,
            inputs=[
                model_name,
                min_sec,
                max_sec,
                min_silence_dur_ms,
                time_suffix,
                input_dir,
            ],
            outputs=[result1],
        )
        transcribe_button.click(
            do_transcribe,
            inputs=[
                model_name,
                model_id,
                language,
                initial_prompt,
                batch_size,
                num_beams,
            ],
            outputs=[result2],
        )

    return app


if __name__ == "__main__":
    app = create_dataset_app()
    app.launch(inbrowser=True)
