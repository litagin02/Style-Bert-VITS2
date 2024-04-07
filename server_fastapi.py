"""
API server for TTS
TODO: server_editor.pyと統合する?
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from scipy.io import wavfile

from config import config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


ln = config.server_config.language


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

# 事前に BERT モデル/トークナイザーをロードしておく
## ここでロードしなくても必要になった際に自動ロードされるが、時間がかかるため事前にロードしておいた方が体験が良い
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)
bert_models.load_model(Languages.EN)
bert_models.load_tokenizer(Languages.EN)
bert_models.load_model(Languages.ZH)
bert_models.load_tokenizer(Languages.ZH)


def raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )


class SynthesisRequest(BaseModel):
    text: str
    encoding: Optional[str] = None
    model_id: int = 0
    speaker_name: Optional[str] = None
    speaker_id: int = 0
    sdp_ratio: float = DEFAULT_SDP_RATIO
    noise: float = DEFAULT_NOISE
    noisew: float = DEFAULT_NOISEW
    length: float = DEFAULT_LENGTH
    language: Languages = ln
    auto_split: bool = DEFAULT_LINE_SPLIT
    split_interval: float = DEFAULT_SPLIT_INTERVAL
    assist_text: Optional[str] = None
    assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT
    style: Optional[str] = DEFAULT_STYLE
    style_weight: float = DEFAULT_STYLE_WEIGHT
    reference_audio_path: Optional[str] = None


class AudioResponse(Response):
    media_type = "audio/wav"


loaded_models: list[TTSModel] = []


def load_models(model_holder: TTSModelHolder):
    global loaded_models
    loaded_models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = TTSModel(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        # 起動時に全てのモデルを読み込むのは時間がかかりメモリを食うのでやめる
        # model.load()
        loaded_models.append(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path(args.dir)
    model_holder = TTSModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)

    limit = config.server_config.limit
    app = FastAPI()
    allow_origins = config.server_config.origins
    if allow_origins:
        logger.warning(
            f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # app.logger = logger
    # ↑効いていなさそう。loggerをどうやって上書きするかはよく分からなかった。

    @app.api_route("/api/style-bert-vits2/synthesis", methods=["POST"], response_class=AudioResponse)
    async def voice(request: Request, params: SynthesisRequest):
        text = params.text
        encoding = params.encoding
        model_id = params.model_id
        speaker_name = params.speaker_name
        speaker_id = params.speaker_id
        sdp_ratio = params.sdp_ratio
        noise = params.noise
        noisew = params.noisew
        length = params.length
        language = params.language
        auto_split = params.auto_split
        split_interval = params.split_interval
        assist_text = params.assist_text
        assist_text_weight = params.assist_text_weight
        style = params.style
        style_weight = params.style_weight
        reference_audio_path = params.reference_audio_path

        """Infer text to speech(テキストから感情付き音声を生成する)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        if request.method == "GET":
            logger.warning(
                "The GET method is not recommended for this endpoint due to various restrictions. Please use the POST method."
            )
        if model_id >= len(
            model_holder.model_names
        ):  # /models/refresh があるためQuery(le)で表現不可
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        model = loaded_models[model_id]
        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(
                    f"speaker_id={speaker_id} not found", "speaker_id"
                )
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(
                    f"speaker_name={speaker_name} not found", "speaker_name"
                )
            speaker_id = model.spk2id[speaker_name]
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        assert style is not None
        if encoding is not None:
            text = unquote(text, encoding=encoding)
        sr, audio = model.infer(
            text=text,
            language=language,
            speaker_id=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )
        logger.success("Audio data generated and sent successfully")
        with BytesIO() as wavContent:
            wavfile.write(wavContent, sr, audio)
            return Response(content=wavContent.getvalue(), media_type="audio/wav")

    @app.get("/api/style-bert-vits2/models/info")
    def get_loaded_models_info():
        """ロードされたモデル情報の取得"""

        result: dict[str, dict[str, Any]] = dict()
        for model_id, model in enumerate(loaded_models):
            result[str(model_id)] = {
                "config_path": model.config_path,
                "model_path": model.model_path,
                "device": model.device,
                "spk2id": model.spk2id,
                "id2spk": model.id2spk,
                "style2id": model.style2id,
            }
        return result

    @app.post("/api/style-bert-vits2/models/refresh")
    def refresh():
        """モデルをパスに追加/削除した際などに読み込ませる"""
        model_holder.refresh()
        load_models(model_holder)
        return get_loaded_models_info()

    @app.get("/api/style-bert-vits2/status")
    def get_status():
        """実行環境のステータスを取得"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_total = memory_info.total
        memory_available = memory_info.available
        memory_used = memory_info.used
        memory_percent = memory_info.percent
        gpuInfo = []
        devices = ["cpu"]
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    @app.get("/api/style-bert-vits2/tools/get_audio", response_class=AudioResponse)
    def get_audio(
        request: Request, path: str = Query(..., description="local wav path")
    ):
        """wavデータを取得する"""
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
        )
        if not os.path.isfile(path):
            raise_validation_error(f"path={path} not found", "path")
        if not path.lower().endswith(".wav"):
            raise_validation_error(f"wav file not found in {path}", "path")
        return FileResponse(path=path, media_type="audio/wav")

    logger.info(f"server listen: http://127.0.0.1:{config.server_config.port}")
    logger.info(f"API docs: http://127.0.0.1:{config.server_config.port}/docs")
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
