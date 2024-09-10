from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from pathlib import Path
from huggingface_hub import hf_hub_download
from style_bert_vits2.models.infer import get_net_g, get_text
from style_bert_vits2.models.hyper_parameters import HyperParameters
import torch
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
from style_bert_vits2.tts_model import TTSModel
import numpy as np


bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")


model_file = "model_assets/tsukuyomi/tsukuyomi_e100_s2700.safetensors"
config_file = "model_assets/tsukuyomi/config.json"
style_file = "model_assets/tsukuyomi/style_vectors.npy"
assets_root = Path("model_assets")

for file in [model_file, config_file, style_file]:
    print(file)
    hf_hub_download("tuna2134/tsukuyomi-v2", file, local_dir="model_assets")


hyper_parameters = HyperParameters.load_from_json(assets_root / config_file)


"""
model = get_net_g(
    str(assets_root / model_file),
    hyper_parameters.version,
    "cpu",
    hyper_parameters,
    train_mode=False,
)
"""

text = "なんで動かないの"

bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
    text,
    Languages.JP,
    hyper_parameters,
    "cpu",
    assist_text=None,
    assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
    given_phone=None,
    given_tone=None,
)

lol = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device="cpu",
)
device = "cpu"
style_id = lol.style2id[DEFAULT_STYLE]
style_vector = lol.get_style_vector(style_id, DEFAULT_STYLE_WEIGHT)

x_tst = phones.to(device).unsqueeze(0)
tones = tones.to(device).unsqueeze(0)
lang_ids = lang_ids.to(device).unsqueeze(0)
bert = bert.to(device).unsqueeze(0)
ja_bert = ja_bert.to(device).unsqueeze(0)
en_bert = en_bert.to(device).unsqueeze(0)
x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
style_vec_tensor = torch.from_numpy(style_vector).to(device).unsqueeze(0)

model = get_net_g(
    str(assets_root / model_file),
    hyper_parameters.version,
    device,
    hyper_parameters,
    train_mode=False,
)

torch.onnx.export(
    model,
    (
        x_tst,
        x_tst_lengths,
        torch.LongTensor([0]).to(device),
        tones,
        lang_ids,
        bert,
        style_vec_tensor,
    ),
    "model.onnx",
    verbose=True,
    dynamic_axes={
        "x_tst": {1: "batch_size"},
        "x_tst_lengths": {0: "batch_size"},
        "tones": {1: "batch_size"},
        "language": {1: "batch_size"},
        "bert": {2: "batch_size"},
    },
    input_names=[
        "x_tst",
        "x_tst_lengths",
        "sid",
        "tones",
        "language",
        "bert",
        "ja_bert",
        "en_bert
        "style_vec",
    ],
    output_names=["output"],
)
