import onnxruntime
from style_bert_vits2.models.infer import get_net_g, get_text
from style_bert_vits2.models.hyper_parameters import HyperParameters
import torch
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.tts_model import TTSModel
from pathlib import Path
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
from style_bert_vits2.nlp import (
    clean_text,
    cleaned_text_to_sequence,
    bert_models,
)
import soundfile as sf
import numpy as np
from time import time
from typing import Optional


bert_session = onnxruntime.InferenceSession(
    "models/deberta-v2-large-jp-char-wwm_opt.onnx"
)


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    tokenizer = bert_models.load_tokenizer(Languages.JP)

    inputs = tokenizer(text, return_tensors="pt")
    res = bert_session.run(
        [output_name],
        {
            "input_ids": inputs["input_ids"].detach().numpy(),
            "attention_mask": inputs["attention_mask"].detach().numpy(),
        },
    )[0]

    assert len(word2ph) == len(text) + 2, text
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = np.tile(res[i], (word2phone[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T


def get_text_onnx(
    text: str,
    hps: HyperParameters,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
):
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph = clean_text(
        text,
        "JP",
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, "JP")

    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    assert bert_ori.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone, tone, language = np.array(phone), np.array(tone), np.array(language)
    return bert_ori, phone, tone, language


bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")


model_file = ".model_assets/tsukuyomi/tyc_e100_s5300.safetensors"
config_file = "model_assets/tsukuyomi/config.json"
style_file = "model_assets/tsukuyomi/style_vectors.npy"
assets_root = Path("../model_assets")


session = onnxruntime.InferenceSession(
    "models/model_opt.onnx", providers=["CPUExecutionProvider"]
)
session.get_modelmeta()
inputs = session.get_inputs()
output_name = session.get_outputs()[0].name

helper = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device="cpu",
)

with open("content.txt", "r") as f:
    text = f.read()

bert, x_tst, tones, lang_ids = get_text_onnx(
    text,
    helper.hyper_parameters,
    "cpu",
    assist_text=None,
    assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
)

device = "cpu"
style_id = helper.style2id[DEFAULT_STYLE]
style_vector = helper.get_style_vector(style_id, DEFAULT_STYLE_WEIGHT)

bert = np.expand_dims(bert, axis=0)
x_tst_lengths = np.array([x_tst.shape[0]], dtype=np.int64)
x_tst = np.expand_dims(x_tst, axis=0)
tones = np.expand_dims(tones, axis=0)
lang_ids = np.expand_dims(lang_ids, axis=0)
style_vec_tensor = np.expand_dims(style_vector, axis=0)

before = time()
output = session.run(
    [output_name],
    {
        inputs[0].name: x_tst,
        inputs[1].name: x_tst_lengths,
        inputs[2].name: np.array([0], dtype=np.int64),
        inputs[3].name: tones,
        inputs[4].name: lang_ids,
        inputs[5].name: bert,
        inputs[6].name: style_vec_tensor,
    },
)
print(time() - before)
audio = helper.convert_to_16_bit_wav(output[0][0, 0])
sf.write("v2.output.wav", audio, 44100)
