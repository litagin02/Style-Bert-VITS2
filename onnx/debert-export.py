from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn


class ORTDeberta(nn.Module):
    def __init__(self, model_name):
        super(ORTDeberta, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, input_ids, token_type_ids, attention_mask):
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        res = self.model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        return res


model = ORTDeberta("ku-nlp/deberta-v2-large-japanese-char-wwm")

tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-large-japanese-char-wwm")
inputs = tokenizer("今日はいい天気ですね", return_tensors="pt")

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]),
    "models/deberta-v2-large-jp-char-wwm.onnx",
    input_names=["input_ids", "token_type_ids", "attention_mask"],
    output_names=["output"],
    verbose=True,
    dynamic_axes={
        "input_ids": {1: "batch_size"},
        "attention_mask": {1: "batch_size"}
    }
)