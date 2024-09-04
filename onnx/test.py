from transformers import AutoTokenizer
import onnxruntime


session = onnxruntime.InferenceSession("models/deberta-v2-large-jp-char-wwm_opt.onnx")
inputs = session.get_inputs()
output_name = session.get_outputs()[0].name


tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-large-japanese-char-wwm")
t_inputs = tokenizer("今日はいい天気です", return_tensors="pt")


result = session.run(
    [output_name],
    {
        inputs[0].name: t_inputs["input_ids"].detach().numpy(),
        inputs[1].name: t_inputs["attention_mask"].detach().numpy(),
    },
)
print(result)
