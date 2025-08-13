# inference_ojousama.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
LORA = "swallow8b-ojousama-lora"  # 学習結果

# 1) Tokenizer
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 2) 4bit量子化（推論も軽量に）
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 3) Base + LoRA を読み込む
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    quantization_config=bnb,
)
model = PeftModel.from_pretrained(base, LORA)
model.eval()

def chat(user_msg: str, sys_prompt: str | None = None, max_new_tokens: int = 300):
    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": user_msg})

    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    # 入力長以降 = 生成部分のみデコード
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    print(tok.decode(gen_ids, skip_special_tokens=True).strip())

if __name__ == "__main__":
    SYS = "あなたは常に上品なお嬢様口調で丁寧に応答します。語尾には『〜ですわ』『〜ましてよ』等を適度に用い、上品で思いやりのあるトーンを保ってください。"
    chat("自己紹介して", SYS)
