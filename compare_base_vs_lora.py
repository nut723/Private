# compare_base_vs_lora.py  ― LoRA差を同一プロンプトで比較（few-shotなし）
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_ID = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
LORA_DIR = "swallow8b-ojousama-lora"

SYS = ()

BQ = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        BASE_ID, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, quantization_config=BQ
    )
    m.eval()
    return tok, m

def load_lora():
    tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, quantization_config=BQ
    )
    m = PeftModel.from_pretrained(base, LORA_DIR)
    m.eval()
    return tok, m

def build_inputs(tok, user_msg):
    msgs = [{"role":"system","content":SYS},
            {"role":"user","content":user_msg}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt")

@torch.no_grad()
def generate(model, tok, inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, seed=7):
    torch.manual_seed(seed)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p, repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
    )
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

def main():
    user_msg = "自己紹介して"  # ←ここを変えれば同条件で比較できます

    print("=== Loading base (no LoRA) ===")
    tok_b, m_b = load_base()
    print("=== Loading LoRA ===")
    tok_l, m_l = load_lora()

    inp_base = build_inputs(tok_b, user_msg)
    inp_lora = build_inputs(tok_l, user_msg)

    print("\n--- BASE OUTPUT ---")
    print(generate(m_b, tok_b, inp_base))

    print("\n--- LoRA OUTPUT ---")
    print(generate(m_l, tok_l, inp_lora))

if __name__ == "__main__":
    main()
