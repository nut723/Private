# chat_ojousama_repl.py
import os
import sys
import time
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
LORA = "swallow8b-ojousama-lora"  # あなたの学習出力フォルダ

SYS = ()


# 省メモリ量子化（推論）
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def load_model():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb,
    )
    model = PeftModel.from_pretrained(base, LORA)
    model.eval()
    return tok, model

def build_prompt(tok, history):
    # history は [{"role":"system"/"user"/"assistant","content":...}, ...]
    return tok.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )

def generate_reply(tok, model, history, max_new_tokens=300, temperature=0.7, top_p=0.9):
    prompt = build_prompt(tok, history)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.07,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def main():
    print("Loading model...")
    tok, model = load_model()
    print("Ready. Type your message. (`/reset` to clear, `/save` to save log, `bye` to exit)")

    # 会話履歴（最初にsystem）
    history = [{"role": "system", "content": SYS}]
    transcript = []  # ログ保存用

    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break

        if user.lower() == "bye":
            print("Assistant> それではごきげんよう。どうぞお健やかにお過ごしくださいませ。")
            break
        if user == "/reset":
            history = [{"role": "system", "content": SYS}]
            transcript.clear()
            print("Assistant> 会話履歴をリセットいたしましたわ。")
            continue
        if user == "/save":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"ojousama_chatlog_{ts}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for turn in transcript:
                    f.write(json.dumps(turn, ensure_ascii=False) + "\n")
            print(f"Assistant> ログを保存いたしましたわ: {path}")
            continue
        if not user:
            continue

        # 履歴にユーザ発話を追加
        history.append({"role": "user", "content": user})
        # 生成
        reply = generate_reply(tok, model, history)
        # 履歴に応答を追加
        history.append({"role": "assistant", "content": reply})
        # ログに記録
        transcript.append({"user": user, "assistant": reply})
        # 表示
        print(f"Assistant> {reply}")

if __name__ == "__main__":
    main()
