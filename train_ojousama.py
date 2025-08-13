# train_ojousama.py ー packing無しでステップ数を確保する版
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
DATA_PATH  = "ojousama_from_matsu.jsonl"
OUTPUT_DIR = "swallow8b-ojousama-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map="auto", torch_dtype="auto",
    low_cpu_mem_usage=True, quantization_config=bnb,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, peft_config)

def print_trainable_params(m):
    t = s = 0
    for _, p in m.named_parameters():
        n = p.numel(); s += n
        if p.requires_grad: t += n
    print(f"trainable params: {t:,} / {s:,} ({t/s*100:.4f}%)")
print_trainable_params(model)

raw = load_dataset("json", data_files=DATA_PATH, split="train")

SYSTEM_PROMPT = (
    "あなたは常に上品で礼節ある『お嬢様』の話し方で応答します。"
    "出自や企業名・モデル名は名乗りません。"
)

def to_text(ex):
    ins = (ex.get("instruction") or "").strip()
    out = (ex.get("output") or "").strip()
    msgs = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":ins},
        {"role":"assistant","content":out},
    ]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {"text": txt}

ds_text = raw.map(to_text, remove_columns=raw.column_names)

# ★ packingしない：各例をそのままトークナイズ（右パディング）
MAX_LEN = 512  # まずは512。余裕あれば768〜1024
def tok(batch):
    return tokenizer(batch["text"], max_length=MAX_LEN, padding="max_length",
                     truncation=True, add_special_tokens=False)
ds_tok = ds_text.map(tok, batched=True, remove_columns=["text"])

# labels = input_ids（全トークン損失の簡易SFT）
def add_labels(ex):
    ex["labels"] = ex["input_ids"].copy()
    return ex
ds_tok = ds_tok.map(add_labels)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,                 # ←増やす
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # 実効BS=16
    learning_rate=5e-4,                 # ←やや上げる
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    data_collator=collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("done ->", OUTPUT_DIR)
