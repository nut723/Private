from datasets import load_dataset
import json, re

CSV_URL = "https://raw.githubusercontent.com/matsuvr/OjousamaTalkScriptDataset/main/ojousamatalkscript200.csv"
OUT = "ojousama_from_matsu.jsonl"

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# CSVの列名は prompt / completion
ds = load_dataset("csv", data_files=CSV_URL, split="train")

count = 0
with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        user = clean(ex.get("prompt", ""))
        ojou = clean(ex.get("completion", ""))
        if not user or not ojou:
            continue
        rec = {"instruction": user, "input": "", "output": ojou}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1

print("wrote", count, "records ->", OUT)
