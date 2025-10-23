import os
import json

data_dir = "/Users/okawara_natsumi/Downloads/KokoroChat-main/kokorochat_dialogues"
output_high = "/Users/okawara_natsumi/Downloads/kokorochat_high_for_labelstudio.json"
output_low = "/Users/okawara_natsumi/Downloads/kokorochat_low_for_labelstudio.json"

high_data = []
low_data = []

for filename in os.listdir(data_dir):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(data_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # list or dict
    items = data if isinstance(data, list) else [data]

    for item in items:
        # 点数を確認
        review = item.get("review_by_client_jp", {})
        score = review.get("点数")

        if score is None:
            continue

        dialogue = item.get("dialogue", [])
        if not dialogue:
            continue

        # テキスト連結
        dialogue_text = "\n".join([
            f"{turn['role']}: {turn['utterance']}" for turn in dialogue
        ])

        entry = {"data": {"text": dialogue_text}}

        if 98 > score >= 70:
            high_data.append(entry)
        elif score < 70:
            low_data.append(entry)

# 出力
with open(output_high, "w", encoding="utf-8") as out_high:
    json.dump(high_data, out_high, ensure_ascii=False, indent=2)

with open(output_low, "w", encoding="utf-8") as out_low:
    json.dump(low_data, out_low, ensure_ascii=False, indent=2)

print(f"✅ Highスコア: {len(high_data)}件 → {output_high}")
print(f"✅ Lowスコア: {len(low_data)}件 → {output_low}")
