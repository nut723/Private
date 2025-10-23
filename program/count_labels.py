import json
from collections import Counter

# Label StudioからエクスポートしたJSONファイルのパス
export_path = "/Users/okawara_natsumi/Downloads/annotations_export.json"

# ラベル件数を集計
label_counts = Counter()

with open(export_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for task in data:
    annotations = task.get("annotations", [])
    for ann in annotations:
        results = ann.get("result", [])
        for r in results:
            # "labels" フィールドを取り出す
            labels = r.get("value", {}).get("labels", [])
            for label in labels:
                label_counts[label] += 1

print("📊 ラベル別件数（Label Studioエクスポート解析）")
print("-" * 40)
for label, count in label_counts.most_common():
    print(f"{label}: {count} 件")

print("-" * 40)
print(f"🟢 合計: {sum(label_counts.values())} 件")
