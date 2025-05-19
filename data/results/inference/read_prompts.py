import json

with open("prompts.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx,line in enumerate(lines[9:10]):
    key = f"Resume_{idx+10}"
    data = json.loads(line)
    print(data[key][0])