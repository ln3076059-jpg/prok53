import json

with open("kaggle_log.txt", "r", encoding="utf-8") as f:
    content = f.read()

with open("metrics_output.txt", "w", encoding="utf-8") as out:
    try:
        logs = json.loads(content)
        for entry in logs:
            if "data" in entry:
                data = entry["data"]
                if "all" in data and "Images" not in data:
                    out.write(data.strip() + "\n")
                elif "top1" in data.lower():
                    out.write(data.strip() + "\n")
                elif "Speed:" in data:
                    out.write(data.strip() + "\n")
                elif "Results saved to" in data:
                    out.write(data.strip() + "\n")
    except Exception as e:
        out.write(f"Error parsing json: {e}\n")
