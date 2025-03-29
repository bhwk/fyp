import json

with open("agent-workflow-results.json", "r") as f:
    data = json.load(f)

print(len(data))
