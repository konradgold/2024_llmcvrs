import datetime
import json

with open('judgeGPT/prompts/file1.json', 'r') as f:
    start = json.load(f)

init_from = "gpt2"
write_dir = "out/write_" + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + ".json"