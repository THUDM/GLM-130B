import json
import tqdm
from icetk import icetk
from multiprocessing import Pool

DATA_PATH = "/mnt/yrfs/aohan/data/english_data/pile/val.jsonl"
OUTPUT_PATH = "/mnt/yrfs/aohan/data/english_data/pile/val_tokenized.jsonl"


def get_data(line):
    item = json.loads(line)
    item["text_pretokenized"] = item["text"]
    item["text"] = icetk.encode(item["text_pretokenized"])
    return json.dumps(item) + "\n"


with open(DATA_PATH, "r") as file:
    data = file.readlines()

with Pool(16) as p:
    result = list(tqdm.tqdm(p.imap(get_data, data), total=len(data)))

with open(OUTPUT_PATH, "w") as file:
    file.writelines(result)
