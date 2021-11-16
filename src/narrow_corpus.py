import json
from tqdm import tqdm
import pandas as pd
import sqlite3

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("/home/kw/Data/FEVEROUS/feverous_wikiv1.db")
cur = con.cursor()
selected_file = "/home/kw/Data/FEVEROUS/train.pages.p20.jsonl"
num_lines = 1000

pages = set()

with open(selected_file) as f:
    maybe_line = f.readline()

    for i in tqdm(range(num_lines)):

        datapoint = json.loads(maybe_line)
        maybe_line = f.readline()

        predicted_pages = [p[0] for p in datapoint['predicted_pages']]

        for page in predicted_pages:
            pages.add(page)

rows = cur.execute("select * from wiki where id='10';")

for row in rows:
    print(row)
