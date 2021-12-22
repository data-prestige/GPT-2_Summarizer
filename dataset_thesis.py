import pandas as pd
from pre_processing import cleaner_d2v
import json
import os

df = pd.read_json( r'C:\Users\Mario\Desktop\Tor Vergata Data Science\Tesi Master\train.json')
df_2 = pd.read_json(r'C:\Users\Mario\Desktop\Tor Vergata Data Science\Tesi Master\test.json')
df_3 = pd.read_json(r'C:\Users\Mario\Desktop\Tor Vergata Data Science\Tesi Master\val.json')

data = pd.concat([df, df_2, df_3])

def write_json(i,article, abstract):
	""" Saves a json file."""

	file = os.path.join(r'C:\Users\Mario\Desktop\Tor Vergata Data Science\Tesi Master', 'chat_data', 'file_' + str(i) + '.json')
	js_example = {}
	js_example['id'] = i
	js_example['article'] = article
	js_example['abstract'] = abstract
	with open(file, 'w') as f:
		json.dump(js_example, f, ensure_ascii=False)

directory = 'chat_data'

def tokenizer_to_json(dataset, directory):
    tokenizer = cleaner_d2v.add_special_tokens('gpt2-large')
    train_ids = []
    i = 0
    for index, row in dataset.iterrows():
        article, abstract = tokenizer.encode(row['dialogue']), tokenizer.encode(row['summary'])
        if len(article) > 0 and len(abstract) > 0 and (len(article) + len(abstract)) <= 1023:
        	train_ids.append(i)
        	write_json(i, article, abstract)
        i += 1
        if i % 100 == 0:
            print(i, " files written")

    file = os.path.join(r'C:\Users\Mario\Desktop\Tor Vergata Data Science\Tesi Master', directory, 'index_chat_data.json')

    x, y = int(len(train_ids) * 0.8), int(len(train_ids) * 0.9)
    valid_ids = train_ids[x:y]
    test_ids = train_ids[y:]
    train_ids = train_ids[:x]
    with open(file, 'w') as f:
        js = {}
        js['train_ids'] = train_ids
        js['valid_ids'] = valid_ids
        js['test_ids'] = test_ids
        json.dump(js, f)

tokenizer_to_json(dataset = data, directory=directory)