import pandas as pd
from pre_processing import cleaner_d2v
import numpy as np
import re
import json
import os

'''
Data Cleaning
'''

df = pd.read_csv(r'C:\Users\Mario\Desktop\Summarization_project\video_giornalismo_dataset.csv', sep = ';')

# As one can notice, the title and Sottotitolo are included at the start of the description and they will be our final label
df['description_clean'] = df.apply(lambda row : row['description'].replace(str(row['title']), ''), axis=1)
df['description_clean'] = df.apply(lambda row : row['description_clean'].replace(str(row['Sottotitolo']), ''), axis=1)
# Notice that we needed to remove them sequently because weird spacing occurs if title + Sottotitolo are merged first

# Grab everything between inverted commas
df['description_talk'] = df['description_clean'].str.findall(r'\"([^()]+)\"')

# Remove +++ AAA Contents +++
df['description_talk'] = df['description_clean'].str.replace(r'\+([^()]+)\+', '')

# Put nan if empty list
df['description_talk'] = df.description_talk.apply(lambda x: np.nan if len(x)==0 else x)

# From list to Series
df['description_talk'] = df['description_talk'].apply(pd.Series).iloc[:,0]

# So that we can now fill nan values with atypical description
df['description_filled'] = df["description_talk"].fillna(df["description_clean"])

# Extract text within brackets for ex: (Roma)... --> Roma
regex = re.compile(".*?\((.*?)\)")
df['city'] = df['description_filled'].str.extract(regex)

# Make tags Uppecase as in text
df['tags'] = df['tags'].str.upper() # all uppercase

# Remove city and tag
df['description_filled'] = df.apply(lambda row : row['description_filled'].replace(str(row['city']), ''), axis=1)
df['description_filled'] = df.apply(lambda row : row['description_filled'].replace(str(row['tags']), ''), axis=1)

# Get label from title and Sottotitolo
df['label'] = df['title'] + '.' + '\n' + df['Sottotitolo']

# Clean text
# df['text'] = df.description_filled.apply(lambda x: cleaner_d2v.text_cleaning(x))
# data = df[['text', 'label']]

df['description_filled'] = df.apply(lambda row : row['description_filled'].replace('\n', ''), axis=1)
df['description_filled'] = df.apply(lambda row : row['description_filled'].replace('().', ''), axis=1)


df['description_filled'] = df.apply(lambda row : row['description_filled'].replace(r"\'", "'"), axis=1)
df['description_filled'] = df['description_filled'].str.replace(r'\(([^()]+)\)', '')
df['description_filled'] = df.apply(lambda row : row['description_filled'].replace('SPORT', ''), axis=1)
df['description_filled'] = df.apply(lambda row : row['description_filled'].replace('SPETTACOLO', ''), axis=1)
df['description_filled'] = df.apply(lambda row : row['description_filled'].replace('APPROFONDIMENTI', ''), axis=1)
df['description_filled'] = df.apply(lambda row : row['description_filled'].replace('+++RIPETIZIONE CORRETTA+++', ''), axis=1)


data = df[['description_filled', 'label']]
data['description_filled'] = df.apply(lambda row : row['description_filled'].replace('"', ''), axis=1)
data['description_filled'] = data['description_filled'].str[2:]
data['description_filled'] = data['description_filled'].replace(' ', np.NaN)
data['description_filled'] = data['description_filled'].replace('  ', np.NaN)

data['description_filled'].fillna(df['description'], inplace = True)
data['description_filled'] = data.apply(lambda row : row['description_filled'].replace('\n', ''), axis=1)
data = data.drop(213) # dropping row 213, very weird text

'''
Write to Json tokenized txt
'''

def write_json(i,article, abstract):
	""" Saves a json file."""

	file = os.path.join(os.getcwd(), 'articoli', 'file_' + str(i) + '.json')
	js_example = {}
	js_example['id'] = i
	js_example['article'] = article
	js_example['abstract'] = abstract
	with open(file, 'w') as f:
		json.dump(js_example, f, ensure_ascii=False)

directory = 'articoli'

def tokenizer_to_json(dataset, directory):
    tokenizer = cleaner_d2v.add_special_tokens()
    train_ids = []
    i = 0
    for index, row in dataset.iterrows():
        article, abstract = tokenizer.encode(row['description_filled']), tokenizer.encode(row['label'])
        if len(article) > 0 and len(abstract) > 0 and (len(article) + len(abstract)) <= 1023:
        	train_ids.append(i)
        	write_json(i, article, abstract)
        i += 1
        if i % 100 == 0:
            print(i, " files written")

    file = os.path.join(os.getcwd(), directory, 'index_articoli.json')

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


'''
Plot text length
'''


import matplotlib.pyplot as plt

data['len_desc'] = data.description_filled.apply(lambda x: x.split(' '))
data['len_desc'] = data.len_desc.apply(lambda x: len(x))

data['len_label'] = data.label.apply(lambda x: x.split(' '))
data['len_label'] = data.len_label.apply(lambda x: len(x))

data['txt_length'] = data.len_desc + data.len_label

# plot the distribution of articles sizes
plt.hist(data['txt_length'], color='green', bins=6, edgecolor='black')
plt.title("Files_Distribution_By_Size(no. of words)")
plt.xlabel('No Of Words')
plt.ylabel('Files')
plt.show()
plt.savefig(" files distribution by length")

'''
Train GPT-2 with GEPPETTO model
'''
















