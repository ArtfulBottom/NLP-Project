import pandas as pd
import numpy as np
import os
import string

np.random.seed(71)

texts = [] 
labels = []
table = str.maketrans('', '', string.punctuation)

directories = ['../aclimdb/train/pos/', '../aclimdb/train/neg/', '../aclimdb/test/pos/', '../aclimdb/test/neg/']
sentiments = [1, 0, 1, 0]

def clean(text):
    text = text.lower()
    words = text.split()
    
    stripped = [w.translate(table) for w in words]
    stripped = list(filter(None, stripped))

    return ' '.join(stripped)

for i, directory in enumerate(directories):
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as f:
            text = f.read()

            texts.append(clean(text))
            labels.append(sentiments[i])

np_array_texts = np.array(texts)
np_array_labels = np.array(labels)

np_array = np.concatenate((np_array_texts[:, np.newaxis], np_array_labels[:, np.newaxis]), axis=1)
np.random.shuffle(np_array)

df = pd.DataFrame(data=np_array, columns=['text', 'label'])
df.to_csv('data/imdb.csv', index=False)
