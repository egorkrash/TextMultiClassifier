import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer

EMBEDDING_DIM = 100
MAX_SEQ_LEN = 255


def clean_df(df, x=1, y=15):
    ind = df.Speaker.value_counts()[x:y].index
    return df[(df.Speaker.isin(ind)) & (~ df.Speaker.isin(['CANDIDATES', 'OTHER']))]


data = pd.read_csv('primary_debates_cleaned.csv')
data = clean_df(data)

LE = LabelEncoder()

Y_l_encoded = LE.fit_transform(data.Speaker.values)

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

for index, text in enumerate(data.Text.values):
    label_id = Y_l_encoded[index]
    labels_index[text] = label_id

    texts.append(text)
    labels.append(label_id)

tokenizer = Tokenizer(nb_words=9579)
tokenizer.fit_on_texts(texts)
