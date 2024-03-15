from keras.preprocessing.text import Tokenizer
from constants import *
import numpy as np
import util


X, y = util.get_sentiment140(TRAIN_DATA_FILE_PATH)

texts = X
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index


# Initialize an empty dictionary to hold the GloVe vectors for the tokens in your dataset
filtered_glove = {}

with open(GLOVE_TWITTER_27B_100D_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]  # The token
        if word in word_index:  # Check if this word is in your dataset's tokens
            vector = np.asarray(values[1:], dtype='float32')  # The vector for this word
            filtered_glove[word] = vector

# Optionally, save the filtered GloVe vectors for later use
with open(FILTERED_GLOVE_FILE_PATH, 'w', encoding='utf-8') as f:
    for word, vector in filtered_glove.items():
        f.write(word + " " + " ".join(map(str, vector)) + "\n")
