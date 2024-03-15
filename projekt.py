import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
import datetime
from gensim.models import KeyedVectors
from keras.initializers import Constant
from constants import *
import util


def create_model(embedding_matrix):
    model = Sequential()
    embedding_layer = Embedding(MAX_VOCAB_SIZE,
                                100,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model.add(Embedding(MAX_VOCAB_SIZE, 100, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(100, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping])

def validate_model(model, X_val, y_val, X_train):
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    with open('model_metrics_over_time.txt', 'a') as file:
        now = str(datetime.datetime.now())
        train_size = len(X_train)
        val_size = len(X_val)
        file.write(f"{now}, loss: {loss:>10.4f}, accuracy: {accuracy:>10.4f}, train_size: {train_size}, val_size: {val_size}\n")

def main():
    X, y = util.get_sentiment140(TRAIN_DATA_FILE_PATH)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)

    tokenizer = Tokenizer(num_words=MAX_SEQUENCE_LENGTH)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    X_train_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    sequences = tokenizer.texts_to_sequences(X_val)
    X_val_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    glove = KeyedVectors.load_word2vec_format(FILTERED_GLOVE_FILE_PATH, binary=False, no_header=True)

    embedding_matrix = np.zeros((MAX_VOCAB_SIZE, 100))
    for word, i in tokenizer.word_index.items():
        if i < MAX_VOCAB_SIZE:
            try:
                embedding_vector = glove[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            except KeyError:
                continue

    for i in range(1):
        model = create_model(embedding_matrix)
        train_model(model, X_train_padded, y_train)
        validate_model(model, X_val_padded, y_val, X_train)
        model.save(MODEL_FILE_PATH)


if __name__ == "__main__":
    main()
