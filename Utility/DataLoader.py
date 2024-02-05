import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DataLoader(object):

    def __init__(self):
        self.train_comments = None
        self.val_comments = None
        self.test_comments = None

    def load_dataset(self, train_x, train_y, val_x, val_y, test_x, list_classes):

        list_sentences_train = train_x['string'].fillna("no comment").values
        list_sentences_val = val_x['string'].fillna("no comment").values
        train_comments = []
        for text in list_sentences_train:
            train_comments.append(text)
        self.train_comments = train_comments
        val_comments = []
        for text in list_sentences_val:
            val_comments.append(text)
        self.val_comments = val_comments

        list_sentences_test = test_x['string'].fillna("no comment").values
        test_comments = []
        for text in list_sentences_test:
            test_comments.append(text)
        self.test_comments = test_comments

        train_y = train_y[list_classes].values
        val_y = val_y[list_classes].values
        print('Shape of train_y :', train_y.shape)
        print('Shape of val_y :', val_y.shape)
        return train_y, val_y

    
    def load_embedding(self, embedding_paths, keras_like=True):
        if keras_like:
            embedding_index = {}
            for path in embedding_paths:
                f = open(path, 'r', encoding = 'utf-8')
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word not in embedding_index:
                        coefs = np.asarray(values[1:], dtype='float32')
                        embedding_index[word] = coefs
            f.close()
            
            print('Total %s word vectors' % len(embedding_index))
            return embedding_index
    
    def tokenize(self, tokenizer, max_sequence_length):
        tokenizer.fit_on_texts(self.train_comments + self.test_comments)
        train_sequences = tokenizer.texts_to_sequences(self.train_comments)
        test_sequences = tokenizer.texts_to_sequences(self.test_comments)
        val_sequences = tokenizer.texts_to_sequences(self.val_comments)

        word_index = tokenizer.word_index

        train_x = pad_sequences(train_sequences, maxlen = max_sequence_length)
        test_x = pad_sequences(test_sequences, maxlen = max_sequence_length)
        val_x = pad_sequences(val_sequences, maxlen = max_sequence_length)

        print('Shape of train_x tensor:', train_x.shape)
        print('Shape of test_data tensor:', test_x.shape)
        print('Shape of val_data tensor:', val_x.shape)
        print('Found %s unique tokens' % len(word_index))

        return train_x, test_x, val_x, word_index
    
    def create_embedding_matrix(self, word_index, embedding_dim, embedding_index, max_features):
        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.zeros((nb_words, embedding_dim))
        null_count = 0
        null_words = []
        for word, i in word_index.items():
            if i >= max_features:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                null_count += 1
                null_words.append(word)
        print('Null word embeddings: %d' % null_count)
        return embedding_matrix
            

        