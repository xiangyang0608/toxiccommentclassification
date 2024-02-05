
import keras
from keras.layers import Dense, Lambda
from keras.layers import Input, Embedding, SpatialDropout1D, Dense
from keras.models import Model
from keras.layers import Dense, Lambda, Dropout, concatenate
from keras.layers import Input, Embedding, SpatialDropout1D, Dense, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.layers import Layer
from keras import initializers
from keras import backend as K
import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, SpatialDropout1D, Reshape, GlobalAveragePooling1D, Flatten, Bidirectional, add, Conv1D, GlobalMaxPooling1D
from keras.layers import Concatenate, GRU
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten

import numpy as np
np.random.seed(42)



class AttentionWeightedAverage(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        # self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[2], 1), name='{}_W'.format(self.name), initializer=self.init)
        # self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

def get_baseline(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(nb_words, 
                                embedding_dim, 
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.25)(embedding_layer)
    
    
    x = Dense(128, activation='relu')(embedding_layer)
    
    last = Lambda(lambda t: t[:, -1], name='last')(x)
    
    output_layer = Dense(out_size, activation='sigmoid')(last)
    model = Model(inputs=input_layer, outputs=output_layer)
    adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model

def get_LSTM(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    inp = Input(shape=(max_sequence_length,))

    x = Embedding(nb_words, 
                  embedding_dim, 
                  weights=[embedding_matrix],
                  input_length=max_sequence_length,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.35)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.35)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.35)(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.5)(x)
    
    x = Dense(50, activation="relu")(x)
    out = Dense(out_size, activation='sigmoid')(x)
    model = Model(inp, out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def get_LSTM_attn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    inp = Input(shape=(max_sequence_length,))

    x = Embedding(nb_words, 
                  embedding_dim, 
                  weights=[embedding_matrix],
                  input_length=max_sequence_length,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.35)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.35)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.35)(x)
    
    last = Lambda(lambda t:t[:, -1], name='last')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    attn = AttentionWeightedAverage()(x)
    x = concatenate([last, avg_pool, max_pool, attn])
    x = Dropout(0.5)(x)
    
    x = Dense(50, activation="relu")(x)
    out = Dense(out_size, activation='sigmoid')(x)
    model = Model(inp, out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def get_GRU(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    recurrent_units = 64
    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.35)(embedding_layer)

    rnn_1 = Bidirectional(GRU(recurrent_units, return_sequences=True))(embedding_layer)
    rnn_2 = Bidirectional(GRU(recurrent_units, return_sequences=True))(rnn_1)
    x = Concatenate(axis=2)([rnn_1, rnn_2])

    last = Lambda(lambda t: t[:, -1], name='last')(x)
    maxpool = GlobalMaxPooling1D()(x)
    average = GlobalAveragePooling1D()(x)

    all_views = Concatenate(axis=1)([last, maxpool, average])
    x = Dropout(0.5)(all_views)
    x = Dense(144, activation="relu")(x)
    output_layer = Dense(out_size, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model

def get_gru_rnn_attention(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    recurrent_units = 64
    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.35)(embedding_layer)

    rnn_1 = Bidirectional(GRU(recurrent_units, return_sequences=True))(embedding_layer)
    rnn_2 = Bidirectional(GRU(recurrent_units, return_sequences=True))(rnn_1)
    x = Concatenate(axis=2)([rnn_1, rnn_2])

    last = Lambda(lambda t: t[:, -1], name='last')(x)
    maxpool = GlobalMaxPooling1D()(x)
    attn = AttentionWeightedAverage()(x)
    average = GlobalAveragePooling1D()(x)

    all_views = Concatenate(axis=1)([last, maxpool, average, attn])
    x = Dropout(0.5)(all_views)
    x = Dense(144, activation="relu")(x)
    output_layer = Dense(out_size, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model

def get_textCNN(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):    
    inp = Input(shape=(max_sequence_length, ))
    x = Embedding(nb_words,
                  embedding_dim, 
                  weights=[embedding_matrix],
                  input_length=max_sequence_length,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((max_sequence_length, embedding_dim, 1))(x)
    
    conv_0 = Conv2D(32, kernel_size=(1, embedding_dim), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_1 = Conv2D(32, kernel_size=(2, embedding_dim), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_2 = Conv2D(32, kernel_size=(3, embedding_dim), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    conv_3 = Conv2D(32, kernel_size=(5, embedding_dim), kernel_initializer='normal',
                                                                                    activation='elu')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(max_sequence_length - 1 + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_sequence_length - 2 + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_sequence_length - 3 + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(max_sequence_length - 5 + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(out_size, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model