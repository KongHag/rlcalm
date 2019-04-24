#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder 

"""
COPYRIGHT A2IM-ROBOTADVISORS & INSTITUT LOUIS BACHELIER
DEVELOPPER : JDEM-ILB
DATE : 07-04-2019
DESCRIPTION :
THIS MODULE BUILD A MODEL FOR KAGGLE CHALLENGE USING A LSTM WITH ATTENTION
THIS SCRIPT IS INSPIRED FROM SEVERAL KERNELS, REFERENCE BELOW :
    - https://www.kaggle.com/thousandvoices/simple-lstm
    - https://www.kaggle.com/rahulvks/lstm-attention-keras
"""
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


print("Loading data...")
df_train = pd.read_csv("/home/ubuntu/Documents/Kaggle/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
print("Train shape:", df_train.shape)
df_test = pd.read_csv("/home/ubuntu/Documents/Kaggle/jigsaw-unintended-bias-in-toxicity-classification/test.csv", encoding="latin-1")
print("Test shape:", df_test.shape)

###############################################################################
######################### Shuffling the dataframe !############################
###############################################################################

df_train = df_train.sample(frac=1).reset_index(drop=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)

###############################################################################
##################### Imposing the training data proprotions !#################
###############################################################################

training_samples = 100000
training_samples_per_class = training_samples // 2
validation_samples = 100000
testing_samples = 100000

df_train = df_train.rename(columns=({"comment_text":"Reviews"}))
df_train = df_train.rename(columns=({"target":"Label"}))
df_test = df_test.rename(columns=({"comment_text":"Reviews"}))

X_train = []
Y_train = []

counting_toxic_comments = 0
counting_non_toxic_comments = 0
exploring_the_data = 0

while counting_toxic_comments < training_samples_per_class :
    review, label = df_train.iloc[exploring_the_data]['Reviews'], df_train.iloc[exploring_the_data]['Label']
    if label < 0.5 and counting_non_toxic_comments < training_samples_per_class :
        X_train += [review]
        Y_train += [label]
        counting_non_toxic_comments += 1
        exploring_the_data += 1
    elif label > 0.5 :
        X_train += [review]
        Y_train += [label]
        counting_toxic_comments += 1
        exploring_the_data += 1
    else :
        exploring_the_data += 1
        
X_test = []

for i in range(len(df_test)):
    review = df_test.iloc[i]['Reviews']
    X_test += [review]
    
###############################################################################
################################## Embedding !#################################
###############################################################################

EMBEDDING_FILE =  '/home/ubuntu/Documents/Kaggle/GloVe/glove.twitter.27B.200d.txt'

def load_embeddings(filename):
    embeddings = {}
    with open(filename) as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

print("Loading embedding")
embeddings = load_embeddings(EMBEDDING_FILE)
print("Embedding is now complete")

###############################################################################
############################ Negation Handling ################################
###############################################################################

#df_train.Reviews = df_train.Reviews.str.replace("n't", 'not')
#df_test.Reviews = df_test.Reviews.str.replace("n't", 'not')

for comment_train, comment_test in  zip(X_train, X_test):
    comment_train.replace("n't", 'not')
    comment_test.replace("n't", 'not')

#df_train['Reviews'] = df_train['Reviews'].astype(str)
#df_test['Reviews'] = df_test['Reviews'].astype(str)
    
###############################################################################
############################ Special characters ###############################
###############################################################################
    
def clean_special_chars(comment):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    for p in punct:
        comment = comment.replace(p, ' ')
    return text

for comment_train, comment_test in  zip(X_train, X_test):
    clean_special_chars(comment_train)
    clean_special_chars(comment_test)

###############################################################################
############################ Numbers Handling #################################
###############################################################################
    
#df_train.Reviews = df_train.Reviews.apply(lambda x: re.sub(r'[0-9]+', '0', x))
#df_test.Reviews = df_test.Reviews.apply(lambda x: re.sub(r'[0-9]+', '0', x))

for comment_train, comment_test in  zip(X_train, X_test):
    re.sub(r'[0-9]+', '0', comment_train)
    re.sub(r'[0-9]+', '0', comment_test)
    
for i in range(len(Y_train)):
    if (Y_train[i] >= 0.5):
        Y_train[i] = 1
        
X = np.r_[X_train, X_test]

###############################################################################
############################## Tokenization ###################################
###############################################################################
s
tokenizer = Tokenizer(lower=True, filters='\n\t')
tokenizer.fit_on_texts(X)
X_train = tokenizer.texts_to_sequences(X_train)
X_test  = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1  # +1 is for zero padding.
print('vocabulary size: {}'.format(vocab_size))

maxlen = len(max((s for s in np.r_[X_train, X_test]), key=len))
X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')

def filter_embeddings(embeddings, word_index, vocab_size, dim=300):
    embedding_matrix = np.zeros([vocab_size, dim])
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        vector = embeddings.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix

embedding_size = 200 ##Twitter Glove format
embedding_matrix = filter_embeddings(embeddings, tokenizer.word_index,
                                     vocab_size, embedding_size)
print('OOV: {}'.format(len(set(tokenizer.word_index) - set(embeddings))))

###############################################################################
################################# Model #######################################
###############################################################################

class Attention(Layer):
    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(Attention())
    """
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    

def build_model(maxlen, vocab_size, embedding_size, embedding_matrix):
    input_words = Input((maxlen, ))
    x_words = Embedding(vocab_size,
                        embedding_size,
                        weights=[embedding_matrix],
                        trainable=False)(input_words)
    
    x_words = SpatialDropout1D(0.3)(x_words)
    x_words = Bidirectional(LSTM(128, return_sequences=True))(x_words)
    x_words = Bidirectional(LSTM(128, return_sequences=True))(x_words)
    
    att = Attention(maxlen)(x_words)
    avg_pool1 = GlobalAveragePooling1D()(x_words)
    max_pool1 = GlobalMaxPooling1D()(x_words)
    
    x = concatenate([att,avg_pool1, max_pool1])

    pred = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_words, outputs=pred)
    return model


model = build_model(maxlen, vocab_size, embedding_size, embedding_matrix)
model.compile(optimizer = Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

###############################################################################
############################ Training the model ###############################
###############################################################################

number_of_epochs = 50
size_of_batch = 2048

history = model.fit(X_train, Y_train,
                    epochs = number_of_epochs, verbose=1,
                    batch_size = size_of_batch, shuffle=True)

###############################################################################
########################### Making a prediction ###############################
###############################################################################

print("Predicting the labels...")
y_pred = model.predict(X_test, batch_size=1024)
y_pred = y_pred.argmax(axis=1).astype(float)
print("Labels predicted!")

df_test['prediction'] = y_pred
df_test[['id', 'prediction']].to_csv('submission.csv', index=False)


