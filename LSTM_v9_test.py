#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import tqdm
import random
import logging
import datetime
import numpy as np
import pandas as pd

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam 
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras import initializers, regularizers, constraints
from keras.layers import Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, LSTM, SpatialDropout1D, Dropout, Bidirectional, concatenate, Embedding

"""
COPYRIGHT A2IM-ROBOTADVISORS & INSTITUT LOUIS BACHELIER
DEVELOPPER : JDEM-ILB
DATE : 07-04-2019
DESCRIPTION :
THIS MODULE BUILDS A MODEL WITH ITS OWN EMBEDDING USING DOC2VEC AND A BILSTM 
THIS SCRIPT IS INSPIRED FROM SEVERAL KERNELS, REFERENCE BELOW :
    - https://www.kaggle.com/thousandvoices/simple-lstm
    - https://www.kaggle.com/rahulvks/lstm-attention-keras
"""
this_folder_path = "/media/ubuntu/Data/Kaggle/"
data_path = this_folder_path + "jigsaw-unintended-bias-in-toxicity-classification/"

print("Loading data...")
df_train = pd.read_csv(data_path + "train.csv")
print("Train shape:", df_train.shape)
df_test = pd.read_csv(data_path + "test.csv")
print("Test shape:", df_test.shape)

###############################################################################
######################### Renaming some columns !##############################
###############################################################################

df_train = df_train.rename(columns=({"comment_text":"Reviews"}))
df_train = df_train.rename(columns=({"target":"Label"}))
df_test = df_test.rename(columns=({"comment_text":"Reviews"}))
df_test = df_test.rename(columns=({"target":"Label"}))

###############################################################################
########################### Cleaning the data #################################
###############################################################################

def clean_emojis(text):
    """ Cleans comments_text from emojis """
    text = re.sub(r'(<3|:\*)', ' Love ', text)
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' Wink ', text)
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' Sad ', text)
    return text

def clean_negation(text):
    """ Cleans negations in english comments"""
    text = text.replace("n't", 'not')
    return text

def clean_special_chars(text):
    """ Cleans special characters in comments
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution"""
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    for p in punct:
        text = text.replace(p, ' ')
    return text

def clean_numbers(text):
    """ Cleans numbers in comments"""
    text = re.sub(r'[0-9]+', '0', text)
    return text

CLEANING_EMOJIS = True
CLEANING_NEGATION = True
CLEANING_SPECIAL_CHARS = True
CLEANING_NUMBERS = True

def clean(text):
    if CLEANING_EMOJIS:
        text = clean_emojis(text)
    if CLEANING_NEGATION:
        text = clean_negation(text)
    if CLEANING_SPECIAL_CHARS:
        text = clean_special_chars(text)
    if CLEANING_NUMBERS:
        text = clean_numbers(text)
    return text

###############################################################################
#################### Reading and labelling the data ###########################
###############################################################################
    
def convert_df_to_bool_toxicity(df):
    bool_df = df.copy()
    bool_df['Label'] = np.where(df['Label'] >= 0.5, True, False)
    return bool_df
    
def read_toxicity_dataset(initial_dataset):
    print("Reading and labelling the data...")
    dataset = convert_df_to_bool_toxicity(initial_dataset)
    X = dataset.Reviews
    Y = np.array([dataset.Label.tolist()])                    
    X = toxic_label_sentences(X, np.transpose(Y))
    return X

def toxic_label_sentences(x, y):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be i_True or i_False where i is
    a dummy variable and where True means toxic. This is the first part of our embedding".
    """
    labeled = []
    for i, v in enumerate(x):
        label = str(i) + '_' + str(y[i])
        v = clean(text=v)
        labeled.append(TaggedDocument(v.split(), label)) # Comment + Tag
    return labeled

###############################################################################
######################## The same for minority ################################
###############################################################################
    
all_identity_columns = ['asian', 'atheist', 'bisexual',
       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
       'jewish', 'latino', 'male', 'muslim', 'other_disability',
       'other_gender', 'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white']

def convert_df_to_bool_minorities(df):
    bool_df = df.copy()
    for minority in all_identity_columns:
        bool_df[minority] = np.where(df[minority] == 1, True, False)
    return bool_df
        
def from_Y_to_label(y, i):
    """
    Output example : '35_asian_Christian' means that review indexed with 35 mentions
    both asian and christian minorities.
    """
    label = str(i) + '_'
    for j in range(len(all_identity_columns)):
        if y[j][i]:
            label += all_identity_columns[j] + '_'
    return label

def read_minorities_dataset(initial_dataset):
    print("Reading and labelling the data...")
    dataset = convert_df_to_bool_minorities(initial_dataset)
    X = dataset.Reviews
    Y = np.array([dataset.asian.tolist(), dataset.atheist.tolist(), 
                        dataset.bisexual.tolist(), dataset.black.tolist(), dataset.buddhist.tolist(), dataset.christian.tolist(), 
                        dataset.female.tolist(), dataset.heterosexual.tolist(), dataset.hindu.tolist(), dataset.homosexual_gay_or_lesbian.tolist(), dataset.intellectual_or_learning_disability.tolist(),
                        dataset.jewish.tolist(), dataset.latino.tolist(), dataset.male.tolist(), dataset.muslim.tolist(),dataset.other_disability.tolist(), 
                        dataset.other_gender.tolist(),dataset.other_race_or_ethnicity.tolist(), dataset.other_religion.tolist(), dataset.other_religion.tolist(), dataset.other_sexual_orientation.tolist(), dataset.physical_disability.tolist(), 
                        dataset.psychiatric_or_mental_illness.tolist(), dataset.transgender.tolist(), dataset.white.tolist()])                     
    X = minority_label_sentences(X, Y)
    return X

def minority_label_sentences(x, y):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be i_ + 'minorities' where 
    'minorities' .
    """
    labeled = []
    for i, v in enumerate(x):
        label = from_Y_to_label(y, i)
        v = clean(text=v)
        labeled.append(TaggedDocument(v.split(), label)) # Comment + Tag
    return labeled

###############################################################################
#################### Building doc2vec model ###################################
###############################################################################
    
WORKERS = 8
ITERATIONS = 20
VOCAB_SIZE = 1669827
LEARNING_RATE = 0.4
BATCH_SIZE = 128
VERBOSE = 1

model_path = this_folder_path + "Doc2Vec/"
toxic_model_name = model_path + "toxic_doc2vec_model"
minority_model_name = model_path + "minority_doc2vec_model"
 
def train_doc2vec(corpus, model_file):
    print("Building the Doc2Vec model")
    d2v = Doc2Vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=10,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=WORKERS,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=0)  # dm defines the training algorithm. If dm=1 means 'distributed memory' (PV-DM)
                                 # and dm =0 means 'distributed bag of words' (PV-DBOW)
    d2v.build_vocab(corpus)
    print("Training Doc2Vec model...")
    for epoch in range(ITERATIONS):
        print("Starting epoch number ", epoch + 1 , ' out of ', ITERATIONS, ' epochs.')
        t0 = datetime.datetime.now().timestamp()
        logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002
        # fix the learning rate, no decay
        d2v.min_alpha = d2v.alpha
        t1 = datetime.datetime.now().timestamp()
        print("The epoch lasted ", t1-t0, " seconds.")
    print("Saving the Doc2Vec model...")
    d2v.save(model_file)
    return d2v

###############################################################################
########################## Training doc2vec models ############################
###############################################################################
    
TRAINING = False
    
if __name__ == "__main__" and TRAINING :
    print("---> Reading and labelling data... \n")
    all_minority_data = read_minorities_dataset(df_train)
    all_toxic_data = read_toxicity_dataset(df_train)
    print("---> Training the doc2vec model for minorities...\n")
    d2v_model_minority = train_doc2vec(all_minority_data, minority_model_name)    
    print("---> Training the doc2vec model for toxicity...\n")
    d2v_model_toxicity = train_doc2vec(all_toxic_data, toxic_model_name)
    print("---> Well done young padawan, your embedding is ready...")

###############################################################################
########## Loading models and getting labelled vectors ########################
###############################################################################

d2v_model_toxicity = Doc2Vec.load(toxic_model_name)
d2v_model_minority = Doc2Vec.load(minority_model_name)

###############################################################################
################## Preparing the Embedding Model ##############################
###############################################################################

x_train = df_train['Reviews'].values
x_test = df_test['Reviews'].values
y_train = df_train['Label'].values
y_train = np.where(y_train >= 0.5, 1, 0)

for i in tqdm.trange(len(x_train)):
    x_train[i] = clean(x_train[i])

for i in tqdm.trange(len(x_test)):
    x_test[i] = clean(x_test[i])
    
x = np.r_[x_train, x_test]

tokenizer = Tokenizer(lower=True, filters='\n\t')
tokenizer.fit_on_texts(x)
x_train = tokenizer.texts_to_sequences(x_train)
x_test  = tokenizer.texts_to_sequences(x_test)
vocab_size = len(tokenizer.word_index) + 1  # +1 is for zero padding.
print('vocabulary size: {}'.format(vocab_size))

maxlen = len(max((s for s in np.r_[x_train, x_test]), key=len))
x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

def filter_embeddings(d2v_model_toxicity, d2v_model_minority, word_index, vocab_size, dim=600):
    embedding_matrix = np.zeros([vocab_size, dim])
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        vector = np.concatenate((d2v_model_toxicity.infer_vector(word), d2v_model_minority.infer_vector(word)))
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix

embedding_size = 600 ##Concatenated sum
embedding_matrix = filter_embeddings(d2v_model_toxicity, d2v_model_minority, tokenizer.word_index,
                                     vocab_size, embedding_size)    

###############################################################################
############################ Attention Class ##################################
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
    
###############################################################################
########################## Building the Model #################################
###############################################################################
    
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
    
    x = Dropout(0.5)(x)
    x = Dense(144, activation='relu')(x)
    pred = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_words, outputs=pred)
    return model   

###############################################################################
########################## Training the Model #################################
###############################################################################

best_model_path = this_folder_path + 'Models/'
weights_folder = this_folder_path + 'Weights/'
number_of_epochs = 30
batch_size = 512
learning_rate = 1e-3
output_file = 'LSTM_v9_Doc2Vec.h5'

def training_model(number_of_epochs, batch_size, learning_rate, output_file, vocab_size,
                   embedding_matrix, embedding_size, maxlen):
    model = build_model(maxlen, vocab_size, embedding_size, embedding_matrix)
    Adam_optimizer = Adam(lr = learning_rate, decay = 1e-6, clipvalue =5)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only = True, save_weights_only = True)
    model.summary()
    model.compile(optimizer = Adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train,
                        epochs = number_of_epochs, verbose=1,
                        batch_size = batch_size, shuffle=True,
                        callbacks = [early_stopping, model_checkpoint])
    model.save_weights(weights_folder + output_file)

training_model(number_of_epochs, batch_size, learning_rate, output_file, vocab_size,
                   embedding_matrix, embedding_size, maxlen)
