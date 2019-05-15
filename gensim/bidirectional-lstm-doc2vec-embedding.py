#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import tqdm
import random
import logging
import datetime
import numpy as np
import pandas as pd

from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import initializers, regularizers, constraints
from keras.layers import Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, LSTM, SpatialDropout1D, Bidirectional, concatenate, Embedding
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, merge, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import InputSpec, Layer
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras.engine import InputSpec, Layer

"""
COPYRIGHT A2IM-ROBOTADVISORS & INSTITUT LOUIS BACHELIER
DEVELOPPER : JDEM-DLA
DATE : 12-05-2019
DESCRIPTION :
THIS MODULE BUILDS A MODEL WITH ITS OWN EMBEDDING USING DOC2VEC AND A BILSTM 
THIS SCRIPT IS INSPIRED FROM SEVERAL KERNELS, REFERENCE BELOW :
    - https://www.kaggle.com/thousandvoices/simple-lstm
    - https://www.kaggle.com/rahulvks/lstm-attention-keras
"""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

this_folder_path = os.path.join(os.path.dirname(__file__))


KAGGLE_KERNEL = False

if KAGGLE_KERNEL :
    
    print(os.listdir("../input"))
    print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))
    print(os.listdir("../input/weights"))
    print(os.listdir("../input/doc2vec"))
    print(os.listdir("../input/submission-1305"))
    
    data_path = '/../input/jigsaw-unintended-bias-in-toxicity-classification/'
    toxic_model_name = '/../input/doc2vec/toxic_doc2vec_model'
    minority_model_name = '/../input/doc2vec/minority_doc2vec_model'
    CLASSIFIER_MODEL_FILE = this_folder_path + '/../input/weights/' + 'LSTM_doc2Vec_weights.h5'
    SUBMISSION_FILE_NAME = '../input/submission/submission-13052019-1007.csv'
else:
    data_path = this_folder_path + "jigsaw-unintended-bias-in-toxicity-classification/" 
    toxic_model_name = this_folder_path + "Doc2Vec/" + "toxic_doc2vec_model" 
    minority_model_name = this_folder_path + "Doc2Vec/" + "minority_doc2vec_model" 
    CLASSIFIER_MODEL_FILE = this_folder_path + 'Weights/' + 'LSTM_doc2Vec_weights.h5'
    SUBMISSION_FILE_NAME = this_folder_path + 'submission-13052019-1100.csv'

EPOCHS = 70 # DEBUG MODE = 1 / TRAINING_MODE = 70
CLASSIFIER_BATCH = 512

DEBUG = False

DOC2VEC_TRAINING = False
CLASSIFICATION_TRAINING = True
SUBMISSION_PREPARATION = True
SUBMISSION = True

CLEANING_EMOJIS = True
CLEANING_URLS = True
CLEANING_ANTISLASHES = True
CLEANING_NEGATION = True
CLEANING_SPECIAL_CHARS = True
CLEANING_NUMBERS = True

DELETING_DUPLICATES = False

###############################################################################
############################### Load Data !####################################
###############################################################################

if not DELETING_DUPLICATES:
    print("Loading data...",flush=True)
    df_train = pd.read_csv(data_path + "train_v2.csv")
    print("Train shape:", df_train.shape,flush=True)
    df_test = pd.read_csv(data_path + "test.csv")
    print("Test shape:", df_test.shape,flush=True)

if DEBUG:
    df_train = df_train[:5000]
    df_test = df_train[:1000]

###############################################################################
########################## Deleting duplicates ################################
###############################################################################

def get_duplicate_indexes(df):
    """ Get the indexes of the review duplicates"""
    df_reviews = pd.DataFrame(df, columns = ["comment_text"])
    duplicated_reviews_RowsDF = df_reviews[df_reviews.duplicated(keep = False)]
    return duplicated_reviews_RowsDF.index.tolist()

def get_weighted_average(labels, weights):
    """Derives a weighted average of labels. 
       In this special case labels are the label while
       weights are the number of toxicity annotators """
    W = sum(weights)
    res = sum([label*weight/W for (label, weight) in zip(labels, weights)])
    return res

def how_many_before_me(id_, l):
    """Nothing too fancy, just get the elements deleted 
       before your index to get the proper lag"""
    res = 0 
    if l :
        for el in l:
            if el < id_ :
                res += 1
    return res

def delete_duplicates(df, duplicates):
    """Deletes all reviews duplicates and implements the new toxicity label
       as a weighted average. Labels with more annotators are considered as
       more relevant."""
    df_res = df.copy()
    duplicates = sorted(duplicates) ##Getting sure everything is in order
    treated_rows = []
    deleted_rows = []
    while duplicates :
        print(len(duplicates), " comments still need to be treated!", flush = True)
        id_0 = duplicates[0] ## Indice
        lag = how_many_before_me(id_0, deleted_rows)
        review_0 = df["comment_text"].iloc[id_0]
        label_0 = df["target"].iloc[id_0]
        weight_0 = df["toxicity_annotator_count"].iloc[id_0]
        labels = [label_0]
        weights = [weight_0]
        treated_rows = [id_0]
        for id_ in duplicates[1:]:
            if df["comment_text"].iloc[id_] == review_0:
                labels += [df["target"].iloc[id_]]
                weights += [df["toxicity_annotator_count"].iloc[id_]]
                treated_rows += [id_]
        weighted_average = get_weighted_average(labels, weights)
        df_res["target"].iloc[id_0 - lag] = weighted_average
        df_res = df_res.drop(treated_rows[1:], axis =0)
        for deleted_row in treated_rows[1:] : 
            deleted_rows += [deleted_row]         
        for idx in treated_rows :
            duplicates.remove(idx)
    return df_res
    
if DELETING_DUPLICATES:
    df_train = pd.read_csv(data_path + "train.csv")
    df_test = pd.read_csv(data_path + "test.csv")
    index_duplicates = get_duplicate_indexes(df_train)
    df_train = delete_duplicates(df_train, index_duplicates)
    
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

def clean_URLs(text):
    """ Cleans URL hyperlinks in english comments"""
    text = re.sub(r"http\S+", "", text)
    return text

def clean_antislashes(text):
    """Deletes the \n, the \t and the \r in an english comment"""
    return ' '.join(''.join(text).split()).strip()

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

def clean(text):
    if CLEANING_EMOJIS:
        text = clean_emojis(text)
    if CLEANING_URLS:
        text = clean_URLs(text)
    if CLEANING_ANTISLASHES:
        text = clean_antislashes(text)
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

if DOC2VEC_TRAINING :
    WORKERS = 8
    ITERATIONS = 20
    VOCAB_SIZE = 1669827
    LEARNING_RATE = 0.4
    BATCH_SIZE = 128
    VERBOSE = 2
    WINDOW = 100

    model_path = this_folder_path + "Doc2Vec/"
    toxic_model_name = model_path + "toxic_doc2vec_model"
    minority_model_name = model_path + "minority_doc2vec_model"
 
def train_doc2vec(corpus, model_file):
    print("Building the Doc2Vec model",flush=True)
    d2v = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=WINDOW,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=WORKERS,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=0)  # dm defines the training algorithm. If dm=1 means 'distributed memory' (PV-DM)
                                 # and dm =0 means 'distributed bag of words' (PV-DBOW)
    d2v.build_vocab(corpus)
    print("Training Doc2Vec model...",flush=True)
    for epoch in range(ITERATIONS):
        print("Starting epoch number ", epoch + 1 , ' out of ', ITERATIONS, ' epochs.',flush=True)
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
        print("The epoch lasted ", t1-t0, " seconds.",flush=True)
    print("Saving the Doc2Vec model...",flush=True)
    d2v.save(model_file)
    return d2v

###############################################################################
########################## Training doc2vec models ############################
###############################################################################
    
    
if __name__ == "__main__" and DOC2VEC_TRAINING :
    print("---> Reading and labelling data... \n",flush=True)
    all_minority_data = read_minorities_dataset(df_train)
    all_toxic_data = read_toxicity_dataset(df_train)
    print("---> Training the doc2vec model for minorities...\n",flush=True)
    d2v_model_minority = train_doc2vec(all_minority_data, minority_model_name)    
    print("---> Training the doc2vec model for toxicity...\n",flush=True)
    d2v_model_toxicity = train_doc2vec(all_toxic_data, toxic_model_name)
    print("---> Well done young padawan, your embedding is ready...",flush=True)

###############################################################################
########## Loading models and getting labelled vectors ########################
###############################################################################

if CLASSIFICATION_TRAINING or SUBMISSION_PREPARATION:
    logging.info("Loading models and getting labelled vectors...")

    d2v_model_toxicity = doc2vec.Doc2Vec.load(toxic_model_name)
    d2v_model_minority = doc2vec.Doc2Vec.load(minority_model_name)

###############################################################################
################## Preparing the EMbedding Model ##############################
###############################################################################


    logging.info("Preparing data ...")

    x_train = df_train['Reviews'].values
    x_test = df_test['Reviews'].values
    y_train = df_train['Label'].values
    y_train = np.where(y_train >= 0.5, 1, 0)

    logging.info("Cleaning data ...")

    for i in tqdm.trange(len(x_train)):
        x_train[i] = clean(x_train[i])

    for i in tqdm.trange(len(x_test)):
        x_test[i] = clean(x_test[i])
        
    x = np.r_[x_train, x_test]

    logging.info("Tokenizing  ...")

    tokenizer = Tokenizer(lower=True, filters='\n\t')
    tokenizer.fit_on_texts(x)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test  = tokenizer.texts_to_sequences(x_test)
    vocab_size = len(tokenizer.word_index) + 1  # +1 is for zero padding.
    logging.info('vocabulary size: {}'.format(vocab_size))

    maxlen = len(max((s for s in np.r_[x_train, x_test]), key=len))
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

def filter_embeddings(d2v_model_toxicity, d2v_model_minority, word_index, vocab_size, dim=600):
    embedding_matrix = np.zeros([vocab_size, dim])
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        vector = np.concatenate((d2v_model_toxicity.infer_vector(word.split()), d2v_model_minority.infer_vector(word.split())))
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix

if CLASSIFICATION_TRAINING or SUBMISSION_PREPARATION :
    logging.info("Building Embedding Matrix  ...")

    embedding_size = 600 ##Concatenated sum
    embedding_matrix = filter_embeddings(d2v_model_toxicity, d2v_model_minority, tokenizer.word_index,
                                     vocab_size, embedding_size)    

###############################################################################
############################ Attention Class ##################################
###############################################################################

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
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
    pred = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_words, outputs=pred)
    return model   

def get_av_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    filter_nums = 300  # 500->375, 400->373, 300->

    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences = SpatialDropout1D(0.25)(embedded_sequences)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    attn_0 = AttentionWeightedAverage()(conv_0)
    avg_0 = GlobalAveragePooling1D()(conv_0)
    maxpool_0 = GlobalMaxPooling1D()(conv_0)

    maxpool_1 = GlobalMaxPooling1D()(conv_1)
    attn_1 = AttentionWeightedAverage()(conv_1)
    avg_1 = GlobalAveragePooling1D()(conv_1)

    maxpool_2 = GlobalMaxPooling1D()(conv_2)
    attn_2 = AttentionWeightedAverage()(conv_2)
    avg_2 = GlobalAveragePooling1D()(conv_2)

    maxpool_3 = GlobalMaxPooling1D()(conv_3)
    attn_3 = AttentionWeightedAverage()(conv_3)
    avg_3 = GlobalAveragePooling1D()(conv_3)

    v0_col = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3], )
    v1_col = Concatenate(axis=1)([attn_0, attn_1, attn_2, attn_3])
    v2_col = Concatenate(axis=1)([avg_1, avg_2, avg_0, avg_3])
    merged_tensor = Concatenate(axis=1)([v0_col, v1_col, v2_col])
    output = Dropout(0.7)(merged_tensor)
    output = Dense(units=144)(output)
    output = Activation('relu')(output)
    output = Dense(units=out_size, activation='sigmoid')(output)

    model = Model(inputs=comment_input, outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model

###############################################################################
########################## Training the Model #################################
###############################################################################


if CLASSIFICATION_TRAINING:
    logging.info("Setting up model ...")
    # MODEL - BIDIRECTIONAL LSTM
    model = build_model(maxlen, vocab_size, embedding_size, embedding_matrix)
    model.compile(optimizer = Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    model_checkpoint = ModelCheckpoint(CLASSIFIER_MODEL_FILE, save_best_only=True, save_weights_only=True)

    logging.info("Training model ...")

    history = model.fit(x_train, y_train,
                         validation_split=0.3,
                         epochs=EPOCHS, 
                         batch_size=CLASSIFIER_BATCH, 
                         shuffle=True,
                         verbose=2,
                         callbacks=[early_stopping, model_checkpoint])
       
###############################################################################
########################## Model Submission #################################
###############################################################################

if SUBMISSION_PREPARATION:
    model = build_model(maxlen, vocab_size, embedding_size, embedding_matrix)
    logging.info("Loading model ...")
    model.load_weights(CLASSIFIER_MODEL_FILE)
    
    print(df_test.head(10),flush=True)
    print(df_test['id'].values[0],flush=True)
    print(df_test['id'].values[1],flush=True)

    logging.info("Predicting labels ...")
    y_pred = model.predict(x_test, batch_size=1024)
    
    logging.info("Labels predicted!")
    df_test['prediction'] = y_pred
    df_test[['id', 'prediction']].to_csv(SUBMISSION_FILE_NAME, index=False)
    
    logging.info("Submission finished!")

if SUBMISSION:
    df_test = pd.read_csv(SUBMISSION_FILE_NAME)
    df_test.to_csv('submission.csv', index=False)
    logging.info("Submission finished!")
