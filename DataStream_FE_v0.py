#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:02:57 2019

@author: JDEM-ILB

DESCRIPTION : THIS SCRIPT AIMS AT BUILDING A SIMPLE PIPELINE FOR THE KAGGLE
CHALLENGE 

"""

import os
import re
import logging
import pickle

import numpy as np
import pandas as pd

from profanity import profanity ##pip install profanity

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

###############################################################################
########################### Loading Data ######################################
###############################################################################

this_folder = "/home/ubuntu/Documents/Kaggle/"
data_folder = "jigsaw-unintended-bias-in-toxicity-classification/"

print("Loading data...")
df_train = pd.read_csv(this_folder + data_folder + "train.csv")
print("Train shape:", df_train.shape)
df_test = pd.read_csv(this_folder + data_folder + "test.csv")
print("Test shape:", df_test.shape) 

df_train = df_train.rename(columns=({"comment_text":"Reviews"}))
df_train = df_train.rename(columns=({"target":"Label"}))
df_test = df_test.rename(columns=({"comment_text":"Reviews"}))

df_train = df_train.sample(frac=1).reset_index(drop=True) #Shuffling data
df_test = df_test.sample(frac=1).reset_index(drop=True)

###############################################################################
##################### Imposing the training data proprotions ##################
###############################################################################

training_samples = 10000
toxic_proportion = 0.5 #50% of toxic comments
training_samples_per_class = int(training_samples*toxic_proportion)

X_train = []
X_test = []
y_train = []

counting_toxic_comments = 0
counting_non_toxic_comments = 0
exploring_the_data = 0

while counting_toxic_comments < training_samples_per_class :
    review, label = df_train.iloc[exploring_the_data]['Reviews'], df_train.iloc[exploring_the_data]['Label']
    if label < 0.5 and counting_non_toxic_comments < training_samples_per_class :
        X_train += [review]
        y_train += [label]
        counting_non_toxic_comments += 1
        exploring_the_data += 1
    elif label > 0.5 :
        X_train += [review]
        y_train += [label]
        counting_toxic_comments += 1
        exploring_the_data += 1
    else :
        exploring_the_data += 1
        
for i in range(len(y_train)): ##Making it 1 or 0
    if (y_train[i] >= 0.5):
        y_train[i] = 1
    else:
        y_train[i] = 0
        
y_train = np.array(y_train)  ## [0 1] means that the comment is toxic
OneHotEncoder = OneHotEncoder(sparse = False)
y_train = y_train.reshape(len(y_train), 1)
y_train_encoded = OneHotEncoder.fit_transform(y_train)
        
###############################################################################
############################ Data listing #####################################
###############################################################################  
        
for i in range(len(df_test)):
    review = df_test.iloc[i]['Reviews']
    X_test += [review]

###############################################################################
########################## Features Engineering ###############################
###############################################################################

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

agressive_punct = "!*."

minority_tags = ['asian', 'atheist', 'bisexual',
       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
       'jewish', 'latino', 'male', 'muslim', 'other_disability',
       'other_gender', 'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white']

n_train = len(X_train)
n_test = len(X_test)

F_train = np.zeros((n_train, 30))
F_test = np.zeros((n_test, 30))

def how_many_puncts(X):
    res = np.zeros((len(X), 1))
    for i in range(len(X)):
        comment = X[i]
        for p in punct:
            res[i] += comment.count(p)
    return res

num_puncts_train = how_many_puncts(X_train)
num_puncts_test = how_many_puncts(X_test)

F_train[:,0] = np.squeeze(num_puncts_train) #Feature 1 : how many times a punct sign appears
F_test[:,0] = np.squeeze(num_puncts_test)

def how_many_puncts_in_a_row(X):
    res = np.zeros((len(X), 3))
    for i in range(len(X)):
        comment = X[i]
        for j in range(len(agressive_punct)):
            p = agressive_punct[j]
            l = [len(x) for x in re.findall(r'[%s]+' % p, comment)]
            if l :
                res[i,j] = max(l)
    return res

num_puncts_train_in_a_row = how_many_puncts_in_a_row(X_train)
num_puncts_test_in_a_row = how_many_puncts_in_a_row(X_test)

F_train[:,1:4] = num_puncts_train_in_a_row #Feature 2 : how many times a punct sign appears in a row
F_test[:,1:4] = num_puncts_test_in_a_row

def how_many_minority(X):
    res = np.zeros((len(X), len(minority_tags)))
    for i in range(len(X)):
        comment = X[i]
        for j in range(len(minority_tags)):
            minority = minority_tags[j]
            res[i, j] = comment.count(minority)
    return res

minority_train = how_many_minority(X_train)
minority_test = how_many_minority(X_test)

F_train[:,4:28] = minority_train #Feature 3 : how often each minority is mentionned
F_test[:,4:28] = minority_test

def is_there_profanity(X):
    res = np.zeros((len(X), 1))
    for i in range(len(X)):
        comment = X[i]
        prof = profanity.contains_profanity(comment)
        res[i] = int(prof)
    return res

profanity_train = is_there_profanity(X_train)
profanity_test = is_there_profanity(X_test)

F_train[:,28] = np.squeeze(profanity_train) #Feature 4 : is there profanity ?
F_test[:,28] = np.squeeze(profanity_test)

def length_of_comment(X):
    res = np.zeros((len(X), 1))
    for i in range(len(X)):
        comment = X[i]
        length = len(comment.split())
        res[i] = length
    return res

length_of_train_comments = length_of_comment(X_train)
length_of_test_comments = length_of_comment(X_test)

F_train[:,29] = np.squeeze(length_of_train_comments) #Feature 5 : length of comments
F_test[:,29] = np.squeeze(length_of_test_comments)

###############################################################################
############################ Embedding model ##################################
###############################################################################    
    
EMBEDDING_FILE =  'GloVe/glove.twitter.27B.200d.txt'

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
embeddings = load_embeddings(this_folder + EMBEDDING_FILE)
print("Embedding is now complete")
        
###############################################################################
############################ Data cleaning ####################################
###############################################################################

cleaning_special_chars = True
cleaning_negations = True
cleaning_numbers = True

def clean_special_chars(X_train, X_test):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    for comment_train, comment_test in  zip(X_train, X_test):
        for p in punct:
            comment_train = comment_train.replace(p, ' ')
            comment_test = comment_test.replace(p, ' ')

def clean_negations(X_train, X_test):
    for comment_train, comment_test in  zip(X_train, X_test):
        comment_train.replace("n't", 'not')
        comment_test.replace("n't", 'not')

def clean_numbers(X_train, X_test):
    for comment_train, comment_test in  zip(X_train, X_test):
        re.sub(r'[0-9]+', '0', comment_train)
        re.sub(r'[0-9]+', '0', comment_test)
       

if cleaning_special_chars:
    clean_special_chars(X_train, X_test)

if cleaning_negations:
    clean_negations(X_train, X_test)

if cleaning_numbers:
    clean_numbers(X_train, X_test)

###############################################################################
############################## Tokenization ###################################
###############################################################################

X = np.r_[X_train, X_test]

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

def from_indexed_words_to_vectors(X, embedding_matrix):
    n, m = X.shape
    _, embedding_size = embedding_matrix.shape
    res = np.zeros((n,m*embedding_size))
    for i in range(n):
        for j in range(m):
            index = X[i,j]
            vector = embedding_matrix[index, :]
            res[i, j*embedding_size:(j+1)*embedding_size] = vector
    return res

X_train = from_indexed_words_to_vectors(X_train, embedding_matrix)

###############################################################################
####################### Merging Vector + features #############################
###############################################################################

X_train = [X_train[i].tolist() + F_train[i, :].tolist() for i in range(len(X_train))]
#X_test = [X_test[i].tolist() + F_test[i, :].tolist() for i in range(len(X_test))]

###############################################################################
################################ Modeling #####################################
###############################################################################

Model_Name = 'RF_FE_Twitter_v0'
path_model = this_folder + 'Models/' 

roc_auc = make_scorer(roc_auc_score)

def train_classifier(X_train, y_train_encoded):
    logging.info("Classifier training")
    # Find the optimal Random Forest Classifier Hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)]
    max_features = ['auto', 'sqrt']
    criterion = ['gini', 'entropy']
    max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
    max_depth.append(None)
    rfc = RandomForestClassifier()
    random_grid = {'n_estimators': n_estimators, 'criterion': criterion, 'max_features': max_features,'max_depth': max_depth}
    rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, scoring = roc_auc, cv = 3, verbose=2, random_state=42)
    rfc_random.fit(X_train, np.array(y_train_encoded))
    best_parameters = rfc_random.best_params_
    model = RandomForestClassifier(n_estimators = best_parameters['n_estimators'], criterion = best_parameters['criterion'], max_features = best_parameters['max_features'], max_depth = best_parameters['max_depth'])
    model.fit(X_train, np.array(y_train_encoded))
    model_file = os.path.join(path_model, Model_Name)
    pickle.dump(model, open(model_file, 'wb'))
    logging.info("Classification model saved on :{}".format(model_file))

    training_predictions = model.predict(X_train)
    
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(roc_auc(X_train, y_train_encoded)))
    logging.info('Training F1 score: {}'.format(f1_score(y_train, training_predictions, average='weighted')))
    logging.info("Saving classification model")
    return model

model = train_classifier(X_train, y_train_encoded)
















