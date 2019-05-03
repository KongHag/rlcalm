#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import random
import logging
import datetime
import numpy as np
import pandas as pd

from gensim.models import doc2vec

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
this_folder_path = "/home/ubuntu/Documents/Kaggle/"
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
        labeled.append(doc2vec.TaggedDocument(v.split(), label)) # Comment + Tag
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
    for minority in all_identity_columns:
        df[minority] = np.where(df[minority] == 1, True, False)
        
def from_Y_to_label(y, i):
    label = str(i) + '_'
    for j in range(len(all_identity_columns)):
        if y[j][i]:
            label += all_identity_columns[j]

def read_minorities_dataset(initial_dataset, test_size):
    print("Reading and labelling the data...")
    dataset = convert_df_to_bool_minorities(initial_dataset)
    X = dataset.Reviews
    Y = np.array([dataset.target.tolist(), dataset.male.tolist(), dataset.female.tolist(), 
                        dataset.homosexual_gay_or_lesbian.tolist(), dataset.christian.tolist(), dataset.jewish.tolist(), dataset.muslim.tolist(), 
                        dataset.black.tolist(), dataset.white.tolist(), dataset.psychiatric_or_mental_illness.tolist(), dataset.severe_toxicity.tolist(), dataset.obscene.tolist(),
                        dataset.identity_attack.tolist(), dataset.insult.tolist(), dataset.threat.tolist(), dataset.rating.tolist(),dataset.toxicity_annotator_count.tolist(), 
                        dataset.identity_annotator_count.tolist(),dataset.article_id.tolist()])                       
    X = minority_label_sentences(X, Y)
    return X

def minority_label_sentences(x, y):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be i_True or i_False where i is
    a dummy variable and where True means toxic. This is the first part of our embedding".
    """
    labeled = []
    for i, v in enumerate(x):
        label = from_Y_to_label(y, i)
        v = clean(text=v)
        labeled.append(doc2vec.TaggedDocument(v.split(), label)) # Comment + Tag
    return labeled

###############################################################################
#################### Building doc2vec model ###################################
###############################################################################
    
WORKERS = 8
ITERATIONS = 15
VOCAB_SIZE = 1669827
LEARNING_RATE = 0.4
BATCH_SIZE = 128
VERBOSE = 1

model_path = this_folder_path + "Doc2Vec/"

toxic_model_name = model_path + "toxic_doc2vec_model"
minority_model_name = model_path + "minority_doc2vec_model"
 
def train_doc2vec(corpus, model_file):
    print("Building the Doc2Vec model")
    d2v = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
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
        print("Starting epoch number ", epoch, ' out of ', ITERATIONS, ' epochs.')
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
    
if __name__ == "__main__":
    print("Reading toxic data...")
    all_toxic_data = read_toxicity_dataset(df_train)
    print("Training the doc2vec Model for Toxicity...")
    d2v_model_toxic = train_doc2vec(all_toxic_data, toxic_model_name)
    print("Successfully saved !\n")
    print("Reading minority data...")
    all_minority_data = read_minorities_dataset(df_train)
    print("Training the doc2vec Model for Minorities...")
    d2v_model = train_doc2vec(all_minority_data, minority_model_name)

###############################################################################
#################### Loading models and getting vectors #######################
###############################################################################

d2v_model_toxicity = doc2vec.Doc2Vec.load(toxic_model_name)
d2v_model_minority = doc2vec.Doc2vec.load(minority_model_name)






























