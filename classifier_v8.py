#!/usr/bin/env python3.6.4
# -*-coding:UTF-8 -*
import logging
import random
import os
import re

import csv
import numpy as np
import pandas as pd
import pickle

from gensim.models import doc2vec
import sklearn
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from datetime import *

from keras import metrics
from keras import regularizers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,LearningRateScheduler
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, UpSampling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

"""
COPYRIGHT A2IM-ROBOTADVISORS & INSTITUT LOUIS BACHELIER
DEVELOPPER : DLA 
DATE : 07-04-2019
DESCRIPTION :
THIS MODULE BUILD A MODEL FOR KAGGLE CHALLENGE
TESTED ON 09-04-2019
"""

GENSIM_MODEL_NAME = "KAGGEL_MODEL_v3"
CLASSIFICATION_MODEL_NAME = 'CLASSIFIER_v4'

tags = ['target', 'severe_toxicity', 'obscene',
       'identity_attack', 'insult', 'threat', 'asian', 'atheist', 'bisexual',
       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
       'jewish', 'latino', 'male', 'muslim', 'other_disability',
       'other_gender', 'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white',
       'rating', 'funny', 'wow',
       'sad', 'likes', 'disagree', 'sexual_explicit']

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

all_identity_columns = ['asian', 'atheist', 'bisexual',
       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
       'jewish', 'latino', 'male', 'muslim', 'other_disability',
       'other_gender', 'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white'] # True / False

toxicity_subtypes_columns = ['severe_toxicity', 'obscene',
       'identity_attack', 'insult', 'threat'] # True / False

articles_info_columns = ['funny', 'wow',
       'sad', 'likes', 'disagree', 'sexual_explicit'] # Rating = approved/rejected  others = number of votes

TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

annotator_bias = ['toxicity_annotator_count', 'identity_annotator_count']

article_colum = ['article_id']

WORKERS = 30
ITERATIONS = 60
VOCAB_SIZE = 1669827
LEARNING_RATE = 0.4
LEARNING_EPOCHS = 10000
BATCH_SIZE = 128
VERBOSE = 1

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positiv

this_folder = os.path.dirname(os.path.abspath(__file__))
path_input = this_folder+'/../jigsaw-unintended-bias-in-toxicity-classification/'
path_model = this_folder+'/../MODEL/'
path = os.path.join(path_input,'train.csv')
path_test = os.path.join(path_input,'test.csv')
submission_file = os.path.join(path_input,'submission-1.csv')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logging.info("Setting up for GPU calculations ")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

logging.info("GPU and memory saving setup OK")

def clean_emojis(text):
    """ Clean comments_text from emojis """
    logging.info("Disambiguate comments_text emojis")
    text = re.sub(r'(<3|:\*)', ' Love ', text)
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' Wink ', text)
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' Sad ', text)
    return text

def convert_to_bool(df, col_name):
    logging.info("Dataset normalization")
    if col_name == 'rating':
        df[col_name] = np.where(df[col_name] == 'approved', True, False)
    elif col_name == 'toxicity_annotator_count':
        df[col_name] = np.where(df[col_name] >= 8.78, True, False)
    elif col_name == 'identity_annotator_count':
        df[col_name] = np.where(df[col_name] >= 1.44, True, False)
    elif col_name == 'article_id':
        df[col_name] = df[col_name]
    else:
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns + toxicity_subtypes_columns + ['rating'] + annotator_bias + article_colum:
        convert_to_bool(bool_df, col)
    return bool_df

def compute_auc(y_true, y_pred):
    try:
        return sklearn.metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        print(subgroup,flush=True)
        record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[dataset[subgroup]])
            }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return sklearn.metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

def read_dataset(path,test_size):
    initial_dataset = pd.read_csv(path, header=0, sep=",")
    dataset = convert_dataframe_to_bool(initial_dataset)
    
    X = dataset.comment_text
    Y = np.array([dataset.target.tolist(), dataset.male.tolist(), dataset.female.tolist(), 
                        dataset.homosexual_gay_or_lesbian.tolist(), dataset.christian.tolist(), dataset.jewish.tolist(), dataset.muslim.tolist(), 
                        dataset.black.tolist(), dataset.white.tolist(), dataset.psychiatric_or_mental_illness.tolist(), dataset.severe_toxicity.tolist(), dataset.obscene.tolist(),
                        dataset.identity_attack.tolist(), dataset.insult.tolist(), dataset.threat.tolist(), dataset.rating.tolist(),dataset.toxicity_annotator_count.tolist(), 
                        dataset.identity_annotator_count.tolist(),dataset.article_id.tolist()])                        
    doc_tags = np.array([dataset.male.tolist(), dataset.female.tolist(), 
                        dataset.homosexual_gay_or_lesbian.tolist(), dataset.christian.tolist(), dataset.jewish.tolist(), dataset.muslim.tolist(), 
                        dataset.black.tolist(), dataset.white.tolist(), dataset.psychiatric_or_mental_illness.tolist(), dataset.severe_toxicity.tolist(), dataset.obscene.tolist(),
                        dataset.identity_attack.tolist(), dataset.insult.tolist(), dataset.threat.tolist(),dataset.toxicity_annotator_count.tolist(), 
                        dataset.identity_annotator_count.tolist(),dataset.article_id.tolist()])
    x_train, x_test, y_train, y_test, doc_tags_train, doc_tags_test = train_test_split(X, np.transpose(Y),np.transpose(doc_tags), random_state=0, test_size=test_size)
    doc_tags_train = np.append(doc_tags_train,np.transpose(['Train' for i in range(len(doc_tags_train))]))
    doc_tags_test = np.append(doc_tags_test,np.transpose(['Test' for i in range(len(doc_tags_test))]))
    x_train = label_sentences(x_train, doc_tags_train)
    x_test = label_sentences(x_test, doc_tags_test)
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data

def label_sentences(corpus, labels):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """
    labeled = []
    for i, v in enumerate(corpus):
        document_tags = [labels[j] for j in range(17)]
        label = labels[17] + '_' + str(i)
        document_tags.append(label)
        v = clean_emojis(text=v)
        labeled.append(doc2vec.TaggedDocument(v.split(), document_tags)) # use multiple tags (similar to Wiki2Doc)
    return labeled

def train_doc2vec(corpus):
    WORKERS = 150
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=1,  # Ignores all words with total frequency lower than this
                          window=10,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=300,  # Dimensionality of the generated feature vectors
                          workers=WORKERS,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=0)  # dm defines the training algorithm. If dm=1 means 'distributed memory' (PV-DM)
                                 # and dm =0 means 'distributed bag of words' (PV-DBOW)
    d2v.build_vocab(corpus)

    logging.info("Training Doc2Vec model")
    for epoch in range(ITERATIONS):
        logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002
        # fix the learning rate, no decay
        d2v.min_alpha = d2v.alpha

    logging.info("Saving trained Doc2Vec model")
    model_file = os.path.join(path_model,GENSIM_MODEL_NAME)
    d2v.save(model_file)
    return d2v

def get_vectors(doc2vec_model, corpus, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    corpus_size = len(corpus)
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        try:
            prefix = vectors_type + '_' + str(i)
            if vectors_type == 'Train':
                try :
                    vectors[i] = doc2vec_model.docvecs.doctags[prefix]
                except:
                    vectors[i] = doc2vec_model.docvecs.doctags['Test_' + str(i)]
            else:
                try :
                    vectors[i] = doc2vec_model.docvecs.doctags[prefix]
                except:
                    vectors[i] = doc2vec_model.docvecs.doctags['Train_' + str(i)]
        except:
            vectors[i] = doc2vec_model.infer_vector(corpus[i][0])
    return vectors

def schedule(epoch, lr):
    if epoch % 10 !=0 and epoch >0:
        return lr
    return 0.001

def classifier_model(d2v,training_vectors, training_labels,testing_vectors, testing_labels,verbose=VERBOSE):
    logging.info("Build train and tests vectors")
    train_vectors = get_vectors(d2v, training_vectors, 300, 'Train')
    logging.info("train vectors shape : {}".format(train_vectors.shape))
    logging.info("Embedding dimension for train {}".format(train_vectors.shape[1]))
    embedding_dim = train_vectors.shape[1]
    train_vectors = np.reshape(train_vectors,(train_vectors.shape[0],1,train_vectors.shape[1]))
    training_labels = np.reshape(training_labels,(training_labels.shape[0],1,training_labels.shape[1]))
    logging.info("Setting up classifier")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=5, min_lr=0.00000000000001, verbose=verbose)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=20, verbose=verbose, mode='auto', baseline=None, restore_best_weights=True)
    schedduler = LearningRateScheduler(schedule, verbose=verbose)
    activations = 'tanh'
    input_shape = embedding_dim
    inputs = Input(shape=(None,input_shape), name='encoder_input')
    x = inputs
    x = Conv1D(filters = 250, kernel_size = 5, bias_initializer='zeros',kernel_initializer='glorot_normal', activation=activations,padding='same')(x)
    x = Dropout(LEARNING_RATE)(x)
    x = MaxPooling1D(pool_size = 2,padding='same')(x)
    x = Conv1D(filters = 200, kernel_size = 10, bias_initializer='zeros',kernel_initializer='glorot_normal', activation=activations,padding='same')(x)
    x = Dropout(LEARNING_RATE)(x)
    x = MaxPooling1D(pool_size = 2,padding='same')(x)
    x = Conv1D(filters = 150, kernel_size = 15, bias_initializer='zeros',kernel_initializer='glorot_normal', activation=activations,padding='same')(x)
    x = Dropout(LEARNING_RATE)(x)
    x = MaxPooling1D(pool_size = 2,padding='same')(x)
    x = Conv1D(filters = 100, kernel_size = 20, bias_initializer='zeros',kernel_initializer='glorot_normal', activation=activations,padding='same')(x)
    x = Dropout(LEARNING_RATE)(x)
    x = MaxPooling1D(pool_size = 2,padding='same')(x)
    x = Conv1D(filters = 50, kernel_size = 25, bias_initializer='zeros',kernel_initializer='glorot_normal', activation=activations,padding='same')(x)
    x = Dropout(LEARNING_RATE)(x)
    x = MaxPooling1D(pool_size = 2,padding='same')(x)
    latent = Dense(19,bias_initializer='zeros',kernel_initializer='glorot_normal',activation=activations)(x)
    model = Model(inputs, latent, name='classifier')
    print(model.summary(),flush=True)
    logging.info("Compiling classifier")
    model.compile(optimizer='adam', 
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    model.fit(train_vectors, 
                training_labels, 
                batch_size=BATCH_SIZE, 
                epochs=LEARNING_EPOCHS, 
                verbose=verbose, 
                validation_split=0.3, 
                callbacks=[early_stopping,reduce_lr,schedduler])
    model_file = os.path.join(path_model,CLASSIFICATION_MODEL_NAME)
    pickle.dump(model, open(model_file, 'wb'))
    logging.info('Classification model saved at {}'.format(model_file))
    return model

def test_classifier(d2v,classifier,testing_vectors,testing_labels):
    logging.info("Build tests vectors")
    test_vectors = get_vectors(d2v, testing_vectors, 300, 'Test')
    logging.info("Reshape tests vectors")
    test_vectors = np.reshape(test_vectors,(test_vectors.shape[0],1,test_vectors.shape[1]))
    logging.info("Prepare prediction")
    testing_predictions = classifier.predict(test_vectors)
    logging.info("Construct prediction dataframe")
    validate_df = pd.DataFrame(testing_labels,columns=['target', 'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness','severe_toxicity', 'obscene',
                        'identity_attack', 'insult', 'threat', 'rating','toxicity_annotator_count', 
                        'identity_annotator_count','article_id']).astype(bool)
    logging.info('Prediction testing shape {}'.format(testing_predictions.shape))
    testing_predictions = np.reshape(testing_predictions,(testing_predictions.shape[0],testing_predictions.shape[2]))
    logging.info('Prediction testing shape {}'.format(testing_predictions.shape))
    logging.info('New prediction testing shape {}'.format(testing_predictions[:,0].shape))
    validate_df[GENSIM_MODEL_NAME] = pd.DataFrame(testing_predictions[:,0]).astype(bool)
    #validate_df[GENSIM_MODEL_NAME] = np.where(validate_df[GENSIM_MODEL_NAME] >= 0.5, True, False)
    validate_df.head() 
    logging.info("Validate_df heads {}".format(validate_df.columns.values.tolist()))
    print("validate_df types",validate_df.dtypes,flush=True)
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, GENSIM_MODEL_NAME, TOXICITY_COLUMN)
    performance = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, GENSIM_MODEL_NAME))
    logging.info('Testing Bias Metric: {0:.2%}'.format(performance))
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions[:,0])))
    logging.info('Testing F1 score: {0:.2%}'.format(f1_score(testing_labels[:,0], testing_predictions[:,0], average='weighted')))

def prepare_submission(d2v,classifier,path_test):
    logging.info('Uploading comments to test')
    test = pd.read_csv('../jigsaw-unintended-bias-in-toxicity-classification/test.csv')
    logging.info('Predicting comments toxicity')
    texts = test[TEXT_COLUMN]
    X = []
    for i, v in enumerate(texts):
        X.append(doc2vec.TaggedDocument(v.split(), 'Validate_' + str(i)))
    vectors = get_vectors(d2v, X , 300, 'Validate')
    vectors = np.reshape(vectors,(vectors.shape[0],1,vectors.shape[1]))
    submission = pd.read_csv('../jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
    predictions = classifier.predict(vectors)   
    predictions = np.reshape(predictions,(predictions.shape[0],predictions.shape[2]))
    submission['prediction'] = pd.DataFrame(predictions[:,0]).astype(bool)
    submission.to_csv(submission_file)
    logging.info('File ready for submission')

if __name__ == "__main__":
    ts1 = datetime.now().timestamp()
    x_train, x_test, y_train, y_test, all_data = read_dataset(path,test_size=0.30)
    ts2 = datetime.now().timestamp()
    logging.info('Training predicted classes: {}'.format(np.unique(y_test[:,0])))
    logging.info("Dataset preparation execution time : {} secondes".format(int(ts2-ts1)))
    # model already trained // uncomment the following line to train embedding model
    #d2v_model = train_doc2vec(all_data)
    ts3 = datetime.now().timestamp()
    logging.info("Gensim model training execution time : {} secondes".format(int(ts3-ts2)))
    model_file = os.path.join(path_model,GENSIM_MODEL_NAME)
    d2v_model = doc2vec.Doc2Vec.load(model_file)
    logging.info("Gensim model loaded without any problems")
    logging.info("Training classification model")
    # Classifier already trained // uncomment the following line to train the classifier model
    classifier = classifier_model(d2v_model, x_train, y_train,x_test, y_test,verbose=VERBOSE)
    model_file = os.path.join(path_model,CLASSIFICATION_MODEL_NAME)
    classifier = pickle.load(open(model_file,'rb'))
    ts4 = datetime.now().timestamp()    
    logging.info("Classification model training execution time : {} secondes".format(int(ts4-ts3)))
    logging.info("Model scores and performances")
    test_classifier(d2v_model,classifier,x_test,y_test)
    ts5 = datetime.now().timestamp()
    logging.info("Classification model testing execution time : {} secondes".format(int(ts5-ts4)))
    logging.info("Preparing submission")
    prepare_submission(d2v_model,classifier,path_test)
    logging.info("Submission ready")
