#!/usr/bin/env python3.6.4
# -*-coding:UTF-8 -*
import logging
import random
import os

import csv
import numpy as np
import pandas as pd
import pickle

from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

from datetime import *

"""
COPYRIGHT A2IM-ROBOTADVISORS & INSTITUT LOUIS BACHELIER
DEVELOPPER : DLA 
DATE : 07-04-2019
DESCRIPTION :
THIS MODULE BUILD A MODEL FOR KAGGLE CHALLENGE
"""


GENSIM_MODEL_NAME = "KAGGEL_MODEL_v2"
CLASSIFICATION_MODEL_NAME = 'RFC_v2'

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

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
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

articles_info_columns = ['rating', 'funny', 'wow',
       'sad', 'likes', 'disagree', 'sexual_explicit'] # Rating = approved/rejected  others = number of votes

TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

WORKERS = 30
ITERATIONS = 60

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positiv

this_folder = os.path.dirname(os.path.abspath(__file__))
path_input = this_folder+'/jigsaw-unintended-bias-in-toxicity-classification/'
path_model = this_folder+'/MODEL/'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
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
    return metrics.roc_auc_score(true_labels, predicted_labels)

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
    # ADD to the training other tags
    # ADD different encoding for comment_text
    dataset = convert_dataframe_to_bool(initial_dataset)
    
    X = dataset.comment_text
    Y = np.array([dataset.target.tolist(), dataset.male.tolist(), dataset.female.tolist(), 
                        dataset.homosexual_gay_or_lesbian.tolist(), dataset.christian.tolist(), dataset.jewish.tolist(), dataset.muslim.tolist(), 
                        dataset.black.tolist(), dataset.white.tolist(), dataset.psychiatric_or_mental_illness.tolist()])

    x_train, x_test, y_train, y_test = train_test_split(X, np.transpose(Y), random_state=0, test_size=test_size)
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data

def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    return labeled

def train_doc2vec(corpus):
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
                    vectors[i] = doc2vec_model.docvecs[prefix]
                except:
                    vectors[i] = doc2vec_model.docvecs['Test_' + str(i)]
            else:
                try :
                    vectors[i] = doc2vec_model.docvecs[prefix]
                except:
                    vectors[i] = doc2vec_model.docvecs['Train_' + str(i)]
        except:
            vectors[i] = doc2vec_model.infer_vector(corpus[i][0])
    return vectors

def train_classifier(d2v, training_vectors, training_labels):
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, training_vectors, 300, 'Train')
    # Find the optimal Random Forest Classifier Hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
    max_depth.append(None)
    rfc = RandomForestClassifier(n_jobs=WORKERS)
    random_grid = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth}
    rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = WORKERS)
    rfc_random.fit(train_vectors, np.array(training_labels))
    best_parameters = rfc_random.best_params_
    model = RandomForestClassifier(n_estimators = best_parameters['n_estimators'], max_features = best_parameters['max_features'], max_depth = best_parameters['max_depth'], n_jobs=WORKERS)
    #model = RandomForestClassifier(n_jobs=WORKERS)
    #print("train_vectors shape",train_vectors.shape)
    #print('train_label shape',training_labels.shape)
    model.fit(train_vectors, np.array(training_labels))
    model_file = os.path.join(path_model,CLASSIFICATION_MODEL_NAME)
    pickle.dump(model, open(model_file, 'wb'))
    logging.info("Classification model saved on :{}".format(model_file))
    #model = pickle.load(open(model_file,'rb'))
    training_predictions = model.predict(train_vectors)
    validate_df = pd.DataFrame(training_labels,columns=['target', 'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'])
    validate_df[GENSIM_MODEL_NAME] = pd.DataFrame(training_predictions[:,0])
    validate_df.head()     
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, GENSIM_MODEL_NAME, TOXICITY_COLUMN)
    performance = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, GENSIM_MODEL_NAME))

    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    logging.info('Training Bias Metric: {}'.format(performance))
    logging.info("Saving classification model")

    return model

def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Classifier testing")
    test_vectors = get_vectors(d2v, testing_vectors, 300, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    validate_df = pd.DataFrame(testing_labels,columns=['target', 'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'])
    print("testing prediction shape",testing_predictions.shape)
    validate_df[GENSIM_MODEL_NAME] = pd.DataFrame(testing_predictions[:,0])
    validate_df.head()    
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, GENSIM_MODEL_NAME, TOXICITY_COLUMN)
    performance = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, GENSIM_MODEL_NAME))
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))
    logging.info('Training Bias Metric: {}'.format(performance))

path = os.path.join(path_input,'train.csv')
path_test = os.path.join(path_input,'test.csv')

production = True
if production:
    if __name__ == "__main__":
        ts1 = datetime.now().timestamp()
        x_train, x_test, y_train, y_test, all_data = read_dataset(path,test_size=0.30)
        ts2 = datetime.now().timestamp()
        print("DATASET PREPARATION EXECUTION TIME = ",int(ts2-ts1))
        d2v_model = train_doc2vec(all_data)
        ts3 = datetime.now().timestamp()
        print("GENSIM MODEL TRAINING EXECUTION TIME = ",int(ts3-ts2))
        #model_file = os.path.join(path_model,GENSIM_MODEL_NAME)
        #d2v_model = doc2vec.Doc2Vec.load(model_file)
        #print("GENSIM MODEL LOADED WITHOUT BUGGS")
        #print("TESTING MODEL WITH CHALLENGE TEST DATA")
        #classifier = train_classifier(d2v_model, x_train, y_train)
        #model_file = os.path.join(path_model,CLASSIFICATION_MODEL_NAME)
        #classifier = pickle.load(open(model_file,'rb'))
        #ts4 = datetime.now().timestamp()
        #print("CLASSIFICATION MODEL TRAINING EXECUTION TIME = ",int(ts4-ts3))
        #test_classifier(d2v_model, classifier, x_test, y_test)
        #ts5 = datetime.now().timestamp()
        #print("CLASSIFICATION MODEL TESTING EXECUTION TIME = ",int(ts5-ts4))

else:
    # verify dataset tags
    file_csv = csv.reader(open(path_test)) 
    dataset = pd.read_csv(path_test, header=0, sep=",")
    print("dataset keys",dataset.keys())
