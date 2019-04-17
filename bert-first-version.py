#!/usr/bin/env python3.6.4
# -*-coding:UTF-8 -*
import logging
import random
import os

import csv
import numpy as np
import pandas as pd
import pickle


from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from tensorflow import keras
import os
import re

"""
COPYRIGHT A2IM-ROBOTADVISORS & INSTITUT LOUIS BACHELIER
DEVELOPPER : MELS-ILB 
DATE : 07-04-2019
DESCRIPTION :
THIS MODULE BUILD A MODEL FOR KAGGLE CHALLENGE
"""

####################
###### PATHS #######
####################

# Set the output directory for saving model file
BERT_MODEL_NAME = 'bert_predicter'
OUTPUT_DIR = 'OUTPUT_DIR_'+BERT_MODEL_NAME #@param {type:"string"}

#####################
##### LOAD DATA #####
#####################

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

this_folder = '/media/ubuntu/Data/kaggle-toxicity' #os.path.dirname(os.path.abspath(__file__))
path_input = this_folder+'/jigsaw-unintended-bias-in-toxicity-classification/'
path_model = this_folder+'/OUTPUT_DIR_MODEL_FILES/'
path = os.path.join(path_input,'train.csv')
path_test = os.path.join(path_input,'test.csv')

def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)
    
def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df

def read_dataset(path,test_size,data_size_threshold=5000):
    initial_dataset = pd.read_csv(path, header=0, sep=",").sample(data_size_threshold)
    # ADD to the training other tags
    # ADD different encoding for comment_text
    dataset = convert_dataframe_to_bool(initial_dataset)
    
    X = dataset.comment_text
    Y = np.array([dataset.target.tolist(), dataset.male.tolist(), dataset.female.tolist(), 
                        dataset.homosexual_gay_or_lesbian.tolist(), dataset.christian.tolist(), dataset.jewish.tolist(), dataset.muslim.tolist(), 
                        dataset.black.tolist(), dataset.white.tolist(), dataset.psychiatric_or_mental_illness.tolist()])

    x_train, x_test, y_train, y_test = train_test_split(X, np.transpose(Y), random_state=0, test_size=test_size)
    train = np.concatenate((np.array(x_train).reshape(-1,1),np.array(y_train)),axis=1)
    test = np.concatenate((np.array(x_test).reshape(-1,1),np.array(y_test)),axis=1)
    return pd.DataFrame(train, columns=['comment_text','target']+identity_columns), pd.DataFrame(test,columns=['comment_text','target']+identity_columns)

##############################
##### DATA PREPROCESSING #####
##############################

DATA_COLUMN = 'comment_text'
LABEL_COLUMN = 'target'
#label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [False, True]

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

##############################
######## KAGGLE METRIC #######
##############################

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positiv

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(list(y_true), list(y_pred))
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
    return metrics.roc_auc_score(list(true_labels), list(predicted_labels))

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

##############################
###### MODEL FINE-TUNED ######
##############################
    
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
  bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
  bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn

##############################
######## GET PREDICT #########
##############################
    
def getPrediction(in_sentences):
  labels = label_list
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

    
def getFinalBiais(test,by_minority_biais=False):
    validate_df = pd.DataFrame.copy(test)
    prediction_list = getPrediction(list(test.comment_text.values))
    validate_df[BERT_MODEL_NAME] = [np.exp(array[1])/np.sum(np.exp(array)) for (comment_text, array, binary_pred) in prediction_list]
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, BERT_MODEL_NAME, TOXICITY_COLUMN)
    performance = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, BERT_MODEL_NAME))
    return_dict_minority_biais = {"final_minority_biais_metric": performance}
    
    if by_minority_biais:
        for subgroup in identity_columns:
            return_dict_minority_biais.update([(subgroup+'_biais_metric', compute_subgroup_auc(validate_df,subgroup,BERT_MODEL_NAME, TOXICITY_COLUMN))])
    
    return return_dict_minority_biais

##############################
###### LEARNING PARAMS #######
##############################
  

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

##############################
######## PROD SCRIPT #########
##############################

PRODUCTION = True
NORMAL_TESTING = True
BIAIS_TESTING = True

if PRODUCTION:
    
    ###########################################################################
    
    # Create model output directory
    tf.gfile.MakeDirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
    
    # Load data from csv files %%% data full size : 1804874
    print(f'Beginning loading data!')
    current_time = datetime.now()
    train, test = read_dataset(path,test_size=0.30, data_size_threshold=1804874)
    print("Loading data took time ", datetime.now() - current_time)

    ###########################################################################

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    print(f'Beginning adapting data to bert format')
    current_time = datetime.now()
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)
    
    print("Loading data took time ", datetime.now() - current_time)
    
    ###########################################################################
    
    # Tokenize data with tensorflow package
    tokenizer = create_tokenizer_from_hub_module()

    # Convert our train and test features to InputFeatures that BERT understands.    
    print(f'Beginning converting data to features')
    current_time = datetime.now()
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print("Converting data to features took time ", datetime.now() - current_time)
    
    ##########################################################################
    
    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    
    # Specify output directory and number of checkpoint steps to save
    print(f'Beginning running config & creating bert model & estimator')
    current_time = datetime.now()
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    # build model
    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    # build estimator
    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})
    
    print("Running config & creating bert model & estimator took time ", datetime.now() - current_time)

    ##########################################################################
    
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)
    
    ##########################################################################
    
    if NORMAL_TESTING: 
        
        # Create an input function for testing. drop_remainder = True for using TPUs.
        test_input_fn = run_classifier.input_fn_builder(
            features=test_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)
        
        print(f'Beginning Testing!')
        current_time = datetime.now()
        estimator_result = estimator.evaluate(input_fn=test_input_fn, steps=None)
        print('\n\n\nGlobal scores on test set : ')
        print(estimator_result)
        print('\n\n\n')
        print("Testing took time ", datetime.now() - current_time)
    
    ##########################################################################
    
    if BIAIS_TESTING: 
    
        print(f'Beginning computing biais metric!')
        current_time = datetime.now()
        final_biais_dict = getFinalBiais(test, by_minority_biais=True)
        print(final_biais_dict)
        print("computing biais metric took time ", datetime.now() - current_time)
    
    
    
    
    
    
