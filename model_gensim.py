#!/usr/bin/env python3.6.4
# -*-coding:UTF-8 -*
import os
import json
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import *
from gensim.test.utils import get_tmpfile
import spacy
import csv
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool

"""
KAGGLE - COMPETITION
DEVELOPPER : DLA 
DATE : 04-04-2019
DESCRIPTION :
THIS MODULE BUILD A GENSIM TRAINING DATASET AND A DOC2VEC MODEL, CAN BE CALLED FOR WORD VECTORIZATION
LAST DEBUG = 04-04-2019
"""

##########################################################################################################################
###################################################### CREDENTIALS #######################################################
##########################################################################################################################

""" GET DATA AND TRANSFORM IT INTO JSON FILE """

this_folder = os.path.dirname(os.path.abspath(__file__))
path = this_folder+'/jigsaw-unintended-bias-in-toxicity-classification/'
train_file_name = 'train.csv'
data_file = os.path.join(path,train_file_name)
raw_dataset = os.path.join(path,'train.json')
path_output = this_folder+'/OUTPUT/'
dataset_file = os.path.join(path_output,'training_dataset.json')
path_model = this_folder+'/MODEL/'
#Read CSV File
def read_csv(file_name, json_file, format):
    csv_rows = []
    with open(file_name,"r") as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            csv_rows.extend([{title[i]:row[title[i]] for i in range(len(title))}])
        write_json(csv_rows, json_file, format)

#Convert csv data into json and write it
def write_json(data, json_file, format):
    with open(json_file, "w") as f:
            json.dump(data,f,indent=2)

#Transfrom csv train file into json
read_csv(file_name=data_file,json_file=raw_dataset,format=None)

print("JSONIZED RAW DATASET FILE IS READY ON",raw_dataset)

nlp = spacy.load('en_coref_lg')

# SET VECTOR SPACE DIMENSION
VECTOR_DIMENSION = 60
GENSIM_MODEL_NAME = "rlcalm_doc2vec_model_v0"

##########################################################################################################################
####################################################### FUNCTIONS ########################################################
##########################################################################################################################

def build_dataset(file_name):
    """ THIS MODULE TRANSFORMS THE RAW DATA INTO GENSIM DATASET """
    response = []
    #path = this_folder +'/jigsaw-unintended-bias-in-toxicity-classification/'
    #data_file = os.path.join(path,file_name)
    with open (file_name,'r') as json_file:
        data = json.load(json_file)
    if data['comment_text'] != None:
        doc = nlp(data['comment_text'])
        sentences = [sent for sent in doc.sents]
        for sent in sentences:
            response.append([token for token in sent.text.split()])
    return response

def train_model(training_data):
    """ THIS MODULE TRAIN GENSIM MODEL ON THE BUILT DATASET """
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(training_data)] 
    print("DOCUMENTS IN DATASET",len(documents))
    model = Doc2Vec(documents, vector_size=VECTOR_DIMENSION, window=50, min_count=1, workers=50,epochs=60)
    model_name = GENSIM_MODEL_NAME
    fname = os.path.join(path_model,model_name)
    model.save(fname)
    return fname

def vectorize(chunk,fname):
    model = Doc2Vec.load(fname)
    return model.infer_vector([t for t in chunk.split()])

##########################################################################################################################
###################################################### BUILD MODEL #######################################################
##########################################################################################################################

print("======================== DATASET BUILDING ===========================")
ts1 = datetime.now().timestamp()
training_data = build_dataset(file_name=raw_dataset)
with open (dataset_file,'w') as json_file:
    json.dump(training_data,json_file,indent=2)
ts2 = datetime.now().timestamp()

print("======================== DATASET STATISTICS ===========================")
print("DATASET FILE IS READY AT",dataset_file)
print("SENTENCES IN DATASET = ",len(training_data))
print("DATASET PREPARED IN ",int(ts2-ts1)," SECONDES")
print("======================== MODEL BUILDING ===========================")
fname = train_model(training_data=training_data)
ts3 = datetime.now().timestamp()
print("MODEL IS READY ON ",fname)
print("MODEL TRAINED IN ",int(ts3-ts2)," SECONDES")
