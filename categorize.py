#!/usr/bin/env python3.6.4
# -*-coding:UTF-8 -*
import os
import json
from datetime import *

"""
KAGGLE - COMPETITION
DEVELOPPER : DLA 
DATE : 04-04-2019
DESCRIPTION :
THIS MODULE SEPARATES TRAINING DATASET INTO SUB-SETS FOR TWO CLASSIFICATION:
MINORITY & TOXICITIY MEASURES
LAST DEBUG = 04-04-2019
"""

toxicity_tags = ['severe_toxicity',
'obscene',
'identity_attack',
'insult',
'threat']

minority_tags = ['asian',
'atheist',
'bisexual',
'black',
'buddhist',
'christian',
'female',
'heterosexual',
'hindu',
'homosexual_gay_or_lesbian',
'intellectual_or_learning_disability',
'jewish',
'latino',
'male',
'muslim',
'other_disability',
'other_gender',
'other_race_or_ethnicity',
'other_religion',
'other_sexual_orientation',
'physical_disability',
'psychiatric_or_mental_illness',
'transgender',
'white']

this_folder = os.path.dirname(os.path.abspath(__file__))
path = this_folder + '/jigsaw-unintended-bias-in-toxicity-classification/'
output_path = this_folder + '/OUTPUT/'
json_data_file = os.path.join(path,'train.json')
with open(json_data_file,'r') as json_file:
    json_data = json.load(json_file)

ts1 = datetime.now().timestamp()
dataset_minority_category = dict()
tagged = []
for tag in minority_tags:
    dataset = [{'id':data['id'],'comment_text':data['comment_text']} for data in json_data if data[tag]==1]
    dataset_minority_category[tag] = dataset
    tagged += dataset
dataset_minority_category['no_tag'] = [data for data in json_data if data not in dataset]
minority_file = os.path.join(output_path,'dataset_minority_category.json')
with open(minority_file,'w') as json_file:
    json.dump(dataset_minority_category,json_file,indent=4)
print("MINORITY CLASSIFICATION DATASET READY")
ts2 = datetime.now().timestamp()
print("MINORITY DATASET PREPARED IN ", int(ts2-ts1)," SECONDES")
dataset_toxicity = [{'id':data['id'],'comment_text':data['comment_text'],'target':data['target'],'severe_toxicity':data['severe_toxicity'],'obscene':data['obscene'],'identity_attack':data['identity_attack'],'insult':data['insult'],'threat':data['threat']} for data in json_data]
toxicity_file = os.path.join(output_path,'dataset_toxicity_measures.json')
with open(toxicity_file,'w') as json_file:
    json.dump(dataset_toxicity,json_file,indent=4)
ts3 = datetime.now().timestamp() 
print("TOXICITY MEASURES DATASET READY")
print("TOXICITY DATASET PREPARED IN ", int(ts3-ts2)," SECONDES")
