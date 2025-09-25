import itertools
import tqdm
import numpy as np 
import pandas as pd
from Bio import SeqIO
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, fbeta_score, matthews_corrcoef
from multiprocessing import Pool

import os, sys, re
from pathlib import Path
file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)

#print(os.path.realpath(__file__))
#print(Path(pPath).parent)

def fasta_to_df(file, dna_only=True):
    sequences = SeqIO.parse(file, "fasta")
    
    data = []
    
    # Iterate over each sequence in the FASTA file
    for seq_record in sequences:
        desc = seq_record.description
        accession = '_'.join(desc.split('|')[0].split('_')[:2])
        #accession = desc.split('|')[0]
        label = desc.split('|')[1]
        position = re.search("\d+\-\d+", seq_record.id)[0]
        start = int(position.split('-')[0])
        end = int(position.split('-')[1])
        sequence = str(seq_record.seq)
    
        if dna_only:
            if len(set(sequence) - set({'A','T','G','C'})) == 0:
                data.append((accession, str(seq_record.seq), start, end, desc.split('|')[-1], label))
        else:
            data.append((accession, str(seq_record.seq), start, end, desc.split('|')[-1], label))
            
    df = pd.DataFrame(data, columns = ['Accession','Sequence','Start','End','Description','Label'])

    return df

#reference: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary

def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res
    
def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df

def multiindex_dict_to_df(values_dict):
    df = pd.DataFrame.from_dict(values_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df
    
def get_encodings(desc, filename, desc_default_para):

    from ilearnplus.util import (FileProcessing, CheckAccPseParameter)
    
    descriptor = FileProcessing.Descriptor(filename, desc_default_para)
    
    if desc in ['DAC', 'TAC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
        status = descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['DCC', 'TCC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
        status = descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['DACC', 'TACC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
        status = descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
        my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(desc, descriptor.sequence_type, desc_default_para)
        cmd = 'descriptor.' + desc + '(my_property_name, my_property_value)'
        status = eval(cmd)
    else:
        cmd = 'descriptor.' + desc + '()'
        status = eval(cmd)

    X = descriptor.encoding_array[1:][:,2:].astype(float)
    y = descriptor.encoding_array[1:][:,1].astype(int)
    groups = ['_'.join(label.split('_')[:2]) for label in descriptor.encoding_array[1:][:,0]]

    return X,y,groups
    
def train_model(X,y,**kwargs):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    model = kwargs['model']
    data_fold = kwargs['data_fold']
    n_fold = kwargs['n_fold']
    scoring = kwargs['scoring']

    if 'groups' in kwargs.keys():
        groups = kwargs['groups'] 
    elif data_fold == "StratifiedGroupKFold":
        print("groups are needed in StratifiedGroupKFold")

    #define the models to be trained
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
        'NaiveBayes': GaussianNB(),
        'GradientBoosting': GradientBoostingClassifier(),
        'Bagging':BaggingClassifier(n_estimators=100, n_jobs=1)
    }

    clf = models[model]

    try:
        if data_fold == "KFold":
            fold = KFold(n_splits = n_fold)
            scores = cross_validate(clf, X, y, cv = fold, scoring = scoring)
        elif data_fold == "StratifiedKFold":
            fold = StratifiedKFold(n_splits = n_fold)
            scores = cross_validate(clf, X, y, cv = fold.split(X, y), scoring = scoring)    
        elif data_fold == "StratifiedGroupKFold":
            fold = StratifiedGroupKFold(n_splits = n_fold)
            scores = cross_validate(clf, X, y, cv = fold.split(X, y, groups), scoring = scoring)
        
        return scores, fold, clf
    except Exception as e:
        print('error {} in train_model'.format(e))


def main(file, **train_params):
    #return {(file, train_params['model'], train_params['encoding'], train_params['data_fold'], train_params['n_fold']):True}
    file = os.path.join(pPath,file)

    df = fasta_to_df(file)
    
    if '/' in file:
        data_name = file.split('/')[-1].split('_')[0]
    else:
        data_name = file.split('_')[0]
    
    #print('data: {}'.format(data_name))
    #print('positive samples: ',len(df[df['Label']=='1']))
    #print('negative samples: ',len(df[df['Label']=='0']))
    #print("\n")
    
    encoding = train_params['encoding']
    k_max = train_params['encoding_params']['k_max']
    k_default = train_params['encoding_params']['k_default']

    results = {}
    
    if 'Kmer' in encoding:
        for k in range(1,k_max+1):
            train_params['encoding_params'].update({'kmer':k})

            key = '{}-{}'.format(encoding,k)
    
            try:
                X,y,groups = get_encodings(encoding, file, train_params['encoding_params'])

                kwargs = {
                    'model': train_params['model'],
                    'data_fold': train_params['data_fold'],
                    'n_fold': train_params['n_fold'],
                    'scoring': train_params['scoring'],
                    'groups': groups
                }

                #print(data_name,train_params['model'],train_params['data_fold'],train_params['n_fold'],key)

                scores, fold, clf = train_model(X,y,**kwargs)

                for score in scores.keys():
                    results.update({(data_name,
                                    train_params['model'],
                                    train_params['data_fold'],
                                    train_params['n_fold'],
                                    key,
                                    score):scores[score]})
            except Exception as e:
                print('error: {} in {}'.format(e,key))

                for score in train_params['scoring'].keys():
                    results.update({(data_name,
                                    train_params['model'],
                                    train_params['data_fold'],
                                    train_params['n_fold'],
                                    key,
                                    'test_'+score):'n/a'})


    else:
        train_params['encoding_params'].update({'kmer':k_default})

        key = encoding

        try:
            X,y,groups = get_encodings(encoding, file, train_params['encoding_params'])

            kwargs = {
                'model': train_params['model'],
                'data_fold': train_params['data_fold'],
                'n_fold': train_params['n_fold'],
                'scoring': train_params['scoring'],
                'groups': groups
            }

            #print(data_name,train_params['model'],train_params['data_fold'],train_params['n_fold'],key)
        
            scores, fold, clf = train_model(X,y,**kwargs)

            for score in scores.keys():
                results.update({(data_name,
                                train_params['model'],
                                train_params['data_fold'],
                                train_params['n_fold'],
                                key,
                                score):scores[score]})
        except Exception as e:
            print('error: {} in {}'.format(e,key))

            for score in train_params['scoring'].keys():
                results.update({(data_name,
                                train_params['model'],
                                train_params['data_fold'],
                                train_params['n_fold'],
                                key,
                                'test_'+score):'n/a'})

    return results

def worker_wrapper(arg):
    file, train_params = arg
    return main(file, **train_params)

