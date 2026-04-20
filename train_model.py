import argparse
from Bio import SeqIO
import numpy as np
import pickle 
import os
from utils.ensemble_cross_val import get_ensemble_model, compute_eval_score
import time

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from logging_config import setup_logging
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="RCKmer-7/SVM,Subsequence/RandomForest", 
                        help="a list of candidates for the ensemble classifier split by ',' e.g. RCKmer-7/SVM,Subsequence/RandomForest,...")  # one or more values
    parser.add_argument("--ensemble-type", type=str, default="stacking",
                        help="type of ensemble classifier: stacking, voting_soft, or voting_hard")
    parser.add_argument("--save-model", type=bool, default=False)
    parser.add_argument("--model-path", type=str, default="utils/models")
    parser.add_argument("--train-path", type=str, default="dataset/train_data/benbow.fasta")
    parser.add_argument("--test-path", type=str, default="dataset/test_data/benbow_test_data.fasta")
    args = parser.parse_args()
    return args

logger = logging.getLogger('train_model')
logger.info('--- Start ---')
args = get_args()
train_filename = args.train_path
dataname = train_filename.split('/')[-1].split('.')[0]
log_dir = os.path.join(pPath,'logs/train_model')

if not os.path.exists(log_dir):
    logger.info('Creating Logging Folder')
    os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir,f'{dataname}.log')
setup_logging(log_filename=log_path)

def main():

    args = get_args()

    #read train data
    train_filename = args.train_path
    logger.info(f"Read train data from {train_filename}")
    dataname = train_filename.split('/')[-1].split('.')[0]
    X = []
    y = []
    groups = []
    for record in SeqIO.parse(train_filename, "fasta"):
        X.append(str(record.seq))
        y.append(record.id.split('|')[1])
        groups.append(record.id.split('|')[0])

    X_train = np.array(X)
    y_train = np.array(y).astype(int)
    groups_train = np.array(groups)

    #read test data
    test_filename = args.test_path
    logger.info(f"Read train data from {test_filename}")
    X_test = []
    y_test = []
    groups_test = []
    genera_test = []
    species_test = []
    desc_test = []
    for record in SeqIO.parse(test_filename, "fasta"):
        X_test.append(str(record.seq))
        y_test.append(record.id.split('|')[1])
        groups_test.append(record.id.split('|')[0])
        genera_test.append(record.description.split(',')[0].split(' ')[2])
        species_test.append(' '.join(record.description.split(',')[0].split(' ')[2:]))
        desc_test.append('_'.join(record.id.split('|')[0].split('_')[0:2]) + ': ' + ' '.join(record.description.split(',')[0].split(' ')[1:]))


    X_test = np.array(X_test).astype(str)
    y_test = np.array(y_test).astype(int)
    groups_test = np.array(groups_test)
    genera_test = np.array(genera_test)
    species_test = np.array(species_test)
    desc_test = np.array(desc_test)

    groups_train = np.array([group.split('.')[0] for group in groups_train])
    groups_test = np.array([group.split('.')[0] for group in groups_test])

    # ensure training data does not contain test data
    test_ids = set(groups_test)

    selected_ids = []
    for i, g in enumerate(groups_train):
        if g not in test_ids:
            selected_ids.append(i)

    reduced_X_train = X_train[selected_ids]
    reduced_y_train = y_train[selected_ids]
    reduced_group_train = groups_train[selected_ids]

    logger.info('X_train: ', reduced_X_train.shape[0])
    logger.info('X_test: ', X_test.shape[0])

    #ensure species in train and test data sets are mutually exclusive
    assert len(set(groups_test)-set(reduced_group_train)) == len(set(groups_test))

    save_model = args.save_model
    #candidates = ['RCKmer-7/SVM', 'Subsequence/RandomForest']
    candidates = args.candidates.split(',')

    ensemble_type = args.ensemble_type
    ensemble = get_ensemble_model(candidates, ensemble_type)

    logger.info("Start training ensemble model...")
    ensemble.fit(reduced_X_train, reduced_y_train)
    logger.info("Training is done")

    logger.info("Select the best single model")
    single_best_scores = []
    #evaluate on test data and find the best single model
    for candidate in candidates:
        single_model = ensemble.named_estimators_[candidate]
        y_pred = single_model.predict(X_test)
        eval_score = compute_eval_score(y_test, y_pred)
        single_best_scores.append(eval_score['F_1'])
        logger.info(f"Candidate: {candidate}, F1: {eval_score['F_1']:.3f}, Precision: {eval_score['Precision']:.3f}, Recall: {eval_score['Recall']:.3f}, MCC: {eval_score['MCC']:.3f}")

    best_single_score = np.argmax(single_best_scores)
    best_single_candidate = candidates[best_single_score]
    best_single_model = ensemble.named_estimators_[best_single_candidate]

    logger.info(f"Best single model: {best_single_candidate}")

    if save_model:
        model_path = args.model_path
        #save ensemble model
        with open(os.path.join(model_path,f'ensemble_{ensemble_type}_{dataname}.pkl'),'wb') as f:
            pickle.dump(ensemble,f)
        #save single model
        with open(os.path.join(model_path,f'single_{dataname}.pkl'),'wb') as f:
            pickle.dump(best_single_model,f)

if __name__=="__main__":
    start_time = time.time()
    logger.info('--- Start training---')
    main()
    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))
