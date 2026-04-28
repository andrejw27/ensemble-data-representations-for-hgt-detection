#import libraries
import argparse
import os 
import numpy as np 
import pandas as pd
from utils.util import read_results, read_dataset, read_fold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import time 

import logging
from logging_config import setup_logging

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="benbow") 
    parser.add_argument("--output-dir", type=str, default="outputs/cross_val") 
    #parser.add_argument("--predictions-dir", type=str, default="outputs/cross_val")
    args = parser.parse_args()
    return args

logger = logging.getLogger('evaluate_cross_val')
logger.info('--- Start ---')
args = get_args()
dataname = args.dataname

log_dir = os.path.join(pPath,f'logs/evaluate_cross_val')

logger.info('Creating Logging Folder')
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir,f'{dataname}.log')
setup_logging(log_filename=log_path)

def entropy_binary(p0_list):
    entropies = []
    for p0 in p0_list:
        p1 = 1 - p0
        probs = np.array([p0, p1])
        probs = probs[probs > 0]  # avoid log(0)
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(entropy)
    return entropies

def compute_metrics(predictions_df, labels, y):
    """
    Compute evaluation metrics for each fold and return a DataFrame.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions.
        labels (dict): Dictionary containing train/validation/test indices for each fold.
        y (np.array): Ground truth labels.
        
    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each fold.
    """
    eval_metrics = []

    for fold in labels.keys():
        fold_predictions_df = predictions_df[predictions_df['fold']==fold]
        fold_predictions_df = fold_predictions_df.dropna(axis=1)
        pred_cols = [col for col in fold_predictions_df.columns if col.startswith('y_')]
        df_preds = fold_predictions_df.set_index('pair_id')[pred_cols]

        test_indices = labels[fold]['valid_idx'] + labels[fold]['test_idx']
        ground_truth = y[test_indices]

        for pair_id, row in df_preds.iterrows():
            pred_prob = row.values
            pred_prob = pred_prob[:len(test_indices)]
            # Apply custom threshold
            threshold = 0.5
            preds = ((1-pred_prob) >= threshold).astype(int)
            mcc = matthews_corrcoef(ground_truth, preds)
            f1 = f1_score(ground_truth, preds)
            accuracy = accuracy_score(ground_truth, preds),
            precision = precision_score(ground_truth, preds, zero_division=0),
            recall = recall_score(ground_truth, preds, zero_division=0),
            true_positive = np.sum((ground_truth == 1) & (preds == 1))
            true_negative = np.sum((ground_truth == 0) & (preds == 0))
            false_positive = np.sum((ground_truth == 0) & (preds == 1))
            false_negative = np.sum((ground_truth == 1) & (preds == 0))

            # Append metrics for the current fold and pair_id
            eval_metrics.append({
                'fold': fold,
                'pair_id': pair_id,
                'mcc': mcc,
                'f1': f1,
                'precision': precision[0],
                'recall': recall[0],
                'accuracy': accuracy[0],
                'true_positive': true_positive,
                'true_negative': true_negative,
                'false_positive': false_positive,
                'false_negative': false_negative
            })
    
    return pd.DataFrame(eval_metrics)

def get_top_n_models(eval_metrics, n=5):
    """
    Get the top n models based on F1 score.
    
    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics.
        n (int): Number of top models to return.
        
    Returns:
        pd.DataFrame: DataFrame containing the top n models.
    """
    avg_eval_metrics = eval_metrics.groupby(['pair_id']).agg({'mcc':'mean', 'f1':'mean','precision':'mean','recall':'mean','accuracy':'mean'}).reset_index().sort_values(by=['f1'], ascending=False)
    avg_eval_metrics['representation'] = avg_eval_metrics['pair_id'].apply(lambda x: x.split('/')[0])
    avg_eval_metrics['model'] = avg_eval_metrics['pair_id'].apply(lambda x: x.split('/')[1])

    #rank the RCKmer and Kmer based on F1 score per dataset
    kmer_family = avg_eval_metrics[(avg_eval_metrics.representation.str.startswith('Kmer'))].copy()
    kmer_family['group_rank'] = kmer_family['f1'].rank(method="first", ascending=False)

    rckmer_family = avg_eval_metrics[(avg_eval_metrics.representation.str.startswith('RCKmer'))].copy()
    rckmer_family['group_rank'] = rckmer_family['f1'].rank(method="first", ascending=False)

    kmer_table = pd.concat([kmer_family,rckmer_family])

    #assign the rank of RCKmer and Kmer to the original table
    df_merged = pd.merge(avg_eval_metrics, kmer_table[['representation','group_rank']], on=['representation'], how='left')
    df_merged = df_merged.fillna(0.0)

    #select the best RCKmer and Kmer only
    sub_final_df = df_merged[df_merged['group_rank']<=1.0].copy()
    #select best model per representation
    sub_final_df['rank'] = sub_final_df.groupby(['representation'])['f1'].rank(method="first", ascending=False)
    sub_final_df = sub_final_df[sub_final_df['rank']==1]


    return sub_final_df.sort_values(by='f1', ascending=False).head(n)
    

if __name__ == "__main__":
# required files: 
# FASTA file for training data
# pre-defined folds in JSON format
# predictions in Excel format

    start_time = time.time()
    logger.info('--- Start ---')

    # read data for each fold
    args = get_args()
    dataname = args.dataname

    logger.info(f"1. Read Data {dataname}")
    # Read training data
    train_filename = f"dataset/train_data/{dataname}.fasta"
    X_train, y_train, groups_train, genera_train, species_train = read_dataset(train_filename)

    logger.info("2. Read Pre-defined Folds")
    # Read fold
    fold_filename = f"dataset/folds/{dataname}_stratified_group_folds_new.json"
    labels = read_fold(fold_filename)

    # read predictions
    logger.info("3. Read predictions of the cross-validation")
    predictions_path = os.path.join(args.output_dir,"predictions")
    files = [f for f in os.listdir(predictions_path) if os.path.isfile(os.path.join(predictions_path, f))]

    predictions_df = pd.DataFrame()

    for file in files:
        if file.endswith('.xlsx') and file.startswith(f'{dataname}_predictions') and not ('0' in file):
            logger.info(f"Reading predictions from {file}")
            filename = os.path.join(predictions_path,file)
            df = read_results(filename, header=['fold','representation','model'])
            predictions_df = pd.concat([predictions_df,df])

    ## Filter out models that are not of interest (e.g., ensemble methods)
    #predictions_df = predictions_df[~predictions_df['model'].isin(['AdaBoost','GradientBoosting','XGBoost','Bagging'])]

    # Create a label column to identify each (feature_a, feature_b) pair
    predictions_df['pair_id'] = predictions_df['representation'].astype(str) + '/' + predictions_df['model'].astype(str)

    logger.info(f"4. Evaluate cross-validation results for {dataname} dataset")
    eval_metrics = compute_metrics(predictions_df, labels, y_train)
    eval_metrics['representation'] = eval_metrics['pair_id'].apply(lambda x: x.split('/')[0])
    eval_metrics['model'] = eval_metrics['pair_id'].apply(lambda x: x.split('/')[1])

    cv_eval_dir = os.path.join(args.output_dir,"evaluation")
    os.makedirs(cv_eval_dir, exist_ok=True)
    output_path = os.path.join(cv_eval_dir,f'{dataname}_crossval.xlsx')
    logger.info(f"5. Saving cross-validation results to {output_path}")
    eval_metrics.to_excel(output_path, index=False)

    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))






