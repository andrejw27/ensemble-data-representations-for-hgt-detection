#import libraries
import argparse
import pandas as pd
import os 
import json 
import json 
from utils.util import read_results,read_dataset, read_fold
from utils.ensemble_selection_util import get_pairwise_kappa, get_solutions, cross_validate_ensemble_candidates
import time

import logging
from logging_config import setup_logging

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="benbow")
    parser.add_argument("--same-clf", type=str2bool, default="False", 
                        help="If True, use the same classifier for all representations, otherwise use the best classifier for each representation")
    parser.add_argument("--error-metric", type=str, default="mcc", 
                        help="error metric to evaluate the multi-objective optimization problem, options: f1, mcc, accuracy, precision, recall")  
    parser.add_argument("--output-dir", type=str, default="outputs")  

    args = parser.parse_args()
    return args    

logger = logging.getLogger('ensemble_selection')
logger.info('--- Start ---')
args = get_args()
dataname = args.dataname
same_clf = args.same_clf

if same_clf:
    flag = 'homogeneous'
else:
    flag = 'heterogeneous'

log_dir = os.path.join(pPath,f'logs/ensemble_selection')

if not os.path.exists(log_dir):
    logger.info('Creating Logging Folder')
    os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir,f'{dataname}_{flag}.log')
setup_logging(log_filename=log_path)

if __name__ == "__main__":
    start_time = time.time()
    logger.info('--- Start ---')

    # read data for each fold
    args = get_args()
    dataname = args.dataname
    logger.info(f"Using same_clf: {same_clf}")

    logger.info(f"1. Read Data {dataname}")
    # Read training data
    filename = f"dataset/train_data/{dataname}.fasta"
    X, y, groups, genera, species = read_dataset(filename)

    logger.info("2. Read Pre-defined Folds")
    # Read fold
    fold_filename = f"dataset/folds/{dataname}_stratified_group_folds_new.json"
    labels = read_fold(fold_filename)

    # read predictions
    logger.info("3. Read predictions of the cross validation")
    predictions_dir = os.path.join(args.output_dir, "cross_val/predictions")

    #for f in os.listdir(predictions_dir):
    #    if f.endswith('.xlsx') and f.startswith(f'{dataname}') and not ('0' in f):
    #        selected_files.append(f)
    files = [f for f in os.listdir(predictions_dir) if os.path.isfile(os.path.join(predictions_dir, f))]

    predictions_df = pd.DataFrame()

    for file in files:
        if file.endswith('.xlsx') and file.startswith(f'{dataname}_predictions') and not ('0' in file):
            filename = os.path.join(predictions_dir,file)
            logger.info(f"Reading predictions from {filename}")
            df = read_results(filename, header=['fold','representation','model'])
            predictions_df = pd.concat([predictions_df,df])

    # Filter out models that are not of interest (e.g., ensemble methods)
    predictions_df = predictions_df[~predictions_df['model'].isin(['AdaBoost','GradientBoosting','XGBoost','Bagging'])]

    # Create a label column to identify each (feature_a, feature_b) pair
    predictions_df['pair_id'] = predictions_df['representation'].astype(str) + '/' + predictions_df['model'].astype(str)

    # Specify the error metric used to calculate the multi-objective optimization problem
    error_metric = args.error_metric 
 
    logger.info("4. Compute kappa error")
    # compute pairwise kappa and error rates
    error_df, pairwise_kappa_df, solutions_df = get_pairwise_kappa(labels, y, predictions_df, same_clf, error_metric=error_metric)
        

    m = 40
    n = 20

    if not same_clf:
        logger.info(f"5. Select top {n} base classifiers for ensemble classifier")
        pfront_ensemble_candidates, chull_ensemble_candidates, best_ensemble_candidates = get_solutions(solutions_df, m, n, error_metric)

        logger.info("6. Cross validate ensemble classifiers")
        # get ensemble candidates and create ensemble classifier based on the candidates
        candidates_dict = {
            'best': best_ensemble_candidates,
            'pfront': pfront_ensemble_candidates,
            'chull': chull_ensemble_candidates
        }
        ensemble_final_results, final_results_df = cross_validate_ensemble_candidates(candidates_dict, labels, y, predictions_df)
    else:
        all_clfs = solutions_df.clf.unique()
        ensemble_final_results = {}
        final_results_df = pd.DataFrame()
        for clf in all_clfs:
            logger.info(f"5. Select top {n} representations for ensemble classifier with {clf}")
            pfront_ensemble_candidates, chull_ensemble_candidates, best_ensemble_candidates = get_solutions(solutions_df[solutions_df['clf']==clf], m, n, error_metric)

            logger.info("6. Cross validate ensemble classifiers")
            # get ensemble candidates and create ensemble classifier based on the candidates
            candidates_dict = {
                'best': best_ensemble_candidates,
                'pfront': pfront_ensemble_candidates,
                'chull': chull_ensemble_candidates
            }
            ensemble_final_results_temp, final_results_df_temp = cross_validate_ensemble_candidates(candidates_dict, labels, y, predictions_df)
            
            final_results_df_temp['clf'] = clf
            final_results_df = pd.concat([final_results_df, final_results_df_temp])

            ensemble_final_results[clf] = ensemble_final_results_temp
    
    output_dir = args.output_dir 
    kappa_solution_dir = f'{output_dir}/multi_objective_solutions'
    os.makedirs(kappa_solution_dir, exist_ok=True)

    ensemble_validation_dir = f'{output_dir}/multi_objective_validation'
    os.makedirs(ensemble_validation_dir, exist_ok=True)


    # assign dataname to solutions_df and store the multi-objective optimization solutions to excel file
    solutions_df['dataname'] = dataname
    solutions_df.to_excel(f'{kappa_solution_dir}/{dataname}_multi_objective_solutions_{flag}.xlsx',index=False)

    # store the cross-validation results on the ensemble candidates
    output_path = os.path.join(ensemble_validation_dir,f'ensemble_selection_{dataname}_{flag}_{error_metric}.xlsx')
    output_path_json = os.path.join(ensemble_validation_dir,f'ensemble_selection_{dataname}_{flag}_{error_metric}.json')
    logger.info(f"7. Saving ensemble selection results to {output_path}")
    final_results_df.to_excel(output_path, index=False)

    # Save nested dictionary to JSON file
    with open(output_path_json, "w") as json_file:
        json.dump(ensemble_final_results, json_file, indent=4)  # indent for pretty logger.infoing

    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))




