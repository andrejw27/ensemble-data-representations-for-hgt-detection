import argparse
import pandas as pd
from utils.ensemble_cross_val import wrapper_cross_val_all_pairs, cross_val_all_pairs
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import logging
from logging_config import setup_logging

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", type=str, default="RCKmer-7",
                        help="Representation to use for cross-validation", 
                        choices=['RCKmer-7', 'Z_curve_48bit', 'PCPseTNC','Kmer-6', 'PseEIIP'])
    parser.add_argument("--filename", type=str, default="benbow_data.fasta")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of splits for cross-validation")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="Number of repeats for cross-validation")
    parser.add_argument("--output_dir", type=str, default="ensemble_results",
                        help="Directory to save the results")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Define the representation pairs for cross-validation
    representation_pairs = {
        'RCKmer-7' : ['RCKmer-7_SVM/RCKmer-2_NaiveBayes',
                'RCKmer-7_SVM/RCKmer-1_SVM',
                'RCKmer-7_SVM/Kmer-1_SVM',
                'RCKmer-7_SVM/RCKmer-1_AdaBoost',
                'RCKmer-7_SVM/Z_curve_12bit_NaiveBayes',
                'RCKmer-7_SVM/PseEIIP_LogisticRegression'],

        'Z_curve_48bit' : ['Z_curve_48bit_RandomForest/RCKmer-2_NaiveBayes',
                        'Z_curve_48bit_RandomForest/RCKmer-1_SVM',
                        'Z_curve_48bit_RandomForest/RCKmer-1_AdaBoost',
                        'Z_curve_48bit_RandomForest/Kmer-6_LogisticRegression',
                        'Z_curve_48bit_RandomForest/Mismatch_XGBoost',
                        'Z_curve_48bit_RandomForest/RCKmer-5_NaiveBayes'],

        'PCPseTNC' : ['PCPseTNC_RandomForest/RCKmer-1_SVM',
                    'PCPseTNC_RandomForest/Kmer-1_SVM',
                    'PCPseTNC_RandomForest/RCKmer-1_AdaBoost',
                    'PCPseTNC_RandomForest/Z_curve_12bit_NaiveBayes',
                    'PCPseTNC_RandomForest/PseEIIP_LogisticRegression',
                    'PCPseTNC_RandomForest/Kmer-6_LogisticRegression'],

        'Kmer-6' : ['Kmer-6_SVM/RCKmer-1_SVM',
                    'Kmer-6_SVM/RCKmer-1_AdaBoost',
                    'Kmer-6_SVM/Z_curve_12bit_NaiveBayes',
                    'Kmer-6_SVM/PseEIIP_LogisticRegression',
                    'Kmer-6_SVM/Kmer-6_LogisticRegression',
                    'Kmer-6_SVM/Mismatch_XGBoost'],

        'PseEIIP' : ['PseEIIP_RandomForest/RCKmer-1_SVM',
                    'PseEIIP_RandomForest/Kmer-1_SVM',
                    'PseEIIP_RandomForest/RCKmer-1_AdaBoost',
                    'PseEIIP_RandomForest/PseEIIP_LogisticRegression',
                    'PseEIIP_RandomForest/Kmer-6_LogisticRegression',
                    'PseEIIP_RandomForest/Mismatch_XGBoost'
                    ]
    }

    # Parse command line arguments
    args = get_args()
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Print the input file and output directory
    print(f"Input file: {args.filename}")

    logger = logging.getLogger(__name__)

    log_path = os.path.join(pPath,'logs/cross_val/')

    os.makedirs(log_path, exist_ok=True)

    log_filename = os.path.join(log_path,'{}.log'.format(args.filename))
    setup_logging(log_filename=log_filename)

    # Ensure the representation is valid
    if args.representation not in ['RCKmer-7', 'Z_curve_48bit', 'PCPseTNC', 'Kmer-6', 'PseEIIP']:
        raise ValueError(f"Invalid representation: {args.representation}. "
                         "Choose from 'RCKmer-7', 'Z_curve_48bit', 'PCPseTNC', 'Kmer-6', 'PseEIIP'.")
    
    # Print the selected representation
    logger.info(f"Selected representation: {args.representation}")

    # Print the start of the cross-validation process
    logger.info(f"Starting cross-validation for {args.representation} with {args.n_splits} splits and {args.n_repeats} repeats.")
    # Filter representation pairs based on the selected representation
    representation_pairs = representation_pairs[args.representation]

    # Define the number of processes based on available CPU cores
    num_processes = multiprocessing.cpu_count()
    num_processes = min(num_processes, len(representation_pairs))
    filename = "dataset/train_data/" + args.filename

    cv_results = cross_val_all_pairs(representation_pairs, filename, args.n_splits, args.n_repeats)

    # Print the completion of the cross-validation process
    logger.info("Cross-validation completed.")

    # Save the final results to a excel file
    output_file = os.path.join(args.output_dir, f"{args.representation}_cross_val_results.xlsx")
    cv_results.to_excel(output_file, index=False)