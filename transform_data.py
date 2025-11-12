import argparse
import pandas as pd
import numpy as np
import pickle
from Bio import SeqIO
from utils.util import get_customer_transformer
from multiprocessing import Pool
from functools import partial
import numpy as np
from tqdm import tqdm
import logging
from logging_config import setup_logging

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="benbow")

    args = parser.parse_args()
    return args

# transform data into different representations and parallelize the process
def transform_sequence(representation, X):
    transformer = get_customer_transformer(representation)
    return (representation, transformer.fit_transform(X))

if __name__ == "__main__":
    # representation to use
    representations = ['NAC','CKSNAP','ASDC','GC', 'Subsequence' , 'Mismatch',
                    'Z_curve_9bit', 'Z_curve_12bit','Z_curve_36bit','Z_curve_48bit','Z_curve_144bit',
                    'PseEIIP','DAC','DCC','DACC','TAC','TCC','TACC', 'Moran','Geary','NMBroto','DPCP','TPCP',
                    'MMI','PseDNC','PseKNC','PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC',
                    'RCKmer-1', 'RCKmer-2', 'RCKmer-3', 'RCKmer-4', 'RCKmer-5', 'RCKmer-6', 'RCKmer-7',
                    'Kmer-1', 'Kmer-2', 'Kmer-3', 'Kmer-4', 'Kmer-5', 'Kmer-6', 'Kmer-7'] 

    args = get_args()
    filename = args.filename
    print(f"Processing file: {filename}")

    # Read raw data 
    file_path = f"dataset/train_data/{filename}.fasta"
    X = []
    y = []
    groups = []
    for record in SeqIO.parse(file_path, "fasta"):
            X.append(str(record.seq))
            y.append(record.id.split('|')[1])
            groups.append(record.id.split('|')[0])

    X = np.array(X).astype(str)
    y = np.array(y).astype(int)
    groups = np.array(groups)

    # transformed_data = {}
    # for representation in tqdm(representations):
    #      transformed_X = transform_sequence(representation, X)
    #      transformed_data[representation] = transformed_X

    # transformed_data['y'] = y
    # transformed_data['groups'] = groups

    with Pool(4) as pool:
        results = list(tqdm(pool.imap(partial(transform_sequence, X=X), representations), 
                            total=len(representations), 
                            desc="Transforming sequences"))
        #results = pool.map(partial(transform_sequence, X=X), representations)

    # Save results into dictionary of {representation: transformed_x} and save it to .pkl
    transformed_data = {}
    for representation, transformed_x in results:
        transformed_data[representation] = transformed_x
    transformed_data['y'] = y
    transformed_data['groups'] = groups
    
    
    # Save to file
    output_dir = "dataset/encoded_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving transformed data to {output_dir}")
    # Create output file name
    output_file = f"{output_dir}/{filename}_transformed.pkl"
    # Save dictionary to pickle
    with open(output_file, 'wb') as f:
        pickle.dump(transformed_data, f)



