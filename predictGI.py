import time
from tqdm import tqdm
import argparse
import os
import pandas as pd 
from tqdm import tqdm
from utils.Predictor import Predictor

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from logging_config import setup_logging
import logging

logger = logging.getLogger("predict_GI")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomes-path", type=str, default="dataset/genomes/benbow_test")
    parser.add_argument("--model", type=str, default="fine_tuned_model.pkl" )
    parser.add_argument("--output-dest", type=str, default="outputs/boundaries_predictions")
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--upper-threshold", type=float, default=0.8)
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    model = args.model.split('.')[0]

    log_dir = os.path.join(pPath,f'logs/predict_GI/{model}')

    #if not os.path.exists(log_path):
    logger.info('Creating Logging Folder')
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir,f'{model}.log')
    setup_logging(log_filename=log_path)

    #folder containing genomes of interest 
    # ("dataset/genomes/benbow_test","dataset/genomes/literature" )
    genome_path = args.genomes_path
    files = [f for f in os.listdir(genome_path) if os.path.isfile(os.path.join(genome_path, f))]

    # path to load the saved model: utils/models/
    model = args.model 

    # path to save the predictions
    output_path = args.output_dest
    output_path = os.path.join(output_path, genome_path.split('/')[-1], model.split('.')[0],
                               str(args.window_size),str(args.upper_threshold))

    logger.info(output_path)

    os.makedirs(output_path, exist_ok=True)

    #run the following code only if the predictions do not exist for the specified genomes and model
    for file in tqdm(files):
        #accept fasta file only
        if file.endswith('.fasta'):
            filename = os.path.join(genome_path,file)

            #create a folder for each genome
            output_dest = os.path.join(output_path, file.split('.fasta')[0])
            if not os.path.exists(output_dest):
                os.mkdir(output_dest)

                #initialize the predicor
                seq = Predictor(filename, output_file_path=output_dest, model_file=model)

                seq.parameters.set_window_size(args.window_size)
                seq.parameters.set_upper_threshold(args.upper_threshold)
                seq.parameters.set_minimum_gi_size(args.window_size)

                #run the predictor
                pred = seq.predict()
                
                #save predictions to excel file
                seq.predictions_to_excel(pred)
            else:
                logger.info("Predictions for {} already exist".format(file))
        else:
            logger.info("The code only accepts fasta file!")

    #read predictions for each genome, then combine them into a file
    results = pd.DataFrame()

    results_dest = args.output_dest
    os.makedirs(results_dest, exist_ok=True)

    for dir in os.listdir(output_path):
        child_dirs = os.path.join(output_path,dir)
        if os.path.isdir(child_dirs):
            for file in os.listdir(child_dirs):
                #if 'out' not in file:
                    res = pd.read_excel(os.path.join(child_dirs,file))
                    res = res.drop(res.columns[0], axis=1)
                    res = res.assign(Genome=dir)
                    results = pd.concat([results,res])

    results = results.rename(columns={'accession':'Accession','start':'Start','end':'End'})
    results = results[results['probability']>0.5]
    results['Accession'] = results.apply(lambda x: x['Accession'].split('|')[0],axis=1)
    results_file = os.path.join(results_dest, model.split('.')[0]+f'_{args.window_size}_{args.upper_threshold}.xlsx')
    results.to_excel(results_file, index=False)
    logger.info(results_dest)

if __name__=="__main__":
    start_time = time.time()
    logger.info('--- Start ---')
    main()
    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))