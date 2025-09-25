import time
import argparse
import itertools
import os, sys
import json 
import numpy as np
import pandas as pd 
import subprocess
import datetime
from concurrent.futures import ThreadPoolExecutor

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from utils.Evaluations import Evaluations
from utils.util import get_organism_info

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-process", type=int, default=4)
    parser.add_argument("--result-type", type=str, default='test')
    parser.add_argument("--genome-folder", type=str, default='dataset/genomes/benbow_set')
    parser.add_argument("--predictions-folder", type=str, default='outputs/boundaries_predictions') 
    args = parser.parse_args()
    return args

def run(args):
        cmd = ['python', 'predictGI.py'] + args
        result = subprocess.run(cmd, capture_output=True)
        print(f"Command: {cmd}\nReturn code: {result.returncode}\nOutput:\n{result.stdout}\n---")
        return result


def evaluate():
    
    args = get_args()
    result_type = args.result_type #test or literature evaluation
    predictions_folder = args.predictions_folder
    model_dict = {}

    #check files in a folder
    for file in os.listdir():
        if file.endswith('.xlsx'):
            predictor = file.split('.xlsx')[0]
            predictor_df = pd.read_excel(os.path.join(predictions_folder,file))
            model_dict.update({predictor:get_organism_info(predictor_df)}) 

    #read ground truth data (GI_literature_set_table, GI_negative_set_table, positive_test_table_gc, negative_test_table_gc)
    if result_type == 'literature':
        pos_table = pd.read_excel("outputs/literature_reference/GI_literature_set_table.xlsx")
        neg_table = pd.read_excel("outputs/literature_reference/GI_negative_set_table.xlsx")
    elif result_type == 'test':
        pos_table = pd.read_excel("outputs/literature_reference/positive_test_table_gc.xlsx")
        neg_table = pd.read_excel("outputs/literature_reference/negative_test_table_gc.xlsx")

    organism_pos_test_dict = get_organism_info(pos_table)
    organism_neg_test_dict = get_organism_info(neg_table)

    total_orgs = organism_pos_test_dict.keys()

    eval = Evaluations()

    print("evaluation of {} data".format(result_type))
    eval_results = eval.evaluations_main_104(total_orgs, 
                                            model_dict, 
                                            organism_pos_test_dict, 
                                            organism_neg_test_dict, 
                                            result_type, 
                                            False)


    def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
                
    eval_dest = f"{predictions_folder}/evaluation/"
    os.makedirs(eval_dest, exist_ok=True)
    json_obj = json.dumps(eval_results, indent=1, default=myconverter)
    json_file = f"{eval_dest}/evaluation_result_{result_type}_gridsearch_latest.json"

    with open(json_file, 'w') as file:
        json.dump(json_obj, file, indent=4)

def main():
    """
    Run the following command to run boundaries prediction task with different parameters
    """ 
    args = get_args()
    params = ['--genomes-path', '--output-dest', '--model', '--window-size', '--upper-threshold']
    genomes_path = [args.genome_folder] 
    output_dest = [args.predictions_folder]
    models = ["ensemble_stacking_benbow.pkl",
                "ensemble_voting_soft_benbow.pkl",
                "RCKmer-7_SVM_benbow.pkl"]
    window_sizes = list(range(5000, 16000,1000))
    upper_ths = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    # for quick testing
    #window_sizes = [10000]
    #upper_ths = [0.8]

    combinations = list(itertools.product(genomes_path, output_dest, models, window_sizes, upper_ths))

    list_params = []
    for value in combinations:
        n_args = len(params)
        args_line = []
        for i in range(n_args):
            args_line += [params[i]] + [str(value[i])]
        list_params.append(args_line) 

   
    n_process = args.n_process
    # Set max_workers to control number of parallel executions (e.g., 4 at once)
    with ThreadPoolExecutor(max_workers=n_process) as executor:
        executor.map(run, list_params)


if __name__=="__main__":
    start_time = time.time()
    print('--- Start ---')
    main()
    print('--- Start evaluating boundaries predictions ---')
    evaluate()
    finish_time = time.time()
    print('--- Finish ---')
    print(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))