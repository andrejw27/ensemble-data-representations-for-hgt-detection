import argparse
import pickle 
import json 
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer,matthews_corrcoef, fbeta_score
from utils.train import true_positive,true_negative,false_positive,false_negative
from utils.ensemble_cross_val import RepeatedStratifiedGroupKFold


import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="benbow")
    parser.add_argument("--multi_objective_solution", type=str, default="pfront")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=5)

    args = parser.parse_args()
    return args

def make_selector(start_idx, end_idx):
    return FunctionTransformer(lambda X: X[:, start_idx:end_idx])

def get_ensemble_classifier(ensemble_candidates, data, ensemble_type='stacking'):
    base_estimators = []
    feature_pointer = 0

    #define the models to be trained
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'NaiveBayes': GaussianNB(),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'Bagging':BaggingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }

    # create the ensemble
    meta_models = {
        "stacking": StackingClassifier(estimators=None, final_estimator=LogisticRegression(), cv=5),
        "voting_soft": VotingClassifier(estimators=None, voting="soft"),
        "voting_hard": VotingClassifier(estimators=None, voting="hard")
    }

    for candidate in ensemble_candidates:
        representation = candidate.split('/')[0]
        model = models[candidate.split('/')[1]]
        n_features = data[representation].shape[-1]
        
        pipeline = Pipeline([
                        ('selector', make_selector(feature_pointer, feature_pointer+n_features)),   
                        ('clf', model)
                    ])
        
        #update feature_pointer
        feature_pointer += n_features

        #append pipeline to base_estimatores
        base_estimators.append((representation, pipeline))

    eclf = meta_models[ensemble_type]
    eclf.estimators = base_estimators
    
    return eclf 

# evaluate a list of models
def evaluate_ensemble(candidates, data, n_splits, n_repeats, ensemble_type='stacking', **kwargs):
    # check for no models
    if len(candidates) == 0:
        return 0.0
    
    # prepare X_train, X_test
    X = [data[candidate.split('/')[0]] for candidate in candidates]
    #stack X to fit the input format of ensemble classifier
    X_stack = np.hstack(X)
    y = data['y']
    groups = data['groups']

    # Define scoring metrics
    scoring = {
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F_1': 'f1',  # F-beta with beta=1 is equivalent to F1-score
        'F_beta_0.5': make_scorer(fbeta_score, beta=0.5),
        'F_beta_2': make_scorer(fbeta_score, beta=2),
        'MCC': make_scorer(matthews_corrcoef),
        'TP': make_scorer(true_positive),
        'TN': make_scorer(true_negative),
        'FP': make_scorer(false_positive),
        'FN': make_scorer(false_negative),
    }

    if 'folds' in kwargs.keys():
        folds = kwargs['folds']
        n_repeats = len(folds)
        n_folds = len(folds[0]["folds"])

        # create cv fold (train_idx, test_idx)
        cv = []

        for repeat in range(n_repeats):
            for fold in range(n_folds):
                train_idx = folds[repeat]["folds"][fold]["train_idx"]
                test_idx = folds[repeat]["folds"][fold]["test_idx"]
                cv.append((train_idx, test_idx))
    else:
        # define the evaluation procedure
        cv = RepeatedStratifiedGroupKFold(n_splits=n_splits, n_repeats=n_repeats)

    # ensemble_candidates = pfront_candidates or chull_candidates
    eclf = get_ensemble_classifier(candidates, data, ensemble_type)

    # evaluate the ensemble
    scores = cross_validate(eclf, X_stack, y, groups=groups, cv=cv, scoring=scoring, n_jobs=5)
    
    # return mean score
    return scores

# perform a single round of growing the ensemble
def grow_round(models_in, models_candidate, data, n_splits=5, n_repeats=5, ensemble_type='stacking', metrics='MCC',**kwargs):
	# establish a baseline
    baseline = evaluate_ensemble(models_in, data, n_splits, n_repeats, ensemble_type, **kwargs)
    # compute baseline mean score
    if type(baseline)==dict:
        mean_baseline = np.mean(baseline[f'test_{metrics}'])
    else:
        mean_baseline = 0.0

    best_score, addition, best_result = mean_baseline, None, baseline
    # enumerate adding each candidate and see if we can improve performance
    for m in models_candidate:
        # copy the list of chosen models
        dup = models_in.copy()
        # add the candidate
        dup.append(m)
        # evaluate new ensemble
        result = evaluate_ensemble(dup, data, n_splits, n_repeats, ensemble_type, **kwargs)
        # mean result
        mean_result = np.mean(result[f'test_{metrics}'])
        # check for new best
        if mean_result > best_score:
            # store the new best
            best_score, addition, best_result = mean_result, m, result
    
    return best_score, addition, best_result
 
# grow an ensemble from scratch
def grow_ensemble(models, data, n_splits, n_repeats, ensemble_type='stacking', metrics='MCC',**kwargs):
    best_score, best_list, best_all_scores = 0.0, list(), list()
	
    # grow ensemble until no further improvement
    while True:
        # add one model to the ensemble
        score, addition, all_scores = grow_round(best_list, models, data, n_splits=n_splits, n_repeats=n_repeats, 
                                                 ensemble_type=ensemble_type, metrics=metrics,**kwargs)
        # check for no improvement
        if addition is None:
            print('>no further improvement')
            break
        # keep track of best score
        best_score = score
        # remove new model from the list of candidates
        models.remove(addition)
        # add new model to the list of models in the ensemble
        best_list.append(addition)
        # add scores to the list of scores in the ensemble
        best_all_scores.append(all_scores)
        # report results along the way
        names = ','.join([n for n in best_list])
        print('>%.3f (%s)' % (score, names))
    return best_score, best_list, best_all_scores

if __name__ == "__main__":
    args = get_args()
    filename = args.filename
    print(f"Processing file: {filename}")

    print("Read Data")
    #read pickle file
    with open(f"dataset/encoded_data/{filename}_transformed.pkl", "rb") as f:
        data = pickle.load(f)

    print("Read Pre-defined Folds")
    # Load folds
    with open(f"dataset/{filename}_stratified_group_folds.json", "r") as f:
        folds = json.load(f)

    print("Get ensemble candidates from multi-objective optimization")
    # get ensemble candidates and create ensemble classifier based on the candidates
    # list of ensemble candidates
    pfront_solutions = [('RCKmer-7/SVM', 18),
                        ('Subsequence/RandomForest', 10),
                        ('RCKmer-6/SVM', 8),
                        ('Kmer-6/SVM', 6),
                        ('RCKmer-5/SVM', 5),
                        ('RCKmer-3/RandomForest', 5),
                        ('Mismatch/RandomForest', 5),
                        ('RCKmer-4/RandomForest', 3),
                        ('RCKmer-1/SVM', 3),
                        ('Z_curve_9bit/RandomForest', 2)]

    chull_solutions = [('RCKmer-7/SVM', 18),
                        ('Subsequence/RandomForest', 12),
                        ('RCKmer-6/SVM', 8),
                        ('Kmer-6/SVM', 7),
                        ('RCKmer-3/RandomForest', 6),
                        ('Mismatch/RandomForest', 6),
                        ('RCKmer-5/SVM', 5),
                        ('RCKmer-4/RandomForest', 4),
                        ('Kmer-1/RandomForest', 3),
                        ('RCKmer-1/SVM', 3)]
    
    best_clf_candidates = ['RCKmer-7/SVM', 'RCKmer-6/SVM', 'Kmer-6/SVM', 'Kmer-7/SVM',
                            'RCKmer-5/SVM', 'Kmer-5/SVM', 'RCKmer-5/RandomForest',
                            'RCKmer-4/RandomForest', 'Kmer-4/SVM', 'RCKmer-6/RandomForest',
                            'Z_curve_48bit/SVM', 'RCKmer-3/RandomForest', 'RCKmer-4/SVM',
                            'Kmer-4/RandomForest', 'Subsequence/RandomForest']
    
    # filtered_best_clf_candidates = ['RCKmer-7/SVM','Kmer-7/SVM','Z_curve_48bit/SVM','PseKNC/RandomForest','Z_curve_144bit/SVM',
    #                                 'PseEIIP/RandomForest','SCPseTNC/RandomForest','TPCP/RandomForest','PCPseTNC/RandomForest',
    #                                 'Subsequence/RandomForest','MMI/RandomForest','SCPseDNC/RandomForest','CKSNAP/RandomForest',
    #                                 'Moran/RandomForest','Geary/RandomForest']
    
    filtered_best_clf_candidates = ['Z_curve_48bit/SVM','PseKNC/RandomForest','Z_curve_144bit/SVM',
                                    'PseEIIP/RandomForest','SCPseTNC/RandomForest']

    pfront_candidates = [pair for pair, _ in pfront_solutions]
    chull_candidates = [pair for pair, _ in chull_solutions]

    multi_objective_solution = args.multi_objective_solution
    candidates_dict = {'pfront': pfront_candidates,
                  'chull': chull_candidates,
                  'best': best_clf_candidates,
                  'filtered_best': filtered_best_clf_candidates}
    candidates = candidates_dict[multi_objective_solution]

    n_splits = args.n_splits
    n_repeats = args.n_repeats
    ensemble_type = 'stacking'
    metrics = 'MCC'
    kwargs = {'folds':folds}

    print(f'Ensemble growing with {multi_objective_solution} candidates')
    best_score, best_list, best_all_scores = grow_ensemble(candidates, data, n_splits, n_repeats, 
                                                           ensemble_type=ensemble_type, metrics=metrics,**kwargs)
    
    cv_results_dict = {}
    for i, model in enumerate(best_list):
        cv_results_dict[model] = best_all_scores[i]

    cv_results = pd.DataFrame.from_dict(cv_results_dict, orient='index')
    cv_results = cv_results.reset_index()

    output_dir = 'outputs/ensemble_grow'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir,f'{multi_objective_solution}_ensemble_grow_{filename}.xlsx')
    cv_results.to_excel(output_file, index=False)

    print('Ensemble growing is done')




