import numpy as np 
import pandas as pd
import os  
from itertools import combinations
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
from utils.util import kappa, compute_pfront_chull
from scipy.stats import mode

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

import logging 
logger = logging.getLogger('ensemble_selection')


def get_pairwise_kappa(labels, y, predictions_df, same_clf=False, error_metric='f1'):
    # compute error
    error_data = []
    pairwise_kappa = []

    for fold in labels.keys():
        fold_predictions_df = predictions_df[predictions_df['fold']==fold]
        fold_predictions_df = fold_predictions_df.dropna(axis=1)
        pred_cols = [col for col in fold_predictions_df.columns if col.startswith('y_')]
        df_preds = fold_predictions_df.set_index('pair_id')[pred_cols]

        test_indices = labels[fold]['valid_idx']
        ground_truth = y[test_indices]

        for pair_id, row in df_preds.iterrows():
            pred_prob = row.values
            pred_prob = pred_prob[:len(test_indices)]
            # Apply custom threshold
            threshold = 0.5
            preds = ((1-pred_prob) >= threshold).astype(int)
            mcc = 1 - matthews_corrcoef(ground_truth, preds)
            f1 = 1 - f1_score(ground_truth, preds)
            accuracy = 1 - accuracy_score(ground_truth, preds),
            precision = 1 - precision_score(ground_truth, preds, zero_division=0),
            recall = 1 - recall_score(ground_truth, preds, zero_division=0),

            if 'ground_truth' not in pair_id:
                error_data.append({
                    'fold': fold,
                    'pair_id': pair_id,
                    'mcc_error_rate': mcc,
                    'f1_error_rate': f1,
                    'accuracy_error_rate': accuracy,
                    'precision_error_rate': precision,
                    'recall_error_rate': recall
                })

        pair_ids = predictions_df['pair_id'].unique()
        unique_pairs = list(combinations(pair_ids, 2))

        for (pair1, pair2) in unique_pairs:
            row1 = df_preds.loc[pair1].values[:len(test_indices)]
            row2 = df_preds.loc[pair2].values[:len(test_indices)]
            threshold = 0.5
            pred1 = ((1-row1) >= threshold).astype(int)
            pred2 = ((1-row2) >= threshold).astype(int)

            kappa_score = kappa(pred1, pred2)
            
            pairwise_kappa.append({
                'fold': fold,
                'pair_1': pair1,
                'pair_2': pair2,
                'kappa': kappa_score
            })
        
    error_df = pd.DataFrame(error_data)
    pairwise_kappa_df = pd.DataFrame(pairwise_kappa)
    pairwise_kappa_df = pairwise_kappa_df.replace(np.nan,0)
    pairwise_kappa_w_error_df = pd.merge(pairwise_kappa_df, error_df, left_on=['fold','pair_1'], right_on=['fold','pair_id'], how='left',suffixes=('_1', '_2'))
    pairwise_kappa_w_error_df = pd.merge(pairwise_kappa_w_error_df, error_df, left_on=['fold','pair_2'], right_on=['fold','pair_id'], how='left',suffixes=('_1', '_2'))
    pairwise_kappa_w_error_df['avg_mcc_error_rate'] = (pairwise_kappa_w_error_df['mcc_error_rate_1'] + pairwise_kappa_w_error_df['mcc_error_rate_2']) / 2
    pairwise_kappa_w_error_df['avg_f1_error_rate'] = (pairwise_kappa_w_error_df['f1_error_rate_1'] + pairwise_kappa_w_error_df['f1_error_rate_2']) / 2
    pairwise_kappa_w_error_df['pair_id'] = pairwise_kappa_w_error_df['pair_1'] + '&' + pairwise_kappa_w_error_df['pair_2']

    if not same_clf:
        logger.info("Using the best classifier for each representation")
        # select best base classifier for each data representation based on mcc error rate
        error_df_reset = error_df.groupby('pair_id')[['f1_error_rate','mcc_error_rate']].mean().reset_index().sort_values('f1_error_rate', ascending=True)
        error_df_reset['representation'] = error_df_reset['pair_id'].apply(lambda x: x.split('/')[0])
        error_df_reset['model'] = error_df_reset['pair_id'].apply(lambda x: x.split('/')[1])
        error_df_reset = error_df_reset[['representation', 'model', 'mcc_error_rate', 'f1_error_rate']]
        #select best model for each representation with lowest error rate
        best_models = error_df_reset.loc[error_df_reset.groupby('representation')['mcc_error_rate'].idxmin()]
        best_models['pair_id'] = best_models['representation'] + '/' + best_models['model']

        # filter pairwise_kappa_w_error_df to only include best models on pair_1 and pair_2
        pairwise_kappa_w_error_df_best = pairwise_kappa_w_error_df[pairwise_kappa_w_error_df['pair_1'].isin(best_models['pair_id']) & pairwise_kappa_w_error_df['pair_2'].isin(best_models['pair_id'])]
        pairwise_kappa_w_error_df_best = pairwise_kappa_w_error_df_best[['fold', 'pair_id', 'pair_1', 'pair_2', 'kappa', 'mcc_error_rate_1', 'mcc_error_rate_2', 'f1_error_rate_1', 'f1_error_rate_2', 'avg_mcc_error_rate', 'avg_f1_error_rate']]

        # #pairwise_kappa_w_error_df_best = pairwise_kappa_w_error_df.copy()
        # adjusted_rv = pd.read_excel('outputs/dataset_correlation/adjusted_rv_benbow.xlsx')
        # reversed_pairs = adjusted_rv.copy()
        # reversed_pairs['Representation_1'], reversed_pairs['Representation_2'] = reversed_pairs['Representation_2'], reversed_pairs['Representation_1']
        # adjusted_rv = pd.concat([adjusted_rv, reversed_pairs], ignore_index=True)

        # # assign rv coefficient to kappa df
        # pairwise_kappa_w_error_df_best['Representation_1'] = pairwise_kappa_w_error_df_best['pair_1'].apply(lambda x: x.split('/')[0])
        # pairwise_kappa_w_error_df_best['Representation_2'] = pairwise_kappa_w_error_df_best['pair_2'].apply(lambda x: x.split('/')[0])
        # pairwise_kappa_w_error_df_best = pd.merge(pairwise_kappa_w_error_df_best, adjusted_rv, 
        #                                             left_on=['Representation_1', 'Representation_2'], 
        #                                             right_on=['Representation_1', 'Representation_2'], 
        #                                             how='left')
        # pairwise_kappa_w_error_df_best.fillna(1, inplace=True)

        # pairwise_kappa_w_error_df_best = pairwise_kappa_w_error_df_best[['fold','pair_id','pair_1','pair_2','kappa','mcc_error_rate_1',
        #                                                         'f1_error_rate_1','mcc_error_rate_2','f1_error_rate_2',
        #                                                         'avg_mcc_error_rate','avg_f1_error_rate']]

        solutions_df = pd.DataFrame()

        for fold in labels.keys():
            data = pairwise_kappa_w_error_df_best[pairwise_kappa_w_error_df_best['fold']==fold].reset_index(drop=True)
            obj_1 = 'kappa' # mean_kappa or kappa
            obj_2 = f'avg_{error_metric}_error_rate'

            solutions_per_fold_df = compute_pfront_chull(data, obj_1, obj_2)

            solutions_df = pd.concat([solutions_df, solutions_per_fold_df])

        return error_df, pairwise_kappa_w_error_df_best, solutions_df
    else:
        logger.info("Using same classifier for all representations")
        pairwise_kappa_w_error_df['model_1'] = pairwise_kappa_w_error_df['pair_1'].apply(lambda x:x.split('/')[1])
        pairwise_kappa_w_error_df['model_2'] = pairwise_kappa_w_error_df['pair_2'].apply(lambda x:x.split('/')[1])
        pairwise_kappa_w_error_df = pairwise_kappa_w_error_df[pairwise_kappa_w_error_df['model_1']==pairwise_kappa_w_error_df['model_2']]
        pairwise_kappa_w_error_df['clf'] = pairwise_kappa_w_error_df['model_1']
        all_clfs = pairwise_kappa_w_error_df['model_1'].unique()

        solutions_df = pd.DataFrame()
        for clf in all_clfs:
            logger.info(f"Processing classifier: {clf}")
            data = pairwise_kappa_w_error_df[pairwise_kappa_w_error_df['clf']==clf].reset_index(drop=True)
            solutions_per_clf = pd.DataFrame()
            for fold in labels.keys():
                data_per_fold = data[data['fold']==fold].reset_index(drop=True)
                obj_1 = 'kappa' # mean_kappa or kappa
                obj_2 = f'avg_{error_metric}_error_rate'

                solutions_per_fold_df = compute_pfront_chull(data_per_fold, obj_1, obj_2)
                solutions_per_fold_df['clf'] = clf

                solutions_per_clf = pd.concat([solutions_per_clf, solutions_per_fold_df])
            
            solutions_df = pd.concat([solutions_df, solutions_per_clf])

        return error_df, pairwise_kappa_w_error_df, solutions_df

def greedy_pruning(preds, y_val,ensemble_method='voting'):
    selected = []
    best_scores = []
    best_score = 0
    remaining = list(range(len(preds)))

    while remaining:
        scores = []
        for i in remaining:
            candidate = selected + [i]

            if ensemble_method == 'voting':
                # Majority vote
                y_pred = np.round(np.mean([preds[j] for j in candidate], axis=0))

            elif ensemble_method == 'stacking':
                # Stack predictions as features
                X_meta = np.column_stack([preds[j] for j in candidate])
                meta_clf = LogisticRegression().fit(X_meta, y_val)
                y_pred = meta_clf.predict(X_meta)

            score = matthews_corrcoef(y_val, y_pred)
            scores.append(score)
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_idx = remaining[scores.index(max_score)]
            selected.append(best_idx)
            remaining.remove(best_idx)
            best_scores.append(best_score)
        else:
            break

    return selected, best_scores

# greedy ensemble selection across folds
def greedy_pruning_across_folds(all_val_preds, all_val_true, all_test_true, ensemble_method='voting_soft'):
    selected = []
    best_scores = []
    best_fold_scores = []
    best_score = 0
    remaining = list(range(all_val_preds[0].shape[1]))

    while remaining:
        scores = []
        all_fold_scores = []
        all_preds = []
        for i in remaining:
            candidate = selected + [i]

            fold_scores = []
            #preds = []
            #iterate over folds
            for val_pred, val_label, test_label in zip(all_val_preds, all_val_true, all_test_true):
                n_val_samples = len(val_label)
                subset_preds_val = [val_pred[:n_val_samples,j] for j in candidate] 
                subset_preds_test = [val_pred[n_val_samples:,j] for j in candidate] 
            
                if ensemble_method == 'voting_soft':
                    # Majority vote
                    y_prob = np.mean(subset_preds_test, axis=0) 
                    y_pred = (np.mean(subset_preds_test, axis=0) > 0.5).astype(int)
                    performance = matthews_corrcoef(test_label, y_pred)
                elif ensemble_method == 'voting_hard':
                    y_prob = np.mean(subset_preds_test, axis=0) 
                    subset_preds_test = (np.array(subset_preds_test)>0.5).astype(int)
                    y_pred = mode(subset_preds_test, axis=0).mode.flatten()
                    performance = matthews_corrcoef(test_label, y_pred)
                elif ensemble_method == 'stacking':
                    # Stack predictions as features
                    X_meta_val = np.column_stack(subset_preds_val)
                    meta_clf = LogisticRegression().fit(X_meta_val, val_label)

                    X_meta_test = np.column_stack(subset_preds_test)
                    y_prob = meta_clf.predict_proba(X_meta_test)
                    y_pred = meta_clf.predict(X_meta_test)
                    performance = matthews_corrcoef(test_label, y_pred)
                    
                #preds.append(y_prob)
                fold_scores.append(performance)
            # average the scores across folds
            score = np.mean(fold_scores)
            scores.append(score)
            all_fold_scores.append(fold_scores)
        max_score = max(scores) #best mean score across folds
        #if np.round(max_score,3) > np.round(best_score,3):
        best_score = max_score
        best_idx = remaining[scores.index(max_score)]
        best_fold = all_fold_scores[scores.index(max_score)]
        selected.append(best_idx)
        remaining.remove(best_idx)
        best_scores.append(best_score)
        best_fold_scores.append(best_fold)

    return selected, best_scores, best_fold_scores

def get_ensemble_candidates(filtered_pred_df, labels, y, ensemble_metdod='voting'):
    
    ensemble_candidates = {}
    ensemble_scores = {}

    for fold in filtered_pred_df.fold.unique():
        preds_df = filtered_pred_df[filtered_pred_df['fold']==fold]
        preds_df = preds_df.dropna(axis=1)
        pred_cols = [col for col in preds_df.columns if col.startswith('y_')]
        df_preds = preds_df.set_index('pair_id')[pred_cols]

        test_indices = labels[fold]
        ground_truth = y[test_indices]
        list_preds = {}

        for pair_id, row in df_preds.iterrows():
            pred_prob = row.values

            # Apply custom threshold
            threshold = 0.5
            preds = ((1-pred_prob) >= threshold).astype(int)
            list_preds[pair_id] = preds

        candidate_label = list(list_preds.keys())
        candidate_pred = list(list_preds.values())

        selected_idx, scores = greedy_pruning(candidate_pred, ground_truth, ensemble_metdod)
        ensemble_candidates[fold] = [candidate_label[idx] for idx in selected_idx]
        ensemble_scores[fold] = scores
    
    return ensemble_candidates, ensemble_scores

def cross_validate_ensemble_candidates(candidates_dict, labels, y, predictions_df, ensemble_methods = ['voting_soft', 'voting_hard', 'stacking']):
    
    ensemble_final_results = {}

    for ensemble_method in ensemble_methods:
        ensemble_selection_results = {}
        for ensemble_solution, candidates in candidates_dict.items():
            preds = {}
            all_val_preds = []
            all_val_true = []
            all_test_true = []
            for fold in labels.keys():
                # get predictions for each fold
                fold_predictions_df = predictions_df[predictions_df['fold']==fold]
                fold_predictions_df = fold_predictions_df[fold_predictions_df['pair_id'].isin(candidates)]
                if fold_predictions_df.empty:
                    logger.info(f"No predictions found for fold {fold} with candidates {candidates}. Skipping this fold.")
                    continue

                # drop columns with NaN values
                fold_predictions_df = fold_predictions_df.dropna(axis=1)
                pred_cols = [col for col in fold_predictions_df.columns if col.startswith('y_')]
                df_preds = fold_predictions_df.set_index('pair_id')[pred_cols]

                # get validation set and test set
                val_indices = labels[fold]['valid_idx']
                test_indices = labels[fold]['test_idx']
                y_val = y[val_indices]
                y_test = y[test_indices]
                

                fold_preds = []
                for pair_id in candidates:
                    pred = 1-df_preds.loc[pair_id].values #probability of class 1
                    fold_preds.append(pred)
                fold_preds = np.array(fold_preds).T  # shape: (n_samples_in_fold, n_classifiers)
                all_val_preds.append(fold_preds)
                all_val_true.append(y_val)
                all_test_true.append(y_test)

            selected, best_scores, best_fold_scores = greedy_pruning_across_folds(all_val_preds, all_val_true, all_test_true, ensemble_method=ensemble_method)
            
            #get the selected representations
            selected_rep = [candidates[idx] for idx in selected]

            ensemble_selection_results[ensemble_solution] = {
                'candidates': [selected_rep[0:i+1] for i in range(len(selected_rep))],
                'best_mean_score': best_scores,
                'best_fold_scores': best_fold_scores,
            }
        ensemble_final_results[ensemble_method] = ensemble_selection_results

    new_dict = {}
    for k1, v1 in ensemble_final_results.items():
        for k2, v2 in v1.items():
            new_dict[k1+'/'+k2]=v2

    final_results_df = pd.DataFrame.from_dict(new_dict, orient='index')
    final_results_df = final_results_df.explode(['candidates','best_mean_score','best_fold_scores'])
    final_results_df = final_results_df.reset_index()
    final_results_df['ensemble_method'] = final_results_df.apply(lambda x: x['index'].split('/')[0],axis=1)
    final_results_df['solution'] = final_results_df.apply(lambda x: x['index'].split('/')[1],axis=1)
    final_results_df.drop(columns='index', inplace=True)

    return ensemble_final_results, final_results_df

def get_solutions(solutions_df, m=40, n=20, error_metric='f1'):
    # select top n pairs of base classifiers per method based on the number of occurrences across folds
    pfront_solutions =  solutions_df[solutions_df['pfront'] != -1].groupby(['pair_id','pair_1','pair_2']).agg({'fold':'count'})
    pfront_solutions = pfront_solutions.rename(columns={'fold':'count'}).reset_index().sort_values(by='count',ascending=False)
    top_n_pfront_solutions = pfront_solutions.head(m)
    pfront_solutions_flattened = top_n_pfront_solutions[['pair_1','pair_2']].values.flatten()
    # select top n base classifiers
    pfront_counts = Counter(pfront_solutions_flattened)
    top_pfront = pfront_counts.most_common(n)
    pfront_ensemble_candidates = [pair for pair, _ in top_pfront]

    chull_solutions = solutions_df[solutions_df['chull'] != -1].groupby(['pair_id','pair_1','pair_2']).agg({'fold':'count'})
    chull_solutions = chull_solutions.rename(columns={'fold':'count'}).reset_index().sort_values(by='count',ascending=False)
    top_n_chull_solutions = chull_solutions.head(m)
    chull_solutions_flattened = top_n_chull_solutions[['pair_1','pair_2']].values.flatten()
    # select top n base classifiers
    chull_counts = Counter(chull_solutions_flattened)
    top_chull = chull_counts.most_common(n)
    chull_ensemble_candidates = [pair for pair, _ in top_chull]

    # pair_1 = solutions_df[['pair_1',f'avg_{error_metric}_error_rate','fold']]
    # pair_1.rename(columns={'pair_1':'pair'}, inplace=True)
    # pair_2 = solutions_df[['pair_2',f'avg_{error_metric}_error_rate','fold']]
    # pair_2.rename(columns={'pair_2':'pair'}, inplace=True)
    # best_base_candidates = pd.concat([pair_1, pair_2]).drop_duplicates(subset=['pair','fold'])
    # best_base_candidates = best_base_candidates.groupby('pair').agg({f'avg_{error_metric}_error_rate':'mean'}).reset_index()
    # best_ensemble_candidates = best_base_candidates.sort_values(f'avg_{error_metric}_error_rate').head(n)['pair'].tolist()

    best_base_candidates_list = []
    for fold in solutions_df.fold.unique():
        # check solutions per fold
        solutions_per_fold_df = solutions_df[solutions_df['fold']==fold]
        best_base_candidates = list(set(solutions_per_fold_df.sort_values(f"avg_{error_metric}_error_rate").iloc[:m, :][["pair_1", "pair_2"]].values.flatten()))
        best_base_candidates_list.extend(best_base_candidates)

    # select top n base classifiers
    best_base_candidates_counts = Counter(best_base_candidates_list)
    top_best_base = best_base_candidates_counts.most_common(n)
    best_ensemble_candidates = [pair for pair, _ in top_best_base]

    return pfront_ensemble_candidates, chull_ensemble_candidates, best_ensemble_candidates