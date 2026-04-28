# ensemble-data-representations-for-hgt-detection

Horizontal gene transfer (HGT) is widely recognized as a major driver of antimicrobial resistance (AMR) dissemination , with genomic islands (GIs) as one of the drivers facilitating the spread. Detecting GIs is essential for improving AMR surveillance.
Numerous computational approaches have been developed for GIs detection, including recent advances in machine learning (ML). Several studies in other fields have shown that ML model performance depends on data representations. Combining multiple data representations in ensemble learning has been shown to improve performance in other genomics tasks. However, this approach has not yet been evaluated for GIs detection. To this end, we investigate the efficacy of integrating diverse data representations in ensemble learning for GIs detection, particularly for classification task. Then, we assess its applicability to localizing GIs, which are clusters of genes acquired through HGT, in a genomic sequence. We implemented a two-stage ensemble selection strategy to determine the optimal combination of data representations. Our ensemble selection strategy reveals that combining low-correlated data representations in an ensemble classifier yields better Recall than individual representation for the classification task. Nevertheless, the ensemble classifier could not localize GIs better, suggesting that the cross-task generalizability remains constrained. This finding presents an opportunity for future research to advance the field by redefining the problem formulation of GIs detection.

---

**Steps to reproduce the results:**

1. Clone this repo:
   ```
   git clone git@github.com:andrejw27/ensemble-data-representations-for-hgt-detection.git
   ```
2. `cd` into the root directory (`ensemble-data-representations-for-hgt-detection`
3. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html): `conda/install.sh`
4. Create conda environment: `sh conda/create_env.sh`
5. Activate conda environment: `conda activate ensemble_rep_hgt`
6. **Optional** : Remove conda environment (if necessary): `sh conda/remove_env.sh`
7. Extract the trained models in `utils/models`
8. Open the `visualization.ipynb` and follow the steps in the notebook

---

1. Cross-validate 44 data representations and 5 machine learning models
   ```
   python run_crossval.py --representation-index 1 --n-worker 6 --filename "benbow" --output-dir "outputs/cross_val/predictions" 
   ```

> --representation-index : refer to set of data representations to be executed
> --n-worker : number of workers to process the script in parallel
> --filename : name of the dataset
> -- output-dir : folder to store the outputs

2. Evaluate cross-validation results
   ```
   python compute_cross_val.py --dataname "benbow" --output-dir "outputs/cross_val"
   ```

> --dataname : name of the dataset
> --output-dir : folder of the outputs from the cross-validation

3. Perform the two-stage ensemble selection strategy to select the best candidates for ensemble classifier
   ```
   python ensemble_selection_cv.py --dataname "benbow" --same-clf "True" --error-metric "mcc"
   ```

> --same-clf : a flag to determine whether to run the ensemble selection strategy using the best classifier per data representation (True/1) or the same classifier for all data representations (False/0). Heterogeneous classifiers use the best classifier per representation, and homogeneous classifiers use same classifier for all representations.
> --error-metric : the evaluation metric used to measure the performance for the selection strategy, we used "mcc" for the study

4. Train models either single classifier or ensemble classifier
   ```
   python train_model.py --candidates "RCKmer-7/SVM,Subsequence/SVM" --ensemble-type "voting_soft" --default-params "True"
   ```

> --candidates : candidates for the ensemble classifier. If the input is single e.g. "RCKmer-7/SVM", the script will train the single classifier
> --ensemble-type : type of ensemble learning, e.g. voting_soft, voting_hard, and stacking
> --default-params : a flag to determine whether to use default hyperparameters or best hyperparameters for the model. Currently, it is only for SVM

5. Evaluate the models on boundaries prediction task -> predict the location of genomic islands (GIs) within a genome
   ```
   python run_predictGI.py --result-type "test"
   ```

> --result-type : a flag to decide which test dataset will be used for evaluation. "test" refers to species in Benbow test dataset, and "literature" refers to species in literature dataset

---
