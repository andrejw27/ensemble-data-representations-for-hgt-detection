# ensemble-data-representations-for-hgt-detection

Horizontal gene transfer (HGT) is widely recognized as a major driver of antimicrobial resistance (AMR) dissemination. Detecting HGT, especially the transmission of AMR genes, is essential for improving AMR surveillance. Numerous computational approaches have been developed for this purpose, including recent advances in machine learning (ML). Several studies in other fields have shown that ML model performance depends on data representations. Combining multiple data representations in ensemble learning has been shown to improve performance in other genomics tasks. However, this approach has not yet been evaluated for HGT detection. Thus, we investigate the efficacy of integrating diverse data representations in ensemble learning for HGT detection, particularly for classification tasks. Then, we assess its applicability to localizing genomic islands (GIs), which are clusters of genes acquired through HGT, in a genomic sequence. We implemented a two-stage ensemble selection strategy to determine the optimal combination of data representations. Our results demonstrate the effectiveness of ensemble learning for HGT detection and highlight the necessity of reframing the HGT detection problem. Our ensemble selection strategy reveals that combining low-correlated data representations yields better recall (Recall: 0.978) than individual learning models (Recall: 0.959). Our study aligns with the state of the art by showing that models designed for classification tasks are not optimal for localizing specific features in a genome. This indicates that future research should redefine the problem of HGT detection.

---

**Steps to reproduce the results:**

1. Clone this repo:

```
git clone git@github.com:andrejw27/ensemble-data-representations-for-hgt-detection.git
```

2. `cd` into the root directory (`ensemble-data-representations-for-hgt-detection`
3. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html): `conda/install.sh`
4. Create conda environment: `sh conda/create_env.sh`
5. Activate conda environment: `conda activate genomic-data-rep`
6. **Optional** : Remove conda environment (if necessary): `sh conda/remove_env.sh`
7. Open the `visualization.ipynb` and follow the steps in the notebook
