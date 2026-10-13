#import packages
import datetime
rundate=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
import sys
import os
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
ACOCSTRING=f'#A3D3A_DCOC {rundate}: '+" ".join(sys.argv)+'\n'
print(ACOCSTRING)

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, IsolationForest, RandomTreesEmbedding, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import pairwise_distances, balanced_accuracy_score, auc, f1_score, precision_recall_curve, silhouette_score, matthews_corrcoef, brier_score_loss, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, cross_val_score, ShuffleSplit, PredefinedSplit, cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin, clone, ClassifierMixin, ClusterMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.compose import ColumnTransformer

treeclf=ExtraTreesClassifier
treergr=ExtraTreesRegressor
development = False

import bblean
from bblean.similarity import jt_sim_matrix_packed

import multiprocessing as mp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
import gzip
import copy
import warnings
warnings.filterwarnings("ignore", message="X has feature names")
warnings.filterwarnings("ignore", message="The objective has been evaluated at point")
warnings.filterwarnings("ignore", message="The groups parameter")
warnings.filterwarnings("ignore", message="Some inputs do not have OOB")
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.Parallel` needs to be used*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used*",
    category=UserWarning,
)

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools, rdFingerprintGenerator, rdMolDescriptors, rdCoordGen, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from skopt.space import Integer, Categorical, Real
from skopt import BayesSearchCV

from wordcloud import WordCloud

class BitBirchClustering(BaseEstimator, ClusterMixin):
    """
    sklearn-compatible wrapper around bblean.BitBirch for use with BayesSearchCV.
    """

    def __init__(self, branching_factor=50, threshold=0.65, merge_criterion="diameter"):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.merge_criterion = merge_criterion
        self.tree_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """
        X should be the packed fingerprints: shape (n_samples, n_bytes), dtype=uint8.
        """
        tree = bblean.BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
        )
        tree.fit(X)

        # Build sklearn-style labels_ from cluster membership
        clusters = tree.get_cluster_mol_ids()
        labels = np.full(X.shape[0], -1, dtype=int)
        for cid, member_indices in enumerate(clusters):
            labels[member_indices] = cid

        self.tree_ = tree
        self.labels_ = labels
        return self

    def get_cluster_mol_ids(self):
        """
        Convenience passthrough to BitBirch tree (if you still want this).
        """
        if self.tree_ is None:
            return None
        return self.tree_.get_cluster_mol_ids()

    def predict(self, X):
        """
        For our BayesSearchCV usage, we only need labels_ on the *training* data.
        So predict just returns labels_ when called with the same X used in fit.
        """
        if self.labels_ is None:
            raise RuntimeError("Estimator not fitted yet.")
        # This is only safe because we always use the same X (full dataset) in cv.
        return self.labels_

    def _more_tags(self):
        # Tell sklearn this is a clusterer, not a classifier
        return {"estimator_type": "clusterer"}

#define a modelability index value more scalable than traditional MODI
def iso_MODI(
    dataframe: pd.DataFrame,
    feature_cols,
    label_col,
    n_splits: int = 5,
    random_state: int = 0,
    shuffle: bool = True
):
    """
    Cross-validated isolation-forest-based modellability index.
    
    Steps per fold:
      1) Stratified split.
      2) For each class: fit IF on training samples of that class.
         Fit a StandardScaler on that model's scores over the *entire training set* (no leakage).
      3) Score validation samples with that IF and transform with that scaler.
      4) Argmax over standardized per-class scores to assign labels.
      5) Compute MCC on the validation split.
    Returns the mean MCC over folds and the per-fold MCCs.

    Notes:
    - If a class has <2 training samples in a fold, that class is skipped for that fold
      (its validation scores are set to -inf so it cannot win the argmax).
    - Label order is fixed using np.unique(y) for reproducibility across folds.
    """
    X = dataframe[feature_cols].to_numpy()
    y = dataframe[label_col].to_numpy()
    labels = np.unique(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_mccs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        feat_scaler = StandardScaler().fit(X_train)
        X_train = feat_scaler.transform(X_train)
        X_val = feat_scaler.transform(X_val)

        # collect standardized validation scores: shape (n_val, n_labels)
        val_score_mat = np.full((len(X_val), len(labels)), -np.inf, dtype=float)

        for j, lbl in enumerate(labels):
            # training data for this class
            mask_lbl = (y_train == lbl)
            n_lbl = int(mask_lbl.sum())

            if n_lbl < 2:
                # can't fit IsolationForest with <2 samples; keep -inf scores for this class
                continue

            # sampling / tree schedule (clamped safely to training class size)
            max_samples = max(256, n_lbl // 100)
            iso_trees_eff = max((n_lbl // max_samples) * 10, 100)

            # 2) fit IF on training samples of this class
            iso_forest = IsolationForest(
                n_estimators=iso_trees_eff,
                max_samples=max_samples,
                n_jobs=-1,
                contamination='auto',
                random_state=random_state
            ).fit(X_train[mask_lbl])

            # fit scaler on the *training fold's* score distribution from this model
            train_scores = iso_forest.score_samples(X_train).reshape(-1, 1)
            scaler = StandardScaler().fit(train_scores)

            # 3) score + standardize the validation fold
            val_scores = iso_forest.score_samples(X_val).reshape(-1, 1)
            val_score_mat[:, j] = scaler.transform(val_scores).ravel()

        # 4) argmax over standardized scores → predicted labels
        pred_idx = np.argmax(val_score_mat, axis=1)
        y_pred = labels[pred_idx]

        # 5) MCC for this fold
        mcc = matthews_corrcoef(y_val, y_pred)
        fold_mccs.append(float(mcc))

    # 6) average MCC
    mean_mcc = float(np.mean(fold_mccs)) if fold_mccs else float("nan")
    return mean_mcc#, fold_mccs

#Define a single estimator that wraps RTE + RF so max_depth & n_estimators stay in sync to be used for optimizing RTE depth
class RandomTreesAutoEncoder(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_estimators: int = 10,
                 max_depth: int = None,
                 max_samples: float = 1.0,
                 max_leaf_nodes: int = None,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 n_neighbors = 1):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_samples  = max_samples
        self.random_state = random_state
        self.n_jobs       = n_jobs
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        # 1) fit the embedding
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        self.rte_ = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        Z = self.rte_.fit_transform(X).toarray().astype(bool)

        # 2) fit the regressor to reconstruct y from the embedding
        self.rfr_ = KNeighborsRegressor(n_neighbors = self.n_neighbors, n_jobs = self.n_jobs, weights = 'distance', metric='jaccard',algorithm='brute')
        self.rfr_.fit(Z, X)
        return self

    def embed(self, X):
        return self.rte_.transform(X).toarray().astype(bool)
    
    def predict(self, X=None):
        if X is None:
            return self.rfr_.predict(None)
        else:
            Z = self.embed(X)
            return self.rfr_.predict(Z)

    def score(self, X, y):
        X_scaled_true = self.scaler_.transform(X)
        X_scaled_pred = self.scaler_.transform(self.predict(X))
        distances = np.linalg.norm(X_scaled_pred - X_scaled_true,axis=1)
        return -distances.mean()

class TogglingCalibratedClassifierCV(CalibratedClassifierCV):
    """
    CalibratedClassifierCV wrapper that lets you toggle whether sample_weight
    is used during fit (for both the base estimator and the calibrator).

    Parameters
    ----------
    estimator : estimator instance, default=None
        Base classifier (same as sklearn.calibration.CalibratedClassifierCV).

    method : {'sigmoid', 'isotonic', 'temperature'}, default='sigmoid'
        Calibration method (same as CalibratedClassifierCV).

    cv : int, cross-validation generator, 'prefit' or None, default=None
        Same as CalibratedClassifierCV.

    n_jobs : int, default=None
        Same as CalibratedClassifierCV.

    ensemble : 'auto', bool, default='auto'
        Same as CalibratedClassifierCV.

    use_weights : bool, default=True
        If True, pass sample_weight to .fit().
        If False, ignore any provided sample_weight.
    """

    def __init__(
        self,
        estimator=None,
        *,
        method="sigmoid",
        cv=None,
        n_jobs=None,
        ensemble="auto",
        use_weights=True,
    ):
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            ensemble=ensemble,
        )
        self.use_weights = use_weights

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit with or without sample_weight, depending on use_weights."""
        if (not self.use_weights) or sample_weight is None:
            # Drop sample_weight entirely
            super().fit(X, y, **fit_params)
            #return self
            #return super().fit(X, y, **fit_params)
        else:
            # Use sample_weight as normal
            super().fit(X, y, sample_weight=sample_weight, **fit_params)
            #return self
            #return super().fit(X, y, sample_weight=sample_weight, **fit_params)
        # set n_jobs on every underlying fitted estimator (covers ensemble=True case)
        
        return self

# define an early  stopper callback
class NoImprovementStopper:
    def __init__(self, patience=5, skip_initial=None, verbose = False):
        """
        patience: how many *counted* iterations in a row with no improvement before stopping
        skip_initial: how many *first* iterations to ignore entirely (if None, defaults to 10)
        """
        self.patience     = patience
        self.skip_initial = skip_initial or 10
        self.iteration    = 0
        self.best_score   = -np.inf
        self.bad_iters    = 0
        self.verbose = verbose

    def __call__(self, result):
        # 1) count total callbacks
        self.iteration += 1
        current_score = -result.func_vals[-1]

        # 3) skip the warm-up, but collect the best score in the warmup
        if self.iteration <= self.skip_initial:
            if current_score > self.best_score:
                self.best_score = current_score
            return False

        # 4) check for improvement
        if current_score > self.best_score:
            self.best_score = current_score
            #self.patience = max(1,self.patience - self.bad_iters)
            self.bad_iters  = 0
        else:
            self.bad_iters += 1
            if self.verbose:
                print(f"No improvement for {self.bad_iters}/{self.patience} iterations.")

        # 5) stop if we’ve gone 'patience' rounds without bettering our best
        if self.bad_iters >= self.patience:
            if self.verbose:
                print("Stopping early due to no improvement.")
            return True

        return False        

def expected_calibration_error(y_true, y_proba, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    bin_ids = np.digitize(y_proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1) #y_proba == 1.0 needs included in the last bin
    ece = 0.0    
    for i in range(n_bins):
        bin_mask = bin_ids == i
        if np.any(bin_mask):
            bin_true = y_true[bin_mask]
            bin_proba = y_proba[bin_mask]
            ece += np.abs(bin_proba.mean() - bin_true.mean()) * bin_mask.mean()
    return ece

#function to standardize molecules
def gen_mol(SMILES):
    u = rdMolStandardize.Uncharger()
    te = rdMolStandardize.TautomerEnumerator()
    te.SetRemoveSp3Stereo(False)
    te.SetRemoveBondStereo(False)
    te.SetMaxTransforms(100)
    try:
        mol=Chem.MolFromSmiles(SMILES)
        old_smiles = Chem.MolToSmiles(mol)
        mol=rdMolStandardize.Cleanup(mol) #standard cleanup
        mol = u.uncharge(rdMolStandardize.FragmentParent(mol)) #get the desalted parent fragment uncharged
        mol = te.Canonicalize(mol) #get the canonical tautomer
        #Chem.RemoveStereochemistry(mol) #remove all stereochemistry as we're not considering it in the fingerprint
        Chem.SanitizeMol(mol)
        smiles=Chem.MolToSmiles(mol)
        rdCoordGen.AddCoords(mol)
        return old_smiles, smiles, mol
    except:
        return SMILES, np.nan, np.nan

def count_macrocycles(mol, min_size=12):
    """
    Returns the number of rings in `mol` having at least `min_size` atoms.
    """
    ri = mol.GetRingInfo()
    # AtomRings() is a tuple of tuples of atom indices for each SSSR ring
    macrocycles = [ring for ring in ri.AtomRings() if len(ring) >= min_size]
    return len(macrocycles)

def calc_1D_properties(mol):
    try:
        num_atoms=rdMolDescriptors.CalcNumHeavyAtoms(mol)
        mw=round(rdMolDescriptors.CalcExactMolWt(mol),2)
        hba=rdMolDescriptors.CalcNumHBA(mol)
        hbd=rdMolDescriptors.CalcNumHBD(mol)
        tpsa=rdMolDescriptors.CalcTPSA(mol,includeSandP=True)
        labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
        rot_bonds= rdMolDescriptors.CalcNumRotatableBonds(mol)
        numrings = rdMolDescriptors.CalcNumRings(mol)
        aliphrings= rdMolDescriptors.CalcNumAliphaticRings(mol)
        arorings= rdMolDescriptors.CalcNumAromaticRings(mol)
        satrings= rdMolDescriptors.CalcNumSaturatedRings(mol)
        macrocycles=count_macrocycles(mol)
        logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        result = [num_atoms, logp, mr, mw, hba, hbd, tpsa, labute_asa, rot_bonds, numrings, satrings, aliphrings, arorings,macrocycles,fsp3]
    except:
        result = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    return result

fpsize=1024
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=fpsize,includeChirality=True)
#rdkfpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpsize)
#apfpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpsize)
#ttfpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpsize)
fp_types={'mfp':mfpgen}#,'ttfp':ttfpgen,'rdkfp':rdkfpgen,'apfp':apfpgen}
def gen_fp(mol):
    mol_fp=np.empty(0)
    for fp in fp_types:
        new_fp=np.array(fp_types[fp].GetFingerprint(mol))
        mol_fp=np.hstack([mol_fp,new_fp])
    return mol_fp.astype(int)

def reduce_correlated_features(
    data: pd.DataFrame,
    features: list,
    linear_threshold: float = 0.99,
    nonlinear_threshold: float = 0.99,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 0,
    n_jobs: int = -1,
    max_samples: float = 1.0
) -> list:
    """
    Hybrid linear + nonlinear redundancy pruning.

    Logic
    -----------------------------------
    1. Compute pairwise squared Pearson correlations among features.

    2. If max pairwise R² > linear_threshold:
         - Identify the pair (f_i, f_j) that achieves this.
         - Compute OOB R² ONLY for these two features using ALL other features.
         - Drop the feature with higher OOB R² (more predictable).
         - If tie: randomly drop one.

    3. Else (no linear redundancy above threshold):
         - Compute nonlinear OOB R² for ALL features.
         - Drop the feature with highest OOB R² if > nonlinear_threshold.
         - Tie-break: drop the one with smallest summed impurity importance.

    4. If no feature meets nonlinear threshold → stop.
    """

    working = features.copy()

    while True:
        # sample for speed but reuse full set if needed
        #data = dataframe.copy() #.sample(frac=0.1)

        # 1 — linear redundancy: compute squared Pearson R² matrix
        corr2 = data[working].corr().pow(2)

        # get max above diagonal
        mask = np.triu(np.ones(corr2.shape, dtype=bool), k=1)
        corr2_ut = corr2.where(mask, other=np.nan)
        max_pair_r2 = np.nanmax(corr2_ut.values)

        # identify the pair (f_i, f_j)
        if max_pair_r2 > linear_threshold:
            i_max, j_max = np.argwhere(corr2_ut.values == max_pair_r2)[0]
            f_i = corr2_ut.index[i_max]
            f_j = corr2_ut.columns[j_max]

            print(f"\nPairwise Linear redundancy detected: {f_i} ↔ {f_j}, R² = {max_pair_r2:.4f}")

            # compute OOB R² for ONLY these two features
            pair_targets = [f_i, f_j]
            oob_pair = {}

            for target in pair_targets:
                X = data[working].drop(columns=[target])
                y = data[target]

                rgr = treergr(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    bootstrap=True,
                    oob_score=False,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    max_samples = max_samples
                )

                oob_pair[target] = cross_val_score(rgr, X, y, cv=5, scoring='r2', n_jobs=1).mean()

            # determine which of the pair is more redundant
            oob_i = oob_pair[f_i]
            oob_j = oob_pair[f_j]

            if oob_i > oob_j:
                drop = f_i
            elif oob_j > oob_i:
                drop = f_j
            else:
                # tie: randomly drop one
                drop = np.random.choice([f_i, f_j])
                print("Tie in OOB R² — randomly dropping")

            print(f"Dropping {drop} (linear pair; OOF R² = {oob_pair[drop]:.4f})\n")
            working.remove(drop)
            continue

        # ----------------------------------------
        # 3 — nonlinear pruning (no linear pair exceeded threshold)
        # ----------------------------------------
        print("\nNo pairwise linear redundancy above threshold — checking nonlinear predictability…")

        oob_R2 = {}
        summed_imp = {f: 0.0 for f in working}

        for target in working:
            X = data[working].drop(columns=[target])
            y = data[target]

            rgr = treergr(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=True,
                oob_score=False,
                random_state=random_state,
                n_jobs=n_jobs,
                max_samples = max_samples
            )

            oob_R2[target] = cross_val_score(rgr, X, y, cv=5, scoring='r2', n_jobs=1).mean()
            rgr.fit(X,y)
            for feat, imp in zip(X.columns, rgr.feature_importances_):
                summed_imp[feat] += imp

        # find the most predictable feature
        highest_oob = max(oob_R2.values())
        top_feats = [f for f,v in oob_R2.items() if v == highest_oob]

        if highest_oob > nonlinear_threshold:
            # if tie, choose the one with least predicted importance
            if len(top_feats) > 1:
                drop = min(top_feats, key=lambda f: summed_imp[f])
                print(f"Tie in nonlinear OOF R² — breaking by least summed importance.")
            else:
                drop = top_feats[0]

            print(f"Dropping {drop} (nonlinear OOF R² = {highest_oob:.4f})\n")
            working.remove(drop)
            continue
        else:
            print("Feature pruning complete — no nonlinear R² exceeds threshold.")
            break

    return working


experiment=sys.argv[1]
foldername = f'{experiment}_{rundate}'
os.mkdir(foldername)

dataset=pd.read_csv('B3DB_classification.tsv',sep='\t')
dataset['label']=0
dataset.loc[dataset['BBB+/BBB-']=='BBB+','label']=1
dataset=dataset.rename(columns={'SMILES':'OLD_SMILES'})
feats_1D = ['HVY_ATMS','aLogP','MR','MW','HBA','HBD','TPSA','LABUTEASA','ROTBONDS','NUMRINGS','SAT_RINGS','ALIPH_RINGS','ARO_RINGS','MACROCYCLES','Fsp3']
scaler_feats= feats_1D 
with mp.Pool(max(mp.cpu_count() * 9 // 10,1)) as pool:
    dataset[['OLD_SMILES','SMILES','ROmol']]=list(tqdm(pool.imap(gen_mol,dataset.OLD_SMILES.values.tolist(),chunksize=max(len(dataset)//10000,1)),total=len(dataset)))
    dataset=dataset[(dataset.ROmol.isna() == False)].reset_index(drop=True)
    dataset=dataset.drop_duplicates(subset='SMILES').reset_index(drop=True)
    
    dataset[feats_1D]=list(tqdm(pool.imap(calc_1D_properties,dataset.ROmol.values.tolist(),chunksize=max(len(dataset)//10000,1)),total=len(dataset)))
    dataset = dataset.dropna(subset=feats_1D).reset_index(drop=True)
    
    print(f'{len(dataset)} records remaining after removing failures and duplicate molecules')
    fps=list(tqdm(pool.imap(gen_fp,dataset.ROmol.values.tolist(),chunksize=max(len(dataset)//10000,1)),total=len(dataset)))

pool.close()
pool.join()

fps=np.array(fps).astype(bool)
fp_cols=[f'{fp}_{i}' for fp in fp_types.keys() for i in range(fpsize)]
fps=pd.DataFrame(fps,columns=fp_cols)
dataset=pd.concat([dataset,fps],axis=1)
feature_cols = fp_cols+scaler_feats

#define an outlier detection pipeline using an oneclasssvm
outlier_pipe = Pipeline([
    ("prep", ColumnTransformer(
        transformers=[
            ("desc", StandardScaler(), slice(len(fp_cols), None)),
        ],
        remainder="passthrough",
    )),
    ("norm", Normalizer(copy=False)),
    ("ocsvm", OneClassSVM(
        kernel="rbf",
        nu=0.05,
        gamma="scale",
    )),
])

def cluster_records(data, scaler_feats = scaler_feats, binary_feats = fp_cols, metric='jaccard', figure_name = None):
    print('Clustering data...')
    print('Optimizing the Random Trees Embedding...')
    
    #random trees embedding will have as many 1s as trees.
    #compute the median number of 1s set in the binary feats and set the number of trees to that value
    #since we use the jaccard distance, this will promote balance between the binary feats and the embedded scaler feats
    #we also then set max leaf nodes so that the final embedding is comparable in length to the binary features
    rte_trees = int(data[binary_feats].sum(axis=1).median().round(0))
    max_leaf_nodes = np.round(len(binary_feats) / rte_trees, 0).astype(int)
    
    rtemb=RandomTreesEmbedding(n_estimators=rte_trees, random_state=0,n_jobs=-1,max_leaf_nodes=max_leaf_nodes,max_depth=None)
    """
    rtemb.fit(data[scaler_feats])
    rte_depths = [tree.get_depth() for tree in rtemb.estimators_]
    #set the tree limit to the upper tukey fence from the unconstrained tree depths
    rte_depth_limit = np.round(np.percentile(rte_depths,75) + 3.0*(np.percentile(rte_depths,75)-np.percentile(rte_depths,25)),0).astype(int)
    rte_depth_limit = min(rte_depth_limit,max(rte_depths))
    print(f'Setting the maximum tree depth to {rte_depth_limit}')
    
    # Set up a Bayesian search over n_estimators and max_depth
    search_spaces = {
        'max_depth':    Integer(1, rte_depth_limit, prior='log-uniform'),
        'n_estimators': [rte_trees],
        'max_leaf_nodes': [max_leaf_nodes],
        'random_state': [0],
        'n_jobs': [-1]
    }
    
    #define a nonself scorer if I want to use all the data to train the RTE
    def nonself_scorer(estimator, X, y, sample_weight=None):
        X_scaled_true = estimator.scaler_.transform(X)
        X_scaled_pred = estimator.scaler_.transform(estimator.predict(None))
        distances = np.linalg.norm(X_scaled_pred - X_scaled_true,axis=1)
        return -distances.mean()
    
    opt = BayesSearchCV(
        estimator=RandomTreesAutoEncoder(),
        search_spaces=search_spaces,
        n_iter=100,
        cv = ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0), #5, #[(np.arange(len(data)),np.arange(len(data)))],
        scoring=None, #nonself_scorer,
        random_state=0,
        n_jobs=1,
        refit = False,
        verbose = False,
        return_train_score = False,
        optimizer_kwargs = {'n_initial_points':5,'initial_point_generator':'lhs'},
    )

    # fit on train (we use X_train both as features and target)
    stopper = NoImprovementStopper(patience=5, skip_initial=opt.optimizer_kwargs.get('n_initial_points', 5),verbose=True)
    opt.fit(data[scaler_feats].values,data['label'].values,callback=stopper)
    print(pd.DataFrame(opt.cv_results_)[['param_max_depth','mean_test_score']].drop_duplicates().sort_values('param_max_depth').to_string(index=False))

    print("Best hyperparameters for Random Trees Embedding:", opt.best_params_)
    print(f"Best Reconstruction Distance: {-opt.best_score_:.4f}")
    rtemb = RandomTreesEmbedding(**opt.best_params_)
    """
    #generate the random trees embedding
    embedding = rtemb.fit_transform(data[scaler_feats]).toarray().astype(int)
    print(f'Length of final embedding: {embedding.shape[1]}')
    emb_cols = [f'emb_{i}' for i in range(embedding.shape[1])]
    embedding=pd.DataFrame(embedding,columns=emb_cols)

    temp_data=pd.concat([data,embedding],axis=1)

    #combine input binary feats and embedded feats
    feature_cols = binary_feats #+ emb_cols
    
    fps_packed = np.packbits(temp_data[feature_cols].values.astype(bool), axis=-1)
    
    rng = np.random.default_rng(0)
    sample_size = min(50000, len(temp_data))
    idx = rng.choice(len(temp_data), size=sample_size, replace=False)
    sim_matrix = jt_sim_matrix_packed(fps_packed[idx])
    sil_dist_mat = (1 - jt_sim_matrix_packed(fps_packed[idx])).copy() #generated with bblean with packed fingerprints
    
    np.fill_diagonal(sim_matrix,np.nan)
    max_sims = np.nanmax(sim_matrix,axis=1)
    min_thresh = np.percentile(max_sims,10)
    max_thresh = np.percentile(max_sims,90)
    print(f'Median similarity to nearest neighbor is {np.median(max_sims).round(4)}')
    print(f'Mean similarity to nearest neighbor is {np.mean(max_sims).round(4)}')
    print(f'25th percentile similarity to nearest neighbor is {np.percentile(max_sims,25).round(4)}')
    
    """
    num_clusters=[]
    silhouette_scores = []
    thresholds=[]
    def sil_scorer(estimator, X, y=None, sample_weight=None):
        labels = estimator.labels_
        num_clusters.append(labels.max() + 1)
        thresholds.append(estimator.threshold)
        try:
            score= float(silhouette_score(sil_dist_mat, labels[idx], metric='precomputed'))
        except:
            score = 0.0
        silhouette_scores.append(score)
        return score
    
    clusterer = BitBirchClustering(branching_factor=50, threshold=0.65, merge_criterion="diameter")
    
    print(f'Limiting similarity threshold search space from {min_thresh} to {max_thresh}')
    search_spaces = {
        'threshold':  (min_thresh,max_thresh), #use the data to inform our range of thresholds to consider
    }
    
    opt = BayesSearchCV(
        estimator=clusterer,
        search_spaces=search_spaces,
        n_iter=100,
        cv = [(np.arange(len(temp_data)),idx)],
        scoring=sil_scorer,
        random_state=0,
        n_jobs=1,
        refit = False,
        verbose = False,
        return_train_score = False,
        optimizer_kwargs = {'n_initial_points':5,'initial_point_generator':'lhs'},
    )

    stopper = NoImprovementStopper(patience=5, skip_initial=opt.optimizer_kwargs.get('n_initial_points', 5),verbose=False)
    opt.fit(X = fps_packed, y=None, callback=stopper)
    cluster_results = pd.DataFrame(opt.cv_results_).sort_values('param_threshold')
    #cluster_results['dist']= 1 - cluster_results['param_threshold']
    print(pd.DataFrame(opt.cv_results_)[['param_threshold','mean_test_score']].drop_duplicates().sort_values('mean_test_score').to_string(index=True))
    
    #define the optimal threshold from the silhouette method
    opt_clusters=num_clusters[np.argmax(silhouette_scores)]
    opt_thresh=thresholds[np.argmax(silhouette_scores)]
    
    if figure_name != None:
        # Plot silhouette score and number of clusters
        fig, ax1 = plt.subplots()

        # Plot silhouette score on the left y-axis
        ax1.scatter(thresholds, silhouette_scores, label='Silhouette Score', color='blue')
        ax1.set_xlabel('Similarity Threshold')
        ax1.set_ylabel('Silhouette Score', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Mark the optimal threshold
        ax1.axvline(x=opt_thresh, color='gray', linestyle='dashed', label='Optimal Threshold')

        # Add number of clusters on the right y-axis
        ax2 = ax1.twinx()
        ax2.scatter(thresholds, num_clusters, label='Number of Clusters', color='red')
        ax2.set_ylabel('Number of Clusters', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)#, loc='lower left')

        # Title and save the plot
        plt.title(
            f'BitBirch-Lean Clustering\n'
            f'Max Score = {np.round(max(silhouette_scores), 3)} | '
            f'Opt. Clusters = {opt_clusters} | Opt. Threshold = {np.round(opt_thresh,4)}'
        )
        fig.tight_layout()
        plt.savefig(f'{foldername}/{figure_name}_{experiment}.png', bbox_inches='tight')
        #plt.show()
        plt.cla()
        plt.clf()
    """
    opt_thresh = np.percentile(max_sims,25)
    tree = bblean.BitBirch(branching_factor=50, threshold=opt_thresh, merge_criterion="diameter") #it is 1 - opt_thresh because bblean wants a similarity threshold
    tree.fit(fps_packed)
    clusters = tree.get_cluster_mol_ids()
    
    cluster_assignment = np.full(len(temp_data), -1, dtype=int)
    for cluster_id, member_indices in enumerate(clusters):
        cluster_assignment[member_indices] = cluster_id
    
    data['cluster']=cluster_assignment
    print(f'Optimal clustering similarity threshold found at {np.round(opt_thresh,4)}, giving {data.cluster.nunique()} clusters.')

#cluster the data so that sample weighting and cluster splitting can occur
df=dataset.copy()
cluster_records(df, figure_name='bblean_clustering')

#sample some clusters to visualize
cluster_counts = df[(df.MW <= df.MW.median()) & (df.HVY_ATMS >= 10)].drop_duplicates(subset=['cluster','MW'])['cluster'].value_counts()
clusters_with_multiple = cluster_counts[cluster_counts >= 4].index

# 1. Find clusters with at least 4 qualifying molecules (same as before)
cluster_counts = df[(df.MW <= df.MW.median()) & (df.HVY_ATMS >= 10)].drop_duplicates(subset=['cluster', 'MW'])['cluster'].value_counts()

clusters_with_multiple = cluster_counts[cluster_counts >= 3].index.to_numpy()

# 2. Shuffle clusters so groupings are random but non-overlapping
np.random.shuffle(clusters_with_multiple)

# 3. Work out how many images we can make (4 clusters per image, max 10 images)
n_images = min(100, len(clusters_with_multiple) // 3)

for i in range(n_images):
    sampled_clusters = clusters_with_multiple[3*i : 3*(i+1)]
    mols = (
        df[df.cluster.isin(sampled_clusters)]
        .drop_duplicates(subset=['cluster', 'MW'])
        .groupby('cluster')
        .sample(n=3, replace=False)   # 3 molecules per cluster
        .sort_values(['cluster', 'MW'])
    )
    img = PandasTools.FrameToGridImage(mols,column='ROmol', legendsCol='cluster',molsPerRow=3,subImgSize=(300, 300))
    img.save(f'{foldername}/clustered_molecules_{experiment}_sample_{i}.png')  # Saves as PNG, you can use .jpg, .bmp, etc.

"""
#compute nearest neighbors
neighbors=1
nbrs = KNeighborsClassifier(n_neighbors=neighbors, algorithm='brute',n_jobs=-1,metric='jaccard') 
nbrs.fit(df[fp_cols].values, df['label'])
distances, indices = nbrs.kneighbors()
df['nearest_label']=nbrs.predict(None)
df['dist']=distances

#shuffle labels for a permutation test
df['shuffled_label']=df.sample(frac=1, random_state=0).label.values
nbrs = KNeighborsClassifier(n_neighbors=neighbors, algorithm='brute',n_jobs=-1,metric='jaccard')
nbrs.fit(df[fp_cols].values, df['shuffled_label'])
df['nearest_shuffled_label']=nbrs.predict(None)
del nbrs

#compute MODI and our isolation forest variant
shuffle_MODI = accuracy_score(df.shuffled_label.values, df.nearest_shuffled_label.values)
MODI = accuracy_score(df.label.values, df.nearest_label.values)
print(f'\nModellability Index: {MODI:.3f}')
print(f'Modellability Index upon Permutation: {shuffle_MODI:.3f}\n')

#perform SVD on the fingerprints to make them amenable for the iso_MODI tests
svd = TruncatedSVD(n_components=min(fps.shape[1],len(df)-1), n_iter=10, random_state=0).fit(df[fp_cols])
cum = np.cumsum(svd.explained_variance_ratio_)
if cum[-1] < 0.75:
    k_75 = svd.n_components
else:
    k_75 = np.searchsorted(cum, 0.75) + 1
    
k = min(max(k_75, svd.n_components // 4), svd.n_components)

svd = svd.transform(df[fp_cols])[:,:k]
svd_cols = [f'svd_{i}' for i in range(k)]
svd=pd.DataFrame(svd,columns=svd_cols)

iMODI = iso_MODI(pd.concat([df,svd],axis=1),svd_cols + scaler_feats,'label')
shuffled_iMODI = iso_MODI(pd.concat([df,svd],axis=1),svd_cols + scaler_feats,'shuffled_label')
print(f'\nIsolation Forest Modellability Index: {iMODI:.3f}')
print(f'Isolation Forest Modellability Index upon Permutation: {shuffled_iMODI:.3f}\n')
df=df.drop(columns=['shuffled_label','nearest_shuffled_label'])

#examine the distribution of kNN distances
sns.histplot(df,x='dist',binrange=(0,1),bins=20,stat='percent',common_norm=False,hue='label')
plt.title(f'kNN Distance Distribution\n Compounds = {len(df)}')
plt.xlabel(f'kNN Average Jaccard Distance (k={neighbors})')
plt.tight_layout()
plt.savefig(f'{foldername}/kNN_distances_{experiment}.png', bbox_inches='tight')
plt.cla()
plt.clf()
"""

#Balanced accuracies
balacc_scores_overall=[]
balacc_scores_in=[]
balacc_scores_out=[]

#MCC
mcc_scores_overall=[]
mcc_scores_in=[]
mcc_scores_out=[]

#f1 scores
f1_scores_overall=[]
f1_scores_in=[]
f1_scores_out=[]

#pr-auc scores
auc_scores_overall=[]
auc_scores_in=[]
auc_scores_out=[]

uncal_eces=[]
cal_eces=[]
uncal_eces_in=[]
cal_eces_in=[]
uncal_eces_out=[]
cal_eces_out=[]

uncal_briers=[]
cal_briers=[]
uncal_briers_in=[]
cal_briers_in=[]
uncal_briers_out=[]
cal_briers_out=[]

overall_count = []
inlier_count =[]
outlier_count=[]

split_type = []

#create the splitters
cvs = 5 # of folds for data splitters; drive the % holdout with this
test_cvs = (1 if development else 5) #drive the number of times to loop through the test folds
val_cvs = (1 if development else 5) #drive the number of times to loop through the val folds
sgkf = StratifiedGroupKFold(n_splits=cvs,shuffle=True,random_state=0)
skf = StratifiedKFold(n_splits=cvs,shuffle=True,random_state=0)
sss = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 0)

#distribute threads between individual forests and the togglingclassifier
threads = max(os.cpu_count() // 10 * 8, 1)
num_trees = 100 if development == False else 25
tgclf_jobs = min(cvs,threads) if development else 1
tree_threads = max(threads // tgclf_jobs,1) if development else -1
print(tree_threads, tgclf_jobs)

#possible classifiers
if development:
    clf_list = [RandomForestClassifier(n_estimators = num_trees, n_jobs=tree_threads, random_state=0)] 
else:
    clf_list = [ExtraTreesClassifier(n_estimators = num_trees, n_jobs=tree_threads, random_state=0), RandomForestClassifier(n_estimators = num_trees, n_jobs=tree_threads, random_state=0)]

param_space = {
    'estimator': Categorical(clf_list),
    'estimator__criterion': ['gini'], #, 'log_loss'],
    #'estimator__random_state': [0],
    #'estimator__n_jobs': [tree_threads],
    #'estimator__n_estimators': [num_trees],
    #'estimator__bootstrap': [True],
    #'estimator__max_samples': [1.0],
    #'estimator__max_depth': Integer(1, 100), #,prior='log-uniform'),
    'estimator__class_weight': [None,'balanced','balanced_subsample'],
    'use_weights': [True,False],
    'method':['temperature','sigmoid', 'isotonic']
}

calibrated_clf = TogglingCalibratedClassifierCV(
    estimator = RandomForestClassifier(),
    ensemble = False,
    cv = skf, 
    n_jobs = tgclf_jobs)

bayes = BayesSearchCV(
    calibrated_clf,
    param_space,
    n_iter=1000 if development == False else 20,
    cv=sgkf,
    scoring= 'neg_log_loss', 
    n_jobs=1,
    random_state=0,
    refit = True,
    return_train_score = True,
    optimizer_kwargs = {'n_initial_points':40 if development == False else 5,'initial_point_generator':'lhs'},
)

print(f'Hyperparameter tuning score: {bayes.scoring}')

#perform nested CV
split_dict = {'Random Holdouts':skf,'Cluster-Aware Holdouts':sgkf, 'Permutation Test':sgkf}
for splitter_name in split_dict.keys():
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Running {splitter_name}...')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    splitter = split_dict[splitter_name]
    for fold, (train_index, test_index) in tqdm(enumerate(splitter.split(df[feature_cols], df['label'], groups=df.cluster.values)), total=splitter.get_n_splits()):
        #break
        if fold == test_cvs:
            break
        split_type.append(splitter_name)
        print(f"Fold {fold}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
        test = df[df.index.isin(test_index)].copy().reset_index(drop=True)
        train = df[df.index.isin(train_index)].copy().reset_index(drop=True)
        
        if splitter_name == 'Permutation Test':
            train['label']=train.sample(frac=1,random_state=0)['label'].values
        
        baseline_brier = ((test['label'] - test['label'].mean())**2).mean()
        print(f'Baseline Brier Score is {baseline_brier:.3f}')
    
        #perform clustering on the training set
        cluster_records(train)
        
        #set cluster weights as 1 / cluster size
        cluster_sizes = train["cluster"].value_counts()
        train["cluster_weight"] = train["cluster"].map(lambda c: 1 / cluster_sizes[c])
        
        #setup the validation folds for hyperparameter tuning - this is done just to enable some subset/reduced number of cvs for development purposes
        train['val_split']=-1
        for val_fold, (_, val_index) in enumerate(sgkf.split(train[feature_cols], train['label'], groups=train.cluster.values)):
            if val_fold == val_cvs:
                break
            train.loc[train.index.isin(val_index),'val_split']=val_fold
        
        ps=PredefinedSplit(train.val_split.values)
        
        #overfit an extratreesclassifier to each training fold with and without sample weights and take the maximum observed depth to limit the search space during hyperparameter tuning
        upper_depth_limit = 0
        for sub_train_idx, _ in ps.split(train[feature_cols], train['label']):
            for criterion in param_space['estimator__criterion']:
                for class_weight in param_space['estimator__class_weight']:
                    for sample_weight in param_space['use_weights']: #[None, train.iloc[sub_train_idx].cluster_weight.values]:
                        sample_weight = train.iloc[sub_train_idx].cluster_weight.values if sample_weight == True else None
                        for test_clf in clf_list:
                            overfit_clf = clone(test_clf)
                            overfit_clf.set_params(n_estimators=num_trees, random_state=0,criterion=criterion,n_jobs=-1, class_weight = class_weight)
                            overfit_clf.fit(train.iloc[sub_train_idx][feature_cols],train.iloc[sub_train_idx]['label'],sample_weight=sample_weight)
                            depths = [tree.get_depth() for tree in overfit_clf.estimators_]
                            upper_depth_limit = max(max(depths),upper_depth_limit) # if max(depths) > upper_depth_limit else upper_depth_limit
        
        print(f'Setting the maximum tree depth to {upper_depth_limit}')
        bayes.search_spaces['estimator__max_depth'] = Integer(1, upper_depth_limit, prior='log-uniform')
        
        #run recursive feature elimination
        remaining_feats = feature_cols.copy()
        best_score = -np.inf
        rfe_rounds = 0
        while rfe_rounds < 10:
            break
            hp_search = clone(bayes)
            hp_search.cv = ps
            stopper = NoImprovementStopper(patience=10, skip_initial=hp_search.optimizer_kwargs.get('n_initial_points', 5))
            hp_search.fit(train[remaining_feats],train['label'].values,groups=train.cluster.values, sample_weight=train.cluster_weight.values,callback=stopper)
            temp_clf = hp_search.best_estimator_
            print(hp_search.best_score_)
            test_condition = (hp_search.best_score_ <= best_score)
            print(test_condition)
            if test_condition:
                rfe_rounds = rfe_rounds + 1
            else:
                rfe_rounds = 0
                best_features = remaining_feats.copy()
                best_score = hp_search.best_score_
                clf = copy.deepcopy(hp_search.best_estimator_)
            
            temp_importance=pd.DataFrame(remaining_feats,columns=['feature'])
            temp_importance['importance']= temp_clf.calibrated_classifiers_[0].estimator.feature_importances_ # permutation_importance(temp_clf , train[remaining_feats], train['label'].values, scoring='neg_log_loss',n_repeats=3 if development else 10,random_state=0,max_samples=1.0,n_jobs=-1).importances_mean.reshape(-1,1) #
            temp_importance = temp_importance.sort_values('importance',ascending=False).reset_index(drop=True)
            remaining_feats =temp_importance.head(-1)['feature'].tolist()
            print(temp_importance.tail(1))
            depths = [tree.get_depth() for tree in temp_clf.calibrated_classifiers_[0].estimator.estimators_]
            print(np.min(depths), np.max(depths))
            #print(hp_search.best_params_)
            bayes.search_spaces['estimator__max_depth'] = Integer(max(1,np.min(depths)-20), min(np.max(depths)+20,upper_depth_limit), prior='log-uniform')
            bayes.optimizer_kwargs['n_initial_points'] = bayes.search_spaces['estimator__max_depth'].high - bayes.search_spaces['estimator__max_depth'].low if development == False else 5
        
        
        #print(len(best_features))
        print(f'Performing hyperparameter tuning for fold {fold}')
        hp_search = clone(bayes)
        hp_search.cv = ps
        
        #set the tree limit to the upper tukey fence from the unconstrained tree depths
        print(f'Setting the maximum tree depth to {upper_depth_limit}')
        hp_search.search_spaces['estimator__max_depth'] = Integer(1, upper_depth_limit, prior='log-uniform')
        stopper = NoImprovementStopper(patience=10, skip_initial=hp_search.optimizer_kwargs.get('n_initial_points', 5))
        hp_search.fit(train[feature_cols],train['label'].values,groups=train.cluster.values, sample_weight=train.cluster_weight.values,callback=stopper)
        print(pd.DataFrame(hp_search.cv_results_).sort_values('mean_test_score')[['param_estimator','param_estimator__class_weight','param_method','param_estimator__max_depth', 'param_use_weights','param_estimator__criterion','mean_test_score']])
        best_score = hp_search.best_score_ #.cv_results_['mean_test_score'][hp_search.best_index_]
        clf=hp_search.best_estimator_
        print('Best Score:', np.round(best_score,3), '| Best Params:', hp_search.best_params_)
        
        cal = clf.calibrated_classifiers_[0]
        underlying_clf = cal.estimator
        #calibrator = cal.calibrators[0]
        depths = [tree.get_depth() for tree in underlying_clf.estimators_]
        print(f'Median tree depth = {np.median(depths)}')
        if np.max(depths) == upper_depth_limit:
            print(f'THE MAX OBSERVED DEPTH ({np.max(depths)}) == THE UPPER LIMIT OF THE SEARCH SPACE ({upper_depth_limit})')
                
        #predict on the test set
        test['prob']=underlying_clf.predict_proba(test[feature_cols])[:,1].reshape(-1,1)
        test['cal_prob']=clf.predict_proba(test[feature_cols])[:,1].reshape(-1,1)
        test['pred']=0
        test.loc[test.cal_prob >= 0.5, 'pred']=1
        
        #train a oneclasssvm for outlier detection
        outclf = clone(outlier_pipe)
        outclf.fit(train[feature_cols])
        
        #apply the isolation forest
        test['IF_SCORE']=outclf.decision_function(test[feature_cols])
        test['in_out']=np.where(test.IF_SCORE < 0, -1,1)
    
        brier_score = brier_score_loss(test['label'], test['prob'])
        ece = expected_calibration_error(test['label'], test['prob'])
        uncal_briers.append(brier_score)
        uncal_eces.append(ece)
        print(f"Uncalibrated: Expected Calibration Error: {ece:.4f} | Brier Score Loss: {brier_score:.4f}")
        prob_true, prob_pred = calibration_curve(test['label'].values, test['prob'].values, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'Uncalibrated (ECE = {ece:.3f} | Brier Loss = {brier_score:.3f})')
        brier_score = brier_score_loss(test['label'], test['cal_prob'])
        ece = expected_calibration_error(test['label'], test['cal_prob'])
        cal_briers.append(brier_score)
        cal_eces.append(ece)
        print(f"Calibrated: Expected Calibration Error: {ece:.4f} | Brier Score Loss: {brier_score:.4f}")
        prob_true, prob_pred = calibration_curve(test['label'].values, test['cal_prob'].values, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='x', label=f'Calibrated (ECE = {ece:.3f} | Brier Loss = {brier_score:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'{splitter_name} Hold Out Test Set (fold {fold})\nCalibration Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{foldername}/{splitter_name}_calibration_curve_fold_{fold}_{experiment}.png', bbox_inches="tight")
        plt.cla()
        plt.clf()
        
        #generate inlier and outlier set
        inliers=test[test.in_out == 1]
        outliers=test[test.in_out == -1]
    
        overall_count.append(len(test))
        inlier_count.append(len(inliers))
        outlier_count.append(len(outliers))
        
        #append uncalibrated brier score by tier
        for score_list, results in [(uncal_briers_in,inliers),(uncal_briers_out,outliers)]:
            try:
                score_list.append(brier_score_loss(results['label'], results['prob']))
            except:
                score_list.append(np.nan)
        
        #append calibrated brier score by tier
        for score_list, results in [(cal_briers_in,inliers),(cal_briers_out,outliers)]:
            try:
                score_list.append(brier_score_loss(results['label'], results['cal_prob']))
            except:
                score_list.append(np.nan)
        
        #append uncalibrated ECE by tier
        for score_list, results in [(uncal_eces_in,inliers),(uncal_eces_out,outliers)]:
            try:
                score_list.append(expected_calibration_error(results['label'], results['prob']))
            except:
                score_list.append(np.nan)
        
        #append calibrated ECE by tier
        for score_list, results in [(cal_eces_in,inliers),(cal_eces_out,outliers)]:
            try:
                score_list.append(expected_calibration_error(results['label'], results['cal_prob']))
            except:
                score_list.append(np.nan)
        
        #append balanced accuracy by tier
        for score_list, results in [(balacc_scores_overall,test),(balacc_scores_in,inliers),(balacc_scores_out,outliers)]:
            try:
                score_list.append(balanced_accuracy_score(results.label.values,results.pred.values))
            except:
                score_list.append(np.nan)
    
        #append MCC by tier
        for score_list, results in [(mcc_scores_overall,test),(mcc_scores_in,inliers),(mcc_scores_out,outliers)]:
            try:
                score_list.append(matthews_corrcoef(results.label.values,results.pred.values))
            except:
                score_list.append(np.nan)
    
        #append f1 by tier
        for score_list, results in [(f1_scores_overall,test),(f1_scores_in,inliers),(f1_scores_out,outliers)]:
            try:
                score_list.append(f1_score(results.label.values,results.pred.values))
            except:
                score_list.append(np.nan)
    
        #append pr-auc by tier
        for score_list, results in [(auc_scores_overall,test),(auc_scores_in,inliers),(auc_scores_out,outliers)]:
            try:
                precision, recall, _ = precision_recall_curve(results.label.values,results.cal_prob.values)
                prauc=auc(recall, precision)
                score_list.append(prauc)
            except:
                score_list.append(np.nan)
        
        print('\n')
    
    #break

#create a bar plot of the segmented results
overall=pd.DataFrame({'MCC':mcc_scores_overall,'Balanced Accuracy':balacc_scores_overall,'F1-score':f1_scores_overall,'PR-AUC':auc_scores_overall,'Experiment':split_type,'Sample Count':overall_count,'Tier':'Overall'})
tier_in=pd.DataFrame({'MCC':mcc_scores_in,'Balanced Accuracy':balacc_scores_in,'F1-score':f1_scores_in,'PR-AUC':auc_scores_in,'Experiment':split_type,'Sample Count':inlier_count,'Tier':f'Inliers'})
tier_out=pd.DataFrame({'MCC':mcc_scores_out,'Balanced Accuracy':balacc_scores_out,'F1-score':f1_scores_out,'PR-AUC':auc_scores_out,'Experiment':split_type,'Sample Count':outlier_count,'Tier':f'Outliers'})

#tiers=pd.concat([oob_overall, oob_tier_in, oob_tier_out,overall,tier_in,tier_out]).reset_index(drop=True).round(3)
tiers=pd.concat([overall,tier_in,tier_out]).reset_index(drop=True).round(3)
tiers.to_csv(f'{foldername}/BBBp_performance_{experiment}.tsv',sep='\t',index=None)

for splitter_name in split_dict.keys():
    #print a table of the means
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'\nOverall {splitter_name} Performance')
    print(tiers[tiers['Experiment'] == splitter_name].groupby('Tier').mean(numeric_only=True).round(3))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#plot results
ax=sns.barplot(tiers[tiers['Sample Count'].notna()],x='Tier',y='Sample Count',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Segmented Sample Count (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f'{foldername}/sample_count_by_tier_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot results
ax=sns.barplot(tiers[tiers['F1-score'].notna()],x='Tier',y='F1-score',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Segmented Model Performance (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,1.05)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f'{foldername}/f1_by_tier_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

#plot performance results
ax=sns.barplot(tiers[tiers['MCC'].notna()],x='Tier',y='MCC',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Segmented Model Performance (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig(f'{foldername}/mcc_by_tier_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

ax=sns.barplot(tiers[tiers['Balanced Accuracy'].notna()],x='Tier',y='Balanced Accuracy',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Segmented Model Performance (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,1.05)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f'{foldername}/balanced_accuracy_by_tier_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

ax=sns.barplot(tiers[tiers['PR-AUC'].notna()],x='Tier',y='PR-AUC',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Segmented Model Performance (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f'{foldername}/prauc_by_tier_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

uncal_df=pd.DataFrame({'ECE':uncal_eces,'Brier Score':uncal_briers,'Calibration':'Uncalibrated','Experiment':split_type,'Tier':'Overall'})
uncal_df_in=pd.DataFrame({'ECE':uncal_eces_in,'Brier Score':uncal_briers_in,'Calibration':'Uncalibrated','Experiment':split_type,'Tier':'Inliers'})
uncal_df_out=pd.DataFrame({'ECE':uncal_eces_out,'Brier Score':uncal_briers_out,'Calibration':'Uncalibrated','Experiment':split_type,'Tier':'Outliers'})
cal_df=pd.DataFrame({'ECE':cal_eces,'Brier Score':cal_briers,'Calibration':'Calibrated','Experiment':split_type,'Tier':'Overall'})
cal_df_in=pd.DataFrame({'ECE':cal_eces_in,'Brier Score':cal_briers_in,'Calibration':'Calibrated','Experiment':split_type,'Tier':'Inliers'})
cal_df_out=pd.DataFrame({'ECE':cal_eces_out,'Brier Score':cal_briers_out,'Calibration':'Calibrated','Experiment':split_type,'Tier':'Outliers'})

ax=sns.barplot(pd.concat([uncal_df,cal_df]),x='Calibration',y='ECE',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'ECE with/without Calibration (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,)
plt.tight_layout()
plt.savefig(f'{foldername}/ECE_by_calibration_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

ax=sns.barplot(pd.concat([cal_df, cal_df_in, cal_df_out]),x='Tier',y='ECE',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Post-Calibration ECEs (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,)
plt.tight_layout()
plt.savefig(f'{foldername}/ECE_by_Segment_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

ax=sns.barplot(pd.concat([uncal_df,cal_df]),x='Calibration',y='Brier Score',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Brier Score with/without Calibration (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{foldername}/Brier_by_calibration_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

ax=sns.barplot(pd.concat([cal_df, cal_df_in, cal_df_out]),x='Tier',y='Brier Score',capsize=0.05,err_kws={'linewidth': 2},hue='Experiment')
plt.title(f'Post-Calibration Brier Scores (n = {test_cvs} repeats)')
plt.xlabel('Segment')
plt.ylim(0,)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{foldername}/Brier_by_segment_{experiment}.png', bbox_inches="tight")
plt.cla()
plt.clf()

#save the calibration data
pd.concat([uncal_df,uncal_df_in,uncal_df_out,cal_df,cal_df_in,cal_df_out]).to_csv(f'{foldername}/BBBp_calibration_{experiment}.tsv',sep='\t',index=None)

#now train on all data following the nested CV
X = df[feature_cols]
y = df['label']

#assign cluster_weights
cluster_sizes = df["cluster"].value_counts()
df["cluster_weight"] = df["cluster"].map(lambda c: 1 / cluster_sizes[c])

train = df.reset_index(drop=True).copy()

#train a oneclass svm for outlier detection
outclf = clone(outlier_pipe)
outclf.fit(train[feature_cols])

#setup the validation folds for hyperparameter tuning
train['val_split']=-1
for val_fold, (_, val_index) in enumerate(sgkf.split(train[feature_cols], train['label'], groups=train.cluster.values)):
    if val_fold == val_cvs:
        break
    train.loc[train.index.isin(val_index),'val_split']=val_fold

ps=PredefinedSplit(train.val_split.values)

print('Tuning/Training the production model.')
hp_search = clone(bayes)
hp_search.cv = ps
stopper = NoImprovementStopper(patience=10 if development == False else 5, skip_initial=hp_search.optimizer_kwargs.get('n_initial_points', 5))

#overfit an extratreesclassifier to each training fold with and without sample weights and take the maximum observed depth to limit the search space during hyperparameter tuning
upper_depth_limit = 0
for sub_train_idx, _ in ps.split(train[feature_cols], train['label']):
    for criterion in param_space['estimator__criterion']:
        for class_weight in param_space['estimator__class_weight']:
            for sample_weight in param_space['use_weights']: #[None, train.iloc[sub_train_idx].cluster_weight.values]:
                sample_weight = train.iloc[sub_train_idx].cluster_weight.values if sample_weight == True else None
                for test_clf in clf_list:
                    overfit_clf = clone(test_clf)
                    overfit_clf.set_params(n_estimators=num_trees, random_state=0,criterion=criterion,n_jobs=-1, class_weight = class_weight)
                    overfit_clf.fit(train.iloc[sub_train_idx][feature_cols],train.iloc[sub_train_idx]['label'],sample_weight=sample_weight)
                    depths = [tree.get_depth() for tree in overfit_clf.estimators_]
                    upper_depth_limit = max(depths) if max(depths) > upper_depth_limit else upper_depth_limit

#set the tree limit to the upper tukey fence from the unconstrained tree depths
print(f'Setting the maximum tree depth to {upper_depth_limit}')
hp_search.search_spaces['estimator__max_depth'] = Integer(1, upper_depth_limit, prior='log-uniform')

hp_search.fit(train[feature_cols],train['label'].values,groups=train.cluster.values,sample_weight=train.cluster_weight.values,callback=stopper)
best_score = hp_search.best_score_
clf=hp_search.best_estimator_
print('Best Score:', np.round(best_score,3), '| Best Params:', hp_search.best_params_)

cal = clf.calibrated_classifiers_[0]
underlying_clf = cal.estimator
#calibrator = cal.calibrators[0]
depths = [tree.get_depth() for tree in underlying_clf.estimators_]
print(f'Median tree depth = {np.median(depths)}')
if np.max(depths) == upper_depth_limit:
    print(f'THE MAX OBSERVED DEPTH ({np.max(depths)}) == THE UPPER LIMIT OF THE SEARCH SPACE ({upper_depth_limit})')

if development == False:
    
    #qet 95% of nonconformity scores for conformal prediction
    with gzip.open(f'{foldername}/BBBp_pred_{experiment}_singlemodel_{rundate}.pkl.gz','wb') as f:
        pickle.dump({'clf':clf,'outclf':outclf,'features':feature_cols,'date':rundate},f)
    
    #train 100 models built on bootstrap samples of the training data and save each
    print('Training multiple models for uncertainty in production')
    clfs=[]
    for n in tqdm(range(100)):
        sub_train = train.groupby('label', group_keys=False).sample(frac=1,replace=True,random_state=n).sample(frac=1, random_state=n)
        sub_clf = clone(clf)
        sub_clf.set_params(estimator__random_state=n)
        sub_clf.fit(sub_train[feature_cols],sub_train['label'].values,sample_weight=sub_train.cluster_weight.values)
        clfs.append(sub_clf)

    with gzip.open(f'{foldername}/BBBp_pred_{experiment}_bootstrapmodels_{rundate}.pkl.gz','wb') as f:
        pickle.dump({'clfs':clfs,'features':feature_cols,'date':rundate},f)

#analyze the importance
print('Computing permutation importance')
sgkf.random_state = 100
for imp_fold, (train_index, imp_index) in enumerate(tqdm(sgkf.split(train[feature_cols], train['label'], groups=train.cluster.values), total=sgkf.get_n_splits())):
    sub_train=train[train.index.isin(train_index)].copy()
    imp_set = train[train.index.isin(imp_index)].copy()
    sub_clf = clone(clf)
    sub_clf.random_state=100
    sub_clf.fit(sub_train[feature_cols],sub_train['label'].values,sample_weight=sub_train.cluster_weight.values)
    
    train.loc[train.index.isin(imp_index),'prob'] = sub_clf.predict_proba(imp_set[feature_cols])[:,1]
    
    imp = permutation_importance(sub_clf, imp_set[feature_cols], imp_set['label'].values, scoring='neg_log_loss',n_repeats=1 if development else 5,random_state=0,max_samples=1.0,n_jobs=1).importances_mean.reshape(-1,1) #clf.calibrated_classifiers_[0].estimator.feature_importances_
    if imp_fold == 0:
        importance=pd.DataFrame(feature_cols,columns=['feature'])
        importance['importance']=imp
    else:
        temp_importance=pd.DataFrame(feature_cols,columns=['feature'])
        temp_importance['importance']=imp
        importance = pd.concat([importance, temp_importance],axis=0)
        del temp_importance
    if development:
        break

importance.to_csv(f'{foldername}/BBBp_permutation_importance_{experiment}.tsv',sep='\t',index=None)

#generate an ordering of just the top 20 most important features on average
order = (
    importance.groupby('feature')['importance']
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .index
)

sns.barplot(importance[importance.feature.isin(order)], x='feature',y='importance',capsize=0.05,err_kws={'linewidth': 2, 'color':'black'}, order=order, color='blue')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.title('BBBp Permutation Feature Importance')
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_permutation_feature_importance.png', bbox_inches="tight")
if development:
    plt.show()
plt.cla()
plt.clf()

importance = importance.groupby('feature',as_index=False).mean().sort_values('importance',ascending=False)

topn_features = 20
freqs = dict(zip(importance.head(topn_features)["feature"], importance.head(topn_features)["importance"]))
wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate_from_frequencies(freqs)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_permutation_wordcloug.png', bbox_inches="tight")
plt.cla()
plt.clf()

#now consider impurity based importance
importance=pd.DataFrame(feature_cols,columns=['feature'])
importance['impurity_importance']=clf.calibrated_classifiers_[0].estimator.feature_importances_
importance = importance.sort_values('impurity_importance',ascending=False).reset_index(drop=True)
importance['cumulative_impurity_importance']= importance["impurity_importance"].cumsum()

# Plot the feature importance
fig, ax1 = plt.subplots()

top_n_features=20

# Bar plot for feature importance
ax1.bar(importance['feature'].head(top_n_features), importance['impurity_importance'].head(top_n_features), color='blue') #,alpha=0.75)
ax1.set_xlabel('Feature')
ax1.set_ylabel('Feature Importance', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0,)
plt.xticks(rotation=90)

# Line plot for cumulative importance
ax2 = ax1.twinx()
ax2.plot(importance['feature'].head(top_n_features), importance['cumulative_impurity_importance'].head(top_n_features), color='red', marker='o', linestyle='-')
ax2.set_ylabel('Cumulative Importance', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0,)

# Display the plot
plt.title('BBBp Impurity Feature Importance')
plt.tight_layout()
plt.savefig(f'{foldername}/BBBp_Impurity_Feature_Importance_{experiment}.png', bbox_inches="tight")
if development:
    plt.show()
plt.cla()
plt.clf()

importance.to_csv(f'{foldername}/BBBp_impurity_importance_{experiment}.tsv',sep='\t',index=None)

topn_features = 20
freqs = dict(zip(importance.head(topn_features)["feature"], importance.head(topn_features)["impurity_importance"]))
wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate_from_frequencies(freqs)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(f'{foldername}/{experiment}_impurity_wordcloug.png', bbox_inches="tight")
plt.cla()
plt.clf()
