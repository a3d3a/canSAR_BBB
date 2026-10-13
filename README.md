# canSAR Blood–Brain Barrier Permeability Predictor

This repository contains the machine-learning pipeline used to develop the blood–brain barrier (BBB) permeability predictor deployed in [canSAR](https://cansar.ai).

## Dataset

The model was trained using the [B3DB (Blood–Brain Barrier Database)](https://github.com/theochem/B3DB), a curated dataset of experimentally characterized BBB-permeable (BBB+) and BBB-impermeable (BBB−) compounds.

Molecular structures were standardized using RDKit, including salt removal, charge neutralization, and tautomer canonicalization. Duplicate standardized structures and molecules that failed processing were removed.

## Molecular Representations

Each compound was represented using:

- 1,024-bit Morgan fingerprints (radius 3, including chirality).
- 15 physicochemical descriptors calculated using RDKit, including molecular weight, aLogP, topological polar surface area (TPSA), hydrogen-bond donors and acceptors, molecular flexibility, and ring characteristics.

## Model Training and Validation

Random Forest and Extra Trees classifiers were evaluated using Bayesian hyperparameter optimization and nested cross-validation.

To assess generalizability to structurally distinct compounds, molecules were clustered using BBLean BitBirch clustering. StratifiedGroupKFold cross-validation was used to keep structurally related molecules together, reducing chemical-series leakage between training and test sets.

Model selection included optimization of classifier hyperparameters, class weighting, cluster-based sample weighting, and probability calibration using sigmoid, isotonic, or temperature scaling.

Performance was evaluated using balanced accuracy, Matthews correlation coefficient (MCC), F1-score, precision–recall AUC (PR-AUC), Brier score, and expected calibration error (ECE). Random holdouts and label-permutation experiments were also included for comparison.

Following validation, the production model was trained using the full curated dataset. An additional ensemble of 100 bootstrap-trained models was generated to support uncertainty estimation.

## Availability

BBB permeability predictions and associated probabilities are available through [canSAR](https://cansar.ai) for more than 7 million compounds.

## Reference

Meng et al. (2021). *B3DB: A Blood–Brain Barrier Database with Data on Physicochemical Properties and Brain Penetration.* Scientific Data.

[https://doi.org/10.1038/s41597-021-01069-5](https://doi.org/10.1038/s41597-021-01069-5)

One detail worth preserving is the distinction between the two validation approaches: your code uses both conventional stratified random splits and cluster-aware splits, with nested optimization inside the outer holdouts. That's a meaningful methodological strength and worth documenting even in a short README.
