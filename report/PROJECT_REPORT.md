# Football Player Analytics & Performance Prediction System

## Abstract

This mini-project studies player-season statistics from both supplied 2025–2026 top-five-European-league FBref CSVs. It applies data inspection and cleaning, goal-count regression, broad-position classification, Gaussian-mixture (EM) clustering, and PCA. The full and light sources are joined by player, team, and competition; matching rows and all overlapping columns are validated before modeling. The full generated dictionary records every combined column, dtype, description, missing-value share, derived status, source-file membership, task use, and selection decision. Features and claims are limited to observed data; xG/xAG and passing/progression variables are absent. Generated model scores and plots are written by `python -m src.run_analysis`.

## Introduction and problem statement

Player totals are affected by playing time and roles, while football statistics overlap mathematically and semantically. The project asks: (1) can season goals be estimated from non-derived observed statistics, (2) can broad role be inferred from performance rather than an input position label, and (3) do unsupervised models find statistically distinct player-season profiles?

## Objectives

1. Inspect the exact data schema, missingness, duplicate records, role labels, competitions, minutes, and correlation structure.
2. Create a full data dictionary and reproducible cleaned dataset.
3. Compare linear, Ridge, and Lasso regression; logistic regression, SVM, and MLP classification.
4. Discover unlabeled player profiles with an EM-based Gaussian mixture and visualize them with PCA.
5. Evaluate models using holdout and cross-validation metrics, and publish meaningful figures in a Streamlit interface.

## Dataset description and audit

Two files are supplied and both are used. The full source has 2,839 rows and 102 columns; the light source has the same 2,839 player/team/competition records and 53 columns, all included in the full source. The pipeline validates one-to-one identity matches with coverage checked in both directions and exact agreement for all shared fields, then keeps one copy of shared columns. The join preserves the full-source row order for reproducible splitting. The resulting merged data has 2,839 rows and 102 columns, with 85 numeric and 17 object columns. It spans five competitions. It has zero exact duplicate rows and zero duplicate player/team/competition records; repeated player names are valid distinct team-season observations. Position values include four single-position labels and five mixed labels. Minutes span 1–3,420 with median 1,074. Missingness is concentrated in goalkeeper-only fields (roughly 93% missing) and rate denominators such as shooting percentages for players with no attempts. Exact per-column results, source membership, and join audit are in `dataset_inspection.json` and `data_dictionary.csv`.

The file does **not** contain xG, xAG, key passes, progressive actions, or passing metrics. These requested candidate features and associated graphs cannot be produced from the data. Shots and assists are used only as observed alternatives; they are not relabeled as xG/xAG.

## Data preprocessing and feature selection

Whitespace is trimmed, numeric values are coerced, positions are mapped using the first listed position, and per-90 rates are calculated from minutes. Exact and player/team/competition duplicates are removed; the source check found none. Missing values are preserved in the cleaned file, then imputed inside model pipelines. Repeated FBref subtable copies, row ranks, identifiers, and target-derived columns are excluded.

A 450-minute threshold (five full matches) is used for modeling/rate-based profiling after inspecting the minute quantiles. It retains 1,999 of 2,839 rows (70.4%); the remaining rows stay in the cleaned file and descriptive EDA. Valid extremes are not clipped or discarded. Numeric correlations are exported for review. In particular, shots and shots on target correlate at about 0.954, so the regression keeps shots and excludes shots-on-target; repeated Gls columns correlate perfectly because they are duplicate table representations.

The explicit task feature sets are listed in the README, `preprocessing.py`, and generated inspection record. Regression excludes G+A, non-penalty goals, penalty goals, goal efficiencies, and goal-rate derivatives to prevent target leakage; xG is unavailable. Classification excludes all position text from model inputs. Clustering excludes position labels throughout its unsupervised fitting and K-selection.

## Methodology and algorithms

### Regression

The target is season `Gls`, using eligible rows and age, minutes, starts, shots, assists, crosses, selected defensive/disciplinary counts, and offsides. Ordinary least squares minimizes squared residuals. Ridge adds an L2 coefficient penalty and Lasso adds an L1 penalty; both control variance, and Lasso can perform coefficient shrinkage to zero. Data are split 80/20, with five-fold K-fold CV on training only. Median imputation and scaling are fit within each fold by pipelines. Report RMSE, MAE, and R², with actual-vs-predicted and residual plots. Since xAG is missing, no xAG predictor experiment or graph is claimed.

### Classification

The target is GK/DEF/MID/FWD mapped from the first source position token. Logistic regression uses a softmax probability model, SVM maximizes margin (RBF kernel), and MLP learns nonlinear mappings with ReLU hidden layers and backpropagation. Each uses stratified 80/20 train/test and 5-fold stratified training CV. The pipeline fits imputation and standardization on training partitions. Accuracy, macro/weighted precision, recall, F1, macro one-vs-rest specificity, and confusion matrices are saved.

### Clustering and PCA

Per-90 attacking, shooting, defensive, discipline, and available goalkeeper statistics are imputed and standardized; PCA retains 90% cumulative variance. Candidate Gaussian mixture counts 2–12 are evaluated with BIC, AIC, and silhouette. K=2 is recognized as a coarse goalkeeper/outfield split; for more informative archetypes, the selected value is the highest-silhouette candidate among K≥3, followed by BIC/AIC and profile interpretation. The source selects K=3 (silhouette ≈0.17); the resulting profiles are high attacking/wide-involvement, defensive-action, and goalkeeper statistical groups. A Gaussian mixture uses EM: posterior memberships are calculated in the E-step and mixture parameters are updated in the M-step. Position labels are never supplied to PCA, fitting, or selecting K. Cluster characteristics and names are summarized only after fitting. PCA components are covariance eigenvectors, and their eigenvalues measure retained variance. The first two PCs provide the cluster scatter plot.

## Evaluation, results, and visualizations

Run `python -m src.run_analysis` to populate `tables/` and `plots/`. On the supplied file, the Ridge model has held-out RMSE **1.57**, MAE **1.08**, and R² **0.703**; the SVM has held-out accuracy **0.748** and macro-F1 **0.801**. The selected K=3 Gaussian mixture has silhouette about **0.17** (weak separation; profiles should be interpreted cautiously), and nine PCs retain **93.0%** variance. BIC/AIC continue to improve for larger K in the search range, so the selected K prioritizes the stronger silhouette among K≥3 and a parsimonious post-fit interpretation; this is a documented trade-off, not a claim that K=3 globally minimizes BIC. Generated tables include regression holdout/CV scores and predictions, classification holdout/CV scores and per-class reports, candidate cluster criteria, PCA explained variance, cluster profiles, and test permutation importance. Figures cover meaningful available EDA (league/position/age and goals/assists/minutes distributions, goals vs shots/SoT, assists vs crosses, age vs goals per 90, selected-feature correlation), regression diagnostics and comparisons, classification confusion matrices/comparison/class balance, and PCA cluster/variance plots.

No numerical performance result is embedded in this report before execution; the generated output reflects the checked-in data and installed scikit-learn version. Interpretation should compare holdout and CV results and consider the small/imbalanced goalkeeper class, correlated counts, and one-season scope.

## Conclusion and limitations

The available data supports practical count prediction, role classification, and statistical player profiling, but is considerably narrower than the proposed full FBref dataset. Missing xG/xAG and passing/progression fields limit expected-goal and creative/progression analysis. Season totals can confound individual skill with minutes, team context, and competition strength; the 450-minute threshold helps rate stability but does not remove those confounders. Position-derived classes reflect the first listed FBref role. Clusters are statistical summaries and should not be treated as definitive tactical roles. A future edition with passing, possession, xG, xAG, and multi-season data could support stronger feature control, time-based validation, and richer tactical profiles.

## Future scope

Add verified xG/xAG and passing/progression data, expand across seasons for temporal validation, evaluate per-90 and minutes-adjusted targets, compare league-aware baselines, and validate cluster descriptions with domain experts or event-level match data. A deployment version could refresh the data and models as new seasons become available.

## References

- Football Players Stats (2025–2026), supplied FBref/Kaggle CSV files.
- Scikit-learn user guide: linear models, preprocessing pipelines, model evaluation, support vector machines, neural networks, Gaussian mixture models, and PCA.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, for regularization and statistical learning concepts.
- Bishop, *Pattern Recognition and Machine Learning*, for Gaussian mixtures and the EM algorithm.
