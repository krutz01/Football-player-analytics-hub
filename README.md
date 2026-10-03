# Football Player Analytics & Performance Prediction System

A reproducible university machine-learning project built from both supplied **Football Players Stats (2025–2026)** CSV sources. The workflow is data-first: inspect the real schemas, publish a complete data dictionary, clean without silently deleting low-minute records, and run regression, classification, EM/Gaussian-mixture clustering, PCA, and a Streamlit dashboard.

## Dataset inspection (actual supplied file)

Both source files are used: `players_data-2025_2026.csv` (full) and `players_data_light-2025_2026.csv` (light). The full CSV has **2,839 rows × 102 columns**; the light CSV has the same 2,839 player/team/competition records and **53 columns**, all present in the full CSV. The pipeline performs a validated one-to-one merge on `(Player, Squad, Comp)`, verifies record coverage in both directions, and checks all 53 shared fields for exact agreement before retaining one copy of each field. The merge preserves full-source row order for reproducible train/test splits. Thus, the modeling table is the union of the two verified sources (102 columns, since light is a strict subset), rather than duplicated copies or two appended cohorts. It has **85 numerical and 17 categorical/object columns**, zero exact duplicate rows, and zero duplicated `(Player, Squad, Comp)` records. There are 152 rows whose player name also occurs elsewhere—these are multiple records for a player across teams, not duplicates. Missing ages occur in two records; keeper-table columns are mostly missing because they apply only to goalkeepers. The complete column-level inventory, dtypes, descriptions, missing percentages, derived status, source-file membership, inclusion decisions, and task mapping is generated at `report/data_dictionary.csv`; `report/dataset_inspection.json` records the two files, merge check, exact schema, missingness, duplicate checks, quantiles, and feature rationale.

There are five observed competitions: Premier League, La Liga, Bundesliga, Serie A, and Ligue 1. The nine source positions are `GK`, `DF`, `MF`, `FW`, and the five comma-separated combinations `MF,FW`, `MF,DF`, `FW,MF`, `DF,MF`, `DF,FW`. The ordered first position token maps to GK/DEF/MID/FWD. This is a consistent and explicit rule rather than use of the detailed position field as a predictor.

**Unavailable in this particular CSV:** xG, xAG, key passes, progressive passes/carries/receptions, and passing volume/completion. No models or plots pretend these columns exist. `Sh` is the available shot-volume substitute; assists are the nearest observed creative-output statistic to xAG. The full source includes shooting, playing-time, miscellaneous, and goalkeeper tables, but no passing table.

### Player-time rule

Minutes range from 1 to 3,420; median is 1,074, while the 10th percentile is about 48. **450 minutes (five complete matches)** is the documented minimum for supervised modeling and rate-based player profiling. It reduces extreme rate noise without restricting analysis to only regular starters: 1,999/2,839 (70.4%) player-seasons meet it. Lower-minute records remain in `data/processed/players_cleaned.csv` and the descriptive dataset. Valid extreme performers are retained; no IQR-based row deletion is performed.

### Selected fields and leakage controls

The exact machine-readable lists are in `src/preprocessing.py` and `report/dataset_inspection.json`.

| Task | Selected available features |
|---|---|
| Goals regression | Age, minutes, starts, shots, assists, crosses, tackles won, interceptions, fouls drawn/committed, offsides, yellow cards, red cards |
| Position classification | Age, minutes, starts, goals, assists, shots/on-target, crosses, tackles won, interceptions, fouls, offsides, cards, plus available goalkeeper measures (GA/90, shots faced, saves, save%, clean-sheet%) |
| Clustering/PCA | Per-90 goals, assists, shots, shots on target, crosses, tackles won, interceptions, fouls drawn/committed, offsides, saves, plus goalkeeper GA/90, save%, clean-sheet% |

Regression targets the season total `Gls`. It excludes xG (absent), all directly derived goal totals (`G+A`, `G-PK`, `G+A-PK`), the shooting-table goal duplicate, penalty goals, goal efficiencies, and rates derived from goals. `Sh` is retained, while `SoT` is excluded from regression because it is a subset of shots and is very highly correlated with `Sh` (Pearson r≈0.954). Repeated FBref table columns (for example `Min_stats_playing_time`) and identifiers are not treated as independent features. The generated numeric-correlation report includes all observed pairs, not only selected features.

## Methods and mathematical background

### Preprocessing

The workflow removes only exact duplicates and repeated `(player, team, competition)` records, trims text, converts numeric fields with coercion, and standardizes broad positions. Blank keeper values are left missing in the cleaned data; supervised scikit-learn pipelines impute them from training data. A missing goalkeeper statistic means “not recorded for this player”, not an observed zero. There is no blanket outlier clipping: high values can be legitimate season performances. Generated per-90 rates use source minutes/90 and are only used in player profiling.

### Regression — season goals

Linear regression minimizes the residual sum of squares, `Σ(yᵢ − (β₀ + xᵢᵀβ))²`. Ridge adds `αΣβⱼ²` to shrink correlated coefficients; Lasso adds `αΣ|βⱼ|` and can set coefficients to zero. Shrinkage reduces variance/overfitting when predictors overlap. The held-out split is 80/20; five-fold CV is run on training data only. Median imputation and standardization are pipeline steps, fit within every training fold. Compare RMSE (goal-count error with larger errors penalized), MAE (typical absolute goal-count error), and R² (variance explained relative to a mean baseline). Test-set permutation importance is descriptive, not another feature-selection pass on the training labels. The xAG-vs-goals plot/experiment is deliberately omitted because xAG is absent; assists-vs-goals is shown instead.

### Classification — broad playing position

Multinomial logistic regression models class probabilities with softmax (the binary sigmoid generalizes to multiple classes). SVM learns a maximum-margin boundary; the RBF kernel permits nonlinear separation. The MLP uses ReLU hidden units and a softmax output; backpropagation computes loss gradients through the layers and updates weights. Inputs are performance features only; the source position field is solely the supervised target. Each model uses an 80/20 stratified holdout and five-fold stratified CV with training-only imputation/scaling. Report accuracy, macro and weighted precision/recall/F1, one-vs-rest macro specificity, and confusion matrices. Macro metrics weight each class equally; weighted metrics reflect support.

### Clustering and PCA

The GMM has latent component assignments `zᵢ`; EM alternates between an E-step estimating posterior membership probabilities and an M-step updating Gaussian mixture weights, means, and covariances. GMM is the EM-based clustering algorithm here; diagonal covariance and multiple starts provide a practical, stable fit. Position labels are not used in preprocessing, feature selection, PCA, fitting, or choosing cluster count. Candidate K=2…8 are compared using BIC (complexity-penalized likelihood), AIC, and silhouette score; choose lowest BIC, using silhouette as supporting separation evidence. Cluster descriptions are computed only after assignments, from feature means, and are interpretations—not labels given to the algorithm.

PCA follows median imputation → standardization → PCA, retaining enough components to reach 90% cumulative explained variance. Each principal component is an eigenvector direction of the sample covariance matrix; its eigenvalue gives variance along that direction. Orthogonal components compress correlated inputs while preserving the largest possible variance. GMM is trained on these retained components; the first two components are used to visualize assignments.

The run compares K=2…12. K=2 is a very coarse goalkeeper/outfield split; for more informative archetypes, the selected value is the highest-silhouette candidate among K≥3, then BIC/AIC and profile interpretability are reviewed. This explicitly balances the silhouette peak against BIC/AIC, which can keep improving as mixtures add components. The supplied dataset selects **K=3** (silhouette ≈0.17); its post-fit statistical profiles are high attacking/wide-involvement, defensive-action, and goalkeeper profiles. These are descriptions derived after fitting, not labels supplied to the algorithm or tactical ground truth.

## Run

Requires Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python -m src.run_analysis
streamlit run app/app.py
```

The analysis command generates the cleaned CSV, data dictionary, dataset inspection, correlations, evaluation tables, serialized models, and PNG figures. The dashboard supports player-season lookup, editable goal prediction inputs, model comparisons, confusion-matrix figures, and cluster/PCA player profiling.

## Project files

```text
data/raw/players_data-2025_2026.csv
data/raw/players_data_light-2025_2026.csv
data/processed/players_cleaned.csv          # generated
notebooks/                                   # guided executable analyses
src/preprocessing.py                         # inspection, dictionary, cleaning
src/feature_selection.py
src/regression.py
src/classification.py
src/clustering.py
src/visualization.py
src/run_analysis.py
app/app.py
models/                                      # generated joblib artifacts
plots/                                       # generated EDA and evaluation figures
tables/                                      # generated metrics/profiles/predictions
report/data_dictionary.csv                   # generated, all 102 fields and source membership
report/dataset_inspection.json               # generated actual-schema record
report/PROJECT_REPORT.md
requirements.txt
```

All numeric results are generated from the checked-in CSV by `python -m src.run_analysis`; the scripts do not hard-code presumed model scores.
