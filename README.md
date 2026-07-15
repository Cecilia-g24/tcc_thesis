# tcc_thesis (https-new)

Master's thesis project: automated assessment of **Transcultural Competence (TCC)** in clinical interview transcripts using psycholinguistic features (LIWC) and large language models.

The dataset consists of role-play interview recordings in which medical students demonstrate TCC across five dimensions. Each transcript is scored by three human raters (0–4 scale). The pipeline extracts linguistic features and evaluates whether those features predict rater scores.

## Five TCC Dimensions

| ID | Dimension |
|----|-----------|
| d1 | Illness Beliefs (*Krankheitsvorstellungen*) |
| d2 | Lack of Knowledge (*Nichtwissen*) |
| d3 | Cultural Factors (*Kulturelle Faktoren*) |
| d4 | Family System (*Familiensystem*) |
| d5 | Ambiguity Tolerance (*Ambiguitätstoleranz*) |

## Data Flow

```text
data/data_raw/
├── transcripts/          ← interview transcripts (.txt, German)
└── results_human_rating/ ← rater scores per dimension (DimensionN.xlsx)
        │
        ▼
  [pre_processor.py]  ← merge + translate + export
        │
        ▼
data/data_clean/
├── 01_csvs_for_liwc_manual_input/
│   ├── full_dataset_de.csv   ← German text + ratings  (→ DE-LIWC2015)
│   └── full_dataset_en.csv   ← German + English translation + ratings  (→ LIWC-22)
│
│   ⚠️  MANUAL STEP: import these CSVs into the LIWC desktop app
│       and export results to 02_results_liwc_dict/ (see below)
│
└── 02_results_liwc_dict/     ← LIWC output CSVs go here
```

## `pre_processor.py`

Single-entry pipeline that produces the two LIWC-ready CSVs from raw data.

**Steps:**

1. **Merge** — auto-discovers all `.txt` transcript files (organised across Part 1 / Part 2+3 / Part 4 subfolders) and all `DimensionN.xlsx` rating files. Joins them on `(participant_id, dimension)` and reports any unmatched records.
2. **Translate** — translates the German transcript text to English using Google Translate (`deep-translator`), with chunking to respect the 4 000-character API limit and automatic retry on failure.
3. **Export** — writes two UTF-8 CSV files with one row per `(participant, dimension)`:
   - `full_dataset_de.csv` — German text only
   - `full_dataset_en.csv` — German text + English translation

**Usage:**

```bash
# default paths (data/data_raw/ → data/data_clean/01_csvs_for_liwc_manual_input/)
python pre_processor.py

# custom paths
python pre_processor.py --data-raw path/to/data_raw --output path/to/out

# merge only, skip translation and file writing
python pre_processor.py --dry-run
```

**Dependencies:** `pandas`, `deep-translator`, `tqdm` (optional)

## Manual LIWC Step

> **After `pre_processor.py` finishes, two CSVs are created in `data/data_clean/01_csvs_for_liwc_manual_input/`. These must be manually imported into the LIWC desktop application. LIWC cannot be called programmatically — this step requires human interaction with the GUI.**

Steps:

1. Open the LIWC desktop app.
2. Import `full_dataset_de.csv` → select the **DE-LIWC2015 Dictionary (German)** → run analysis → export the result CSV.
3. Import `full_dataset_en.csv` → select the **LIWC-22 Dictionary (English)** → run analysis → export the result CSV.
4. Save both exported CSVs into `data/data_clean/02_results_liwc_dict/`.

Dictionaries used:

| CSV | Dictionary | Language | Dimensions |
|-----|------------|----------|------------|
| `full_dataset_de.csv` | DE-LIWC2015 | German | 98 |
| `full_dataset_en.csv` | LIWC-22 | English | 118 |

## Approach 1: LIWC Regression (`scripts/approach_1_liwc/`)

Four near-identical entry points train a LIWC → `average_score` regressor per TCC dimension, varying only language (EN/DE) and optimization goal (MAE/RMSE):

- `01_train_mae_as_goal_en.py` / `01_train_mae_as_goal_de.py`
- `02_train_rmse_as_goal_en.py` / `02_train_rmse_as_goal_de.py`

**Pipeline steps (per script):**

1. **Load config & data** — read `configs/paths.json` to resolve the LIWC results CSV for the chosen language and the output directory (`results/approach1/{en,de}_results/{mae,rmse}_as_goal`).
2. **Per dimension (d1–d5), build features** — filter rows to that dimension, drop rows with missing `average_score`, then drop metadata columns (`id`, `text`, rater columns, etc.) and keep only numeric LIWC columns as `X` (118 features for EN/LIWC-22, 98 for DE/LIWC2015).
3. **Resume check** — skip dimensions that already have a `status/<dim>_DONE.json` marker and exported CSVs, reusing those instead of refitting.
4. **Build the pipeline search grid** (BioPsyKit `SklearnPipelinePermuter`):
   - `imputer`: median
   - `scaler`: Standard / Robust / MinMax
   - `reduce_dim`: `SelectKBest(f_regression, k=10/20/40/all)` / `RFE(Ridge, n_features=10/20/40)` / `passthrough` (no reduction)
   - `reg`: Baseline (DummyRegressor), Ridge, Lasso, ElasticNet, BayesianRidge, SVR (linear/RBF), HistGradientBoosting, RandomForest, GradientBoosting, KNN, DecisionTree
5. **Fit with nested CV** — outer `KFold(5, shuffle=True, seed=42)` for an honest held-out score per pipeline combination; inner `KFold(3, seed=0)` for hyperparameter search (grid, or randomized search for the larger grids), scored by `neg_mean_absolute_error` (01_*) or `neg_root_mean_squared_error` (02_*).
6. **Checkpoint** — pickle the fitted permuter after each dimension (and on interruption/error) so a rerun can resume instead of refitting from scratch.
7. **Compute metrics** — from the outer-fold predictions, compute MAE/RMSE/R²/Pearson/Spearman/etc. for every pipeline combination tried, export per-dimension CSVs, and write the `DONE` marker.
8. **Combine across dimensions** — after all 5 dimensions finish, export project-wide reports: `detailed_pipeline_metrics.csv`, `best_model_per_dimension.csv`, `best_model_family_per_dimension.csv`, `model_family_summary.csv`, `best_vs_baseline.csv` (vs. the Baseline/DummyRegressor), `outer_cv_predictions.csv`, plus `run_metadata.json` and `run_log.txt`.

**Feature reduction — already built in and empirically mixed:**

With 118 EN features against only ~225 rows per dimension (and even fewer per inner-CV training fold), dimensionality reduction is a real risk factor. The pipeline already treats `reduce_dim` as a tuned hyperparameter rather than a fixed preprocessing decision, so nested CV picks per dimension whichever of `SelectKBest` / `RFE` / `passthrough` generalizes best. Across all 20 dimension runs so far (5 dimensions × 2 languages × 2 goals):

- `passthrough` (full feature set) won 11/20 times — usually paired with models that have their own regularization (SVR_RBF, GradientBoosting, RandomForest, HistGradBoost).
- `SelectKBest` won 8/20 times.
- `RFE` won 1/20 time.

Conclusion: no separate/global feature-reduction step is needed — the existing per-pipeline search already covers it, and results show it isn't a one-size-fits-all decision (it depends on the dimension and the model family chosen alongside it).

## Repository Structure

```text
tcc_thesis/
├── pre_processor.py              ← main preprocessing script
├── configs/
│   ├── paths.json                ← configurable data paths
│   └── api_models.json           ← LLM model configuration
├── data/
│   ├── data_raw/
│   │   ├── transcripts/          ← raw .txt transcript files
│   │   └── results_human_rating/ ← DimensionN.xlsx rating files
│   └── data_clean/
│       ├── 01_csvs_for_liwc_manual_input/  ← pre_processor.py output
│       └── 02_results_liwc_dict/           ← LIWC desktop app output
├── scripts/
│   ├── approach_1_liwc/          ← LIWC → regression pipeline (see above)
│   ├── approach_2_llm/           ← LLM-based approach (placeholder, not yet implemented)
│   └── approach_3_hybrid/        ← LIWC + LLM hybrid approach (placeholder, not yet implemented)
├── results/
│   └── approach1/                ← outputs from scripts/approach_1_liwc/
└── utils/                        ← helper scripts
```

## Setup

```bash
# pre_processor.py
pip install pandas deep-translator tqdm openpyxl

# scripts/approach_1_liwc/
pip install biopsykit scikit-learn pandas numpy
```
