# ML4FramePopulation：Population-Based Damage Identification and Resilience Analysis

This project combines structural simulation, vibration-based damage identification, machine learning, experimental validation, and building-portfolio resilience analysis.

## Workflow

| Step | Main Script | Purpose |
| --- | --- | --- |
| 1. Generate buildings | `frame2d.py` | Create random reinforced-concrete frames and obtain modal properties and pushover-based damage thresholds. |
| 2. Prepare earthquakes | `prepare_ground_motions.py` | Organize ground-motion records and metadata. |
| 3. Generate healthy responses | `generate_healthy_wn.py` | Simulate white-noise responses as healthy references. |
| 4. Simulate damage | `generate_eq_and_damaged_wn_batch.py` | Apply earthquakes at different PGA levels, assign damage states, and generate post-earthquake white-noise responses. |
| 5. Build the dataset | `build_population_dataset.py` | Extract damage-sensitive features and combine them with building metadata and damage labels. |
| 6. Train classifiers | `ML_pipeline_batch_betterplot.py` | Compare feature combinations and classifiers with Optuna tuning. |
| 7. Analyze results | `generate_aggregated_results_betterplot.py` | Summarize performance across random seeds. |
| 8. Validate experimentally | `experimental_evaluation_betterplot.py` | Evaluate simulation-trained models on experimental measurements. |
| 9. Assess resilience | `building_portfolio_resilience_analysis.py` | Model inspection, repair, and community recovery under different damage-estimation strategies. |

## Damage Classification

The dataset contains four damage states, DS0–DS3, assigned using maximum interstory drift and building-specific thresholds.

Experiments compare 12 engineering-based damage indicators, reduced feature subsets, and additional building geometry, material, load, and modal information.

Classifiers include Random Forest, XGBoost, LightGBM, SVM, KNN, MLP, Logistic Regression, and Stacking. Training and testing are separated by building ID. Evaluation uses grouped five-fold cross-validation and five random seeds, reporting accuracy, macro-F1, AUC, and confusion matrices.

## Additional Comparisons

- **Catch22:** `build_population_dataset_catch22.py` extracts changes in 22 roof-acceleration features.
- **MiniRocket:** `Minirocket_optuna_multi_classifier.py` uses healthy and damaged ground/roof acceleration histories.
- **Fragility baseline:** `Fragility_classifier.py` classifies damage using single-indicator fragility models.
- **Interpretability:** `feature_interaction_network.py` analyzes SHAP feature importance and interactions.
- **Age-dependent resilience:** `multi_year_resilience_manual_inspection.ipynb` compares recovery for As-built, 30-year, 50-year, and 70-year fragility scenarios.

## Usage

Follow the workflow in order and configure input/output paths before running each script. Ground motions, simulation responses, and experimental data must be supplied or generated. The `run_*.sh` scripts support Slurm batch execution.

Main dependencies: OpenSeesPy, NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, XGBoost, LightGBM, Optuna, SHAP, joblib, pycatch22, and sktime.
