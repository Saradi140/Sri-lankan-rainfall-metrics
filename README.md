# Sri Lankan Rainfall: Ensemble Learning Dashboard

Bagging, boosting, stacking and hybrid models forecasting the next 10-day (dekadal) rainfall total for Sri Lanka's 25 districts.

* **Data:** WFP/CHIRPS subnational dekadal rainfall for Sri Lanka, Jan 2022 to Nov 2026 (`data/lka-rainfall-subnat-5ytd.csv`), district level (adm_level 2).
* **Task:** predict `rfh` (mm in the next dekad) from past rainfall, anomalies, long-term average and seasonal encodings. All features use only information available before the target dekad.
* **Validation:** time-based split. Train before 2025-07-01, test on the unseen 2025-07 to 2026-11 period. Stacking uses time-ordered out-of-fold predictions.

## Models compared
| Family | Models |
|---|---|
| Bagging | Random Forest, Extra Trees |
| Boosting | Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Stacking | 5-model stack with non-negative Ridge meta-learner, plain average blend |
| Hybrid | SARIMAX + RF (residual), SARIMAX + LightGBM (residual), RF + ANN (residual) |
| Baselines | Climatology, persistence, single decision tree, SARIMAX alone, MLP alone |

## Run it
```bash
pip install -r requirements.txt          # dashboard only
streamlit run app.py

pip install -r requirements-notebook.txt # to re-run the training notebook
jupyter notebook rainfall_ensemble_learning.ipynb
```
Deploy free on Streamlit Community Cloud: push this folder to GitHub, choose `app.py` as the entry point.

## Files
* `rainfall_ensemble_learning.ipynb`: full training, evaluation and interpretation (executed, with outputs)
* `app.py`: Streamlit dashboard (overview, model arena, backtest viewer, next-dekad outlook, what-if predictor, insights)
* `models/`: trained `.joblib` models (re-fitted on all data) · `artifacts/`: metrics, predictions, forecasts, SHAP plot


## AI-assistance disclosure
Parts of the code were generated with the help of an AI assistant (Claude, by Anthropic), then run and interpreted by the student author.
