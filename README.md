# Shopzilla CSAT Prediction — ANN-8

ANN-8 is a TensorFlow/Keras five-class ANN experiment that addresses the ANN-7 class-collapse problem using controlled random oversampling and undersampling on the training partition only.

## Files
- `train_ann8.py`: preprocessing, feature engineering, train-only resampling, ANN training and evaluation.
- `prediction_ann8.py`: loads the matching ANN-8 artifacts and performs single/bulk predictions.
- `app_ann8.py`: Streamlit application.
- `artifacts/`: generated model and preprocessing artifacts.
- `reports/`: generated metrics, class distributions and plots.

## Setup (Windows PowerShell)
Use Python 3.11.

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place your original `eCommerce_Customer_support_data.csv` in this folder. It is intentionally not bundled in this ZIP.

## Train
```powershell
python train_ann8.py
```

## Run Streamlit
```powershell
python -m streamlit run app_ann8.py
```

## Important methodology
1. Split original data into stratified train/validation/test partitions.
2. Fit imputation, encoding, TF-IDF and SVD using only the training partition.
3. Resample only the transformed training partition.
4. Keep validation and test distributions unchanged.
5. Select the checkpoint by validation macro-F1 (balanced accuracy breaks close ties).
6. Evaluate once on the untouched test partition.

Input Features
      │
      ▼
Dense Layer — 256 neurons
      │
Batch Normalization
      │
ReLU Activation
      │
Dropout — 40%
      │
      ▼
Dense Layer — 128 neurons
      │
Batch Normalization
      │
ReLU Activation
      │
Dropout — 30%
      │
      ▼
Output Layer — 5 neurons
      │
Softmax
      │
      ▼
CSAT Prediction (1–5)

Initial training resampling targets for CSAT classes 1–5 are 9,000 / 6,000 / 7,000 / 9,000 / 25,000. These are experimental settings and should be revised only based on validation results.

## Reports
- `reports/ann8_metrics.json`
- `reports/ann8_classification_report.json`
- `reports/ann8_confusion_matrix.png`
- `reports/ann8_loss_curve.png`
- `reports/ann8_resampling_details.json`
- `reports/ann8_training_history.csv`
- `reports/ann8_validation_metrics.csv`
- `reports/original_class_distribution.csv`

## Interpretation
Do not judge the model using accuracy alone. Compare macro-F1, balanced accuracy and per-class recall against ANN-7. Resampling cannot guarantee that classes become separable, need to check confusion matrix and use real validation/test data.

This is a post-interaction quality-review model because it uses response delay, handling time and customer remarks. It is not an intake-time prediction model.

ANN-8 uses stratified splitting, train-only preprocessing, TF-IDF/SVD text feature reduction, controlled oversampling/undersampling instead of class weights, a regularized TensorFlow ANN, and Macro-F1-based model selection.
