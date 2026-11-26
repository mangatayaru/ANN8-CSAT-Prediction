from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "eCommerce_Customer_support_data.csv"
ARTIFACTS, REPORTS = ROOT / "artifacts", ROOT / "reports"
ARTIFACTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

TARGET = "CSAT Score"
NUMERIC = ["Item_price", "connected_handling_time", "response_delay_minutes",
           "issue_hour", "issue_weekday", "issue_month", "order_to_issue_hours",
           "remark_length", "remark_exclamation_count", "remark_question_count"]
CATEGORICAL = ["channel_name", "category", "Sub-category", "Tenure Bucket",
               "Agent Shift", "Customer_City", "Product_category"]
DATE_COLUMNS = ["order_date_time", "Issue_reported at", "issue_responded"]
TEXT = "Customer Remarks"

# Controlled training targets. These are experimental caps, not the validation/test distribution.
# Class order is 1, 2, 3, 4, 5.
TARGET_COUNTS = {0: 9000, 1: 6000, 2: 7000, 3: 9000, 4: 25000}

def engineer_features(frame):
    x = frame.copy().drop(columns=[TARGET], errors="ignore")
    for c in DATE_COLUMNS + CATEGORICAL + [TEXT, "Item_price", "connected_handling_time"]:
        if c not in x.columns:
            x[c] = np.nan
    for c in DATE_COLUMNS:
        x[c] = pd.to_datetime(x[c], errors="coerce", dayfirst=True)
    order, issue, response = x["order_date_time"], x["Issue_reported at"], x["issue_responded"]
    x["response_delay_minutes"] = (response - issue).dt.total_seconds() / 60
    x.loc[(x["response_delay_minutes"] < 0) | (x["response_delay_minutes"] > 43200),
          "response_delay_minutes"] = np.nan
    x["issue_hour"] = issue.dt.hour
    x["issue_weekday"] = issue.dt.weekday
    x["issue_month"] = issue.dt.month
    x["order_to_issue_hours"] = (issue - order).dt.total_seconds() / 3600
    x.loc[(x["order_to_issue_hours"] < 0) | (x["order_to_issue_hours"] > 8760),
          "order_to_issue_hours"] = np.nan
    x[TEXT] = x[TEXT].fillna("").astype(str).str.strip()
    x["remark_length"] = x[TEXT].str.len()
    x["remark_exclamation_count"] = x[TEXT].str.count("!")
    x["remark_question_count"] = x[TEXT].str.count(r"\?")
    for c in CATEGORICAL:
        x[c] = x[c].fillna("Unknown").astype(str).str.strip()
        x.loc[x[c].isin(["", "nan", "None"]), c] = "Unknown"
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x[NUMERIC + CATEGORICAL + [TEXT]]

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
        ("remarks", TfidfVectorizer(max_features=2000, ngram_range=(1, 2),
                                    min_df=2, sublinear_tf=True), TEXT)
    ], sparse_threshold=1.0)

def controlled_resample(x, y):
    """Random oversampling/undersampling of TRAIN only, after feature transformation."""
    rng = np.random.default_rng(SEED)
    parts_x, parts_y, details = [], [], {}
    for cls in range(5):
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            raise ValueError(f"Training split contains no examples for CSAT {cls+1}.")
        target_n = TARGET_COUNTS[cls]
        if len(idx) >= target_n:
            chosen = rng.choice(idx, size=target_n, replace=False)
            method = "undersampled" if len(idx) > target_n else "unchanged"
        else:
            chosen = rng.choice(idx, size=target_n, replace=True)
            method = "oversampled"
        parts_x.append(x[chosen])
        parts_y.append(np.full(target_n, cls, dtype=np.int32))
        details[str(cls+1)] = {
            "original_training_records": int(len(idx)),
            "resampled_training_records": int(target_n),
            "method": method
        }
    x_res = np.concatenate(parts_x, axis=0).astype("float32")
    y_res = np.concatenate(parts_y, axis=0).astype("int32")
    order = rng.permutation(len(y_res))
    return x_res[order], y_res[order], details

def build_ann(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,), name="engineered_features")
    x = tf.keras.layers.Dense(256, kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax", name="csat_scores")(x)
    model = tf.keras.Model(inputs, outputs, name="Shopzilla_ANN8")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, save_path):
        super().__init__()
        self.x_val, self.y_val, self.save_path = x_val, y_val, save_path
        self.best, self.best_epoch, self.history = -1.0, 0, []
    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.x_val, verbose=0)
        pred = probs.argmax(axis=1)
        macro = f1_score(self.y_val, pred, average="macro", zero_division=0)
        balanced = balanced_accuracy_score(self.y_val, pred)
        self.history.append({"epoch": epoch+1, "val_macro_f1": float(macro),
                             "val_balanced_accuracy": float(balanced)})
        print(f" — val_macro_f1: {macro:.4f} — val_balanced_accuracy: {balanced:.4f}")
        score = macro + 0.001 * balanced
        if score > self.best:
            self.best, self.best_epoch = score, epoch+1
            self.model.save(self.save_path, include_optimizer=False)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Place eCommerce_Customer_support_data.csv beside train_ann8.py. Missing: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in df:
        raise ValueError(f"Missing target column: {TARGET}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    before = len(df)
    df = df[df[TARGET].isin([1,2,3,4,5])].drop_duplicates().copy()
    print(f"Rows retained: {len(df):,}; rows removed: {before-len(df):,}")
    counts = df[TARGET].value_counts().reindex([1,2,3,4,5], fill_value=0)
    print("Original class counts:", counts.to_dict())
    counts.rename_axis("CSAT Score").reset_index(name="Records").to_csv(REPORTS/"original_class_distribution.csv", index=False)

    y = df[TARGET].astype(int).to_numpy() - 1
    raw_x = df.drop(columns=[TARGET])
    x_train_raw, x_temp, y_train, y_temp = train_test_split(
        raw_x, y, test_size=0.30, random_state=SEED, stratify=y)
    x_val_raw, x_test_raw, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    print("Fitting preprocessing on original training partition only...")
    prep = make_preprocessor()
    tr_sparse = prep.fit_transform(engineer_features(x_train_raw))
    va_sparse = prep.transform(engineer_features(x_val_raw))
    te_sparse = prep.transform(engineer_features(x_test_raw))
    n_components = min(256, tr_sparse.shape[1]-1, tr_sparse.shape[0]-1)
    if n_components < 2:
        raise ValueError("Insufficient transformed features for SVD.")
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=SEED)
    x_train = svd.fit_transform(tr_sparse).astype("float32")
    x_val = svd.transform(va_sparse).astype("float32")
    x_test = svd.transform(te_sparse).astype("float32")

    # Critical: resampling happens only after the original split and train-fitted transforms.
    x_res, y_res, resampling_details = controlled_resample(x_train, y_train)
    print("Training counts before resampling:", np.bincount(y_train, minlength=5).tolist())
    print("Training counts after resampling:", np.bincount(y_res, minlength=5).tolist())
    print("Validation counts (unchanged):", np.bincount(y_val, minlength=5).tolist())
    print("Test counts (unchanged):", np.bincount(y_test, minlength=5).tolist())

    model = build_ann(x_res.shape[1])
    best_path = ARTIFACTS/"ann8_best.keras"
    checkpoint = MacroF1Checkpoint(x_val, y_val, best_path)
    callbacks = [
        checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                             min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True, verbose=1)
    ]
    hist = model.fit(x_res, y_res, validation_data=(x_val, y_val),
                     epochs=80, batch_size=256, callbacks=callbacks, verbose=1)
    pd.DataFrame(hist.history).to_csv(REPORTS/"ann8_training_history.csv", index=False)
    pd.DataFrame(checkpoint.history).to_csv(REPORTS/"ann8_validation_metrics.csv", index=False)

    best_model = tf.keras.models.load_model(best_path, compile=False)
    probs = best_model.predict(x_test, batch_size=512, verbose=0)
    pred = probs.argmax(axis=1)
    labels = list(range(5))
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_test, probs, labels=labels)),
        "predicted_class_distribution": {str(i+1): float(np.mean(pred==i)) for i in labels},
        "best_validation_macro_f1": float(max((r["val_macro_f1"] for r in checkpoint.history), default=0)),
        "best_epoch": int(checkpoint.best_epoch)
    }
    report = classification_report(y_test, pred, labels=labels,
        target_names=["1","2","3","4","5"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=labels)
    (REPORTS/"ann8_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS/"ann8_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORTS/"ann8_resampling_details.json").write_text(json.dumps(resampling_details, indent=2), encoding="utf-8")
    joblib.dump(prep, ARTIFACTS/"ann8_preprocessor.joblib")
    joblib.dump(svd, ARTIFACTS/"ann8_svd.joblib")
    best_model.save(ARTIFACTS/"ann8_model.keras", include_optimizer=False)
    metadata = {
        "model": "ANN-8 TensorFlow/Keras",
        "architecture": [256,128,64,5],
        "output_classes": [1,2,3,4,5],
        "feature_count_after_svd": int(x_train.shape[1]),
        "resampling": "controlled random oversampling and undersampling on training partition only",
        "resampling_targets": {str(k+1):v for k,v in TARGET_COUNTS.items()},
        "resampling_details": resampling_details,
        "metrics": metrics,
        "note": "Validation and test retain original class distributions; selected by validation macro-F1."
    }
    (ARTIFACTS/"ann8_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=[1,2,3,4,5],
           yticklabels=[1,2,3,4,5], xlabel="Predicted CSAT", ylabel="Actual CSAT",
           title="ANN-8 Held-out Confusion Matrix")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(REPORTS/"ann8_confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(hist.history.get("loss", []), label="Training loss")
    ax.plot(hist.history.get("val_loss", []), label="Validation loss")
    ax.set(title="ANN-8 Training History", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS/"ann8_loss_curve.png", dpi=160); plt.close(fig)

    print("\\nANN-8 HELD-OUT TEST METRICS")
    print(json.dumps(metrics, indent=2))
    print("\\nCLASSIFICATION REPORT")
    print(classification_report(y_test, pred, labels=labels,
        target_names=["1","2","3","4","5"], zero_division=0))
    print("ANN-8 artifacts and reports saved.")

if __name__ == "__main__":
    main()
