from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from train_ann8 import engineer_features

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

def load_artifacts():
    paths = [ART/"ann8_model.keras", ART/"ann8_preprocessor.joblib",
             ART/"ann8_svd.joblib", ART/"ann8_metadata.json"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("ANN-8 artifacts missing. Run python train_ann8.py first:\\n" + "\\n".join(missing))
    model = tf.keras.models.load_model(paths[0], compile=False)
    prep, svd = joblib.load(paths[1]), joblib.load(paths[2])
    meta = json.loads(paths[3].read_text(encoding="utf-8"))
    return model, prep, svd, meta

def predict_dataframe(data, batch_size=512):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame or dictionary.")
    model, prep, svd, meta = load_artifacts()
    raw = prep.transform(engineer_features(data))
    x = svd.transform(raw).astype("float32")
    if x.shape[1] != int(meta["feature_count_after_svd"]):
        raise ValueError("ANN-8 feature dimension mismatch. Keep model and preprocessing artifacts together.")
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    pred = probs.argmax(axis=1)
    out = data.copy()
    out["Predicted_CSAT"] = pred + 1
    out["Prediction_Confidence_Percent"] = probs.max(axis=1) * 100
    out["Model_Version"] = "ANN-8"
    for i in range(5):
        out[f"Probability_CSAT_{i+1}"] = probs[:, i] * 100
    return out
