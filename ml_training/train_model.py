"""
Fall Detection Model Training
==============================
Supports three datasets:
  - kaggle    : https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data
                Columns: Date;Timestamp;DeviceOrientation;AccelerationX;AccelerationY;AccelerationZ;Label
                Folders: downSit / freeFall / runFall / runSit / walkFall / walkSit
  - sisfall   : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/  (requires request)
  - mobifall  : https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/ (requires request)

Usage:
  pip install -r requirements.txt
  python train_model.py --dataset kaggle --data_dir ./data/kaggle_fall
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import json

# ── Configuration ────────────────────────────────────────────────────────────

WINDOW_SIZE   = 100   # samples per window (~2 sec at 50 Hz)
STEP_SIZE     = 50    # 50% overlap
SAMPLE_RATE   = 50    # Hz

# SisFall fall activity codes (F01–F15 are falls, D01–D19 are ADLs)
SISFALL_FALL_CODES = {f"F{i:02d}" for i in range(1, 16)}

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_sisfall(data_dir: str) -> pd.DataFrame:
    """
    Load SisFall dataset.
    Expects folder structure:  data_dir/SA01/F01/SA01_F01_R01.txt  etc.
    Each file: 5 columns → ADXL345_x, ADXL345_y, ADXL345_z, ITG3200_x, ITG3200_y (no z gyro in v1)
    """
    records = []
    data_path = Path(data_dir)

    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        for activity_dir in sorted(subject_dir.iterdir()):
            if not activity_dir.is_dir():
                continue
            activity_code = activity_dir.name          # e.g. F01, D05
            is_fall = activity_code in SISFALL_FALL_CODES
            for trial_file in sorted(activity_dir.glob("*.txt")):
                try:
                    df = pd.read_csv(trial_file, header=None,
                                     names=["acc_x","acc_y","acc_z","gyr_x","gyr_y"])
                    df["label"]    = int(is_fall)
                    df["activity"] = activity_code
                    df["subject"]  = subject_dir.name
                    records.append(df)
                except Exception as e:
                    print(f"  Skipping {trial_file}: {e}")

    if not records:
        raise FileNotFoundError(f"No SisFall data found in {data_dir}. "
                                 "Download from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/")
    return pd.concat(records, ignore_index=True)


def load_mobifall(data_dir: str) -> pd.DataFrame:
    """
    Load MobiFall dataset.
    Falls: FOL, FKL, BSC, SDL  |  ADLs: STD, WAL, JOG, JUM, STU, STN, SCH, CSI, CSO
    Each file has columns: timestamp, acc_x, acc_y, acc_z (accelerometer files)
    """
    FALL_TYPES = {"FOL", "FKL", "BSC", "SDL"}
    records = []
    data_path = Path(data_dir)

    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        for activity_dir in sorted(subject_dir.iterdir()):
            if not activity_dir.is_dir():
                continue
            activity_code = activity_dir.name
            is_fall = activity_code in FALL_TYPES
            for acc_file in sorted(activity_dir.glob("*_acc.txt")):
                base = acc_file.stem.replace("_acc", "")
                gyr_file = acc_file.parent / f"{base}_gyro.txt"
                try:
                    acc = pd.read_csv(acc_file, header=None,
                                      names=["ts","acc_x","acc_y","acc_z"])
                    if gyr_file.exists():
                        gyr = pd.read_csv(gyr_file, header=None,
                                          names=["ts","gyr_x","gyr_y","gyr_z"])
                        df = pd.concat([acc[["acc_x","acc_y","acc_z"]],
                                        gyr[["gyr_x","gyr_y","gyr_z"]]], axis=1)
                    else:
                        df = acc[["acc_x","acc_y","acc_z"]].copy()
                        df["gyr_x"] = df["gyr_y"] = df["gyr_z"] = 0.0
                    df["label"]    = int(is_fall)
                    df["activity"] = activity_code
                    df["subject"]  = subject_dir.name
                    records.append(df)
                except Exception as e:
                    print(f"  Skipping {acc_file}: {e}")

    if not records:
        raise FileNotFoundError(f"No MobiFall data found in {data_dir}. "
                                 "Download from https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/")
    return pd.concat(records, ignore_index=True)

def load_kaggle(data_dir: str) -> pd.DataFrame:
    """
    Load Kaggle fall detection dataset.

    Expected folder structure:
        data_dir/
            downSit/   downsit1.csv, downsit2.csv, ...   -> label 0 (non-fall)
            runSit/    ...                               -> label 0
            walkSit/   ...                               -> label 0
            freeFall/  ...                               -> label 1 (fall)
            runFall/   ...                               -> label 1
            walkFall/  ...                               -> label 1

    CSV columns (semicolon-separated):
        Date;Timestamp;DeviceOrientation;AccelerationX;AccelerationY;AccelerationZ;Label
    """
    FALL_FOLDERS     = {"freefall", "runfall", "walkfall"}
    NON_FALL_FOLDERS = {"downsit", "runsit", "walksit"}

    records   = []
    data_path = Path(data_dir)

    folders = [f for f in sorted(data_path.iterdir()) if f.is_dir()]
    if not folders:
        raise FileNotFoundError(
            f"No subfolders found in {data_dir}.\n"
            "Expected: downSit/, freeFall/, runFall/, runSit/, walkFall/, walkSit/"
        )

    for folder in folders:
        folder_key = folder.name.lower()
        if folder_key in FALL_FOLDERS:
            is_fall = 1
        elif folder_key in NON_FALL_FOLDERS:
            is_fall = 0
        else:
            print(f"  Warning: Unknown folder '{folder.name}' -- skipping")
            continue

        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            print(f"  Warning: No CSV files in {folder.name} -- skipping")
            continue

        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file, sep=";", low_memory=False)
                df.columns = [c.strip().lower() for c in df.columns]
                df = df.rename(columns={
                    "accelerationx": "acc_x",
                    "accelerationy": "acc_y",
                    "accelerationz": "acc_z",
                })
                df = df[["acc_x", "acc_y", "acc_z"]].copy()
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                if df.empty:
                    continue
                # No gyroscope in this dataset -> fill with zeros
                df["gyr_x"] = 0.0
                df["gyr_y"] = 0.0
                df["gyr_z"] = 0.0
                df["label"]    = is_fall
                df["activity"] = folder.name
                df["subject"]  = csv_file.stem
                records.append(df)
            except Exception as e:
                print(f"  Skipping {csv_file.name}: {e}")

    if not records:
        raise FileNotFoundError(f"No valid CSV data loaded from {data_dir}.")

    combined = pd.concat(records, ignore_index=True)
    print(f"   Kaggle dataset: {len(combined):,} samples  "
          f"| falls={combined['label'].sum():,}  "
          f"non-falls={(combined['label']==0).sum():,}")
    return combined



# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_smv(ax, ay, az):
    """Signal Magnitude Vector"""
    return np.sqrt(ax**2 + ay**2 + az**2)


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 30 statistical + frequency features from a sensor window.
    Input shape: (WINDOW_SIZE, 6)  → [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    """
    features = []
    for col in range(window.shape[1]):
        sig = window[:, col]
        features += [
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig),
            np.max(sig) - np.min(sig),          # range
            np.percentile(sig, 25),
            np.percentile(sig, 75),
        ]
    # SMV features (accelerometer only)
    smv = compute_smv(window[:,0], window[:,1], window[:,2])
    features += [
        np.max(smv),
        np.mean(smv),
        np.std(smv),
        float(np.argmax(smv)) / len(smv),       # normalised peak position
    ]
    return np.array(features, dtype=np.float32)


def sliding_windows(df: pd.DataFrame):
    """Convert raw sensor dataframe → (X_features, y_labels)"""
    sensor_cols = ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]
    # Ensure all columns exist
    for col in sensor_cols:
        if col not in df.columns:
            df[col] = 0.0

    X, y = [], []
    data   = df[sensor_cols].values
    labels = df["label"].values

    for start in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[start:start + WINDOW_SIZE]
        # Label = 1 if any sample in window is a fall
        label  = int(labels[start:start + WINDOW_SIZE].max())
        X.append(extract_window_features(window))
        y.append(label)

    return np.array(X), np.array(y)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


# ── TFLite Conversion ──────────────────────────────────────────────────────────

def convert_to_tflite(model: tf.keras.Model, output_path: str,
                      X_sample: np.ndarray) -> None:
    """Convert Keras model → quantised TFLite model for Android."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full-integer quantisation (smaller, faster on mobile)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for i in range(min(200, len(X_sample))):
            yield [X_sample[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"\n✅ TFLite model saved → {output_path}  ({size_kb:.1f} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train fall detection model")
    parser.add_argument("--dataset",  choices=["kaggle","sisfall","mobifall"], default="kaggle")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root")
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--output",   default="../android_app/app/src/main/assets")
    args = parser.parse_args()

    print(f"\n📂 Loading {args.dataset} from {args.data_dir} …")
    loaders = {"kaggle": load_kaggle, "sisfall": load_sisfall, "mobifall": load_mobifall}
    loader = loaders[args.dataset]
    df = loader(args.data_dir)
    print(f"   Loaded {len(df):,} raw samples | "
          f"falls={df['label'].sum():,}  non-falls={(df['label']==0).sum():,}")

    print("\n🪟 Creating sliding windows …")
    X, y = sliding_windows(df)
    print(f"   Windows: {len(X):,}  |  fall={y.sum()}  non-fall={(y==0).sum()}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Save scaler params for Android
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(X.shape[1])
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"   Scaler params saved → {out_dir}/scaler_params.json")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n🧠 Training model (epochs={args.epochs}) …")
    model = build_model(X_train.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n📊 Evaluation on test set:")
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred, target_names=["Normal","Fall"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    tflite_path = str(out_dir / "fall_detection.tflite")
    convert_to_tflite(model, tflite_path, X_train[:200])

    # Also save full Keras model for further fine-tuning
    model.save(str(out_dir / "fall_detection_keras.h5"))
    print(f"✅ Keras model saved → {out_dir}/fall_detection_keras.h5")


if __name__ == "__main__":
    main()