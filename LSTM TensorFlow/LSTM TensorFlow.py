#!/usr/bin/env python
# coding: utf-8
"""
BTC-USDT 1 min • LSTM bi-directionnel
Optimisé : XLA, mixed-precision, timing des folds
Affiche un rapport détaillé directement en console.
"""

# ───────────── IMPORTS ─────────────
from pathlib import Path
import numpy as np, pandas as pd, time, datetime as dt, platform, os, sys, talib, joblib
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# ——— Accélérations TensorFlow ———
tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

# ─────────── PARAMÈTRES —──────────
CSV       = Path(r"C:\Users\nicol\Desktop\Python API\BTC_Historical_Data\1m_BTC_Candles\BTC_2023\btc_usdt_february_2023.csv")
OUT       = Path(r"C:\Users\nicol\Desktop\models_btc_lstm_v3"); OUT.mkdir(parents=True, exist_ok=True)

LOOKBACK    = 25        # 25 minutes d’historique
SPLITS      = 5         # validation croisée walk-forward
BATCH       = 300       # batch size
EPOCHS      = 30        # max epochs
PATIENCE    = 3         # patience early stopping
LR          = 1e-3      # learning rate
CLIP_NORM   = 1.0       # clip gradient

# ───── 1. CHARGEMENT & FEATURES ─────
df = pd.read_csv(CSV, usecols=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.sort_values("timestamp", inplace=True)

# Features temporelles cycliques
df["hour"]   = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["dow"]    = df["timestamp"].dt.dayofweek
df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
df["minute_sin"] = np.sin(2*np.pi*df["minute"]/60)
df["minute_cos"] = np.cos(2*np.pi*df["minute"]/60)
df["dow_sin"]    = np.sin(2*np.pi*df["dow"]/7)
df["dow_cos"]    = np.cos(2*np.pi*df["dow"]/7)

# Lags
df["ret1"] = df["close"].pct_change()
df["ret5"] = df["close"].pct_change(5)

# Indicateurs TA-Lib
c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
df["EMA_10"]  = talib.EMA(c, 10)
df["RSI_14"]  = talib.RSI(c, 14)
df["ATR_14"]  = talib.ATR(h, l, c, 14)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Corrélation > 0.9 supprimée
feature_cols = [col for col in df.columns if col not in ("timestamp", "close")]
corr = df[feature_cols].corr().abs()
upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > .9)]
df.drop(columns=to_drop, inplace=True)
feature_cols = [c for c in feature_cols if c not in to_drop]

# ——— 2. SCALING ———
scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
X_all = scaler_X.fit_transform(df[feature_cols]).astype(np.float32)
y_all = scaler_y.fit_transform(df[["close"]]).astype(np.float32).ravel()

def make_dataset(X, y, lookback):
    X_out, y_out = [], []
    for i in range(len(X)-lookback):
        X_out.append(X[i:i+lookback]); y_out.append(y[i+lookback])
    return np.asarray(X_out), np.asarray(y_out)

X, y = make_dataset(X_all, y_all, LOOKBACK)

# ——— 3. MODELE LSTM ———
def build_model():
    model = Sequential([
        Input(shape=X.shape[1:]),
        Bidirectional(LSTM(64, recurrent_dropout=.2, return_sequences=True)),
        BatchNormalization(), Dropout(.3),
        Bidirectional(LSTM(64, recurrent_dropout=.2)),
        BatchNormalization(), Dropout(.3),
        Dense(64, activation="relu"),
        Dense(1, dtype="float32")
    ])
    model.compile(Adam(LR, clipnorm=CLIP_NORM), loss=Huber(), metrics=["mae"])
    return model

# ——— 4. ENTRAINEMENT WALK-FORWARD ———
tscv = TimeSeriesSplit(n_splits=SPLITS)
metrics, durations, loss_histories, preview_preds = [], [], [], []
fold = 1

total_start = time.perf_counter()
print(f"\n===== Entraînement LSTM BTC 1min ({SPLITS} folds) =====")

for train_idx, val_idx in tscv.split(X):
    print(f"\n🔹 Fold {fold}/{SPLITS} — train {len(train_idx)} val {len(val_idx)}")
    model = build_model()
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=2, factor=.5, min_lr=1e-5, monitor="val_loss")
    ]
    t0 = time.perf_counter()
    hist = model.fit(
        X[train_idx], y[train_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=False,
        callbacks=cb,
        verbose=2
    )
    fold_time = time.perf_counter() - t0
    durations.append(fold_time)

    # KPIs
    preds = scaler_y.inverse_transform(model.predict(X[val_idx])).ravel()
    reals = scaler_y.inverse_transform(y[val_idx, None]).ravel()
    mae  = np.mean(np.abs(preds - reals))
    rmse = np.sqrt(np.mean((preds - reals)**2))
    acc  = (np.sign(np.diff(preds)) == np.sign(np.diff(reals))).mean()
    metrics.append((mae, rmse, acc))
    print(f"⏱️  {fold_time:.1f}s  |  MAE:{mae:.2f}  RMSE:{rmse:.2f}  DirAcc:{acc:.2%}")
    # Prévisualisation : premiers et derniers exemples
    preview_preds.append({
        "preds": preds[:3].round(2).tolist() + ["..."] + preds[-3:].round(2).tolist(),
        "reals": reals[:3].round(2).tolist() + ["..."] + reals[-3:].round(2).tolist()
    })
    fold += 1

total_time = time.perf_counter() - total_start

# ——— 5. EXPORT SCALERS & KPI (optionnel) ———
joblib.dump(scaler_X, OUT / "scaler_X.save")
joblib.dump(scaler_y, OUT / "scaler_y.save")
np.save(OUT / "cv_metrics.npy", np.array(metrics))
np.save(OUT / "fold_times.npy", np.array(durations))

# ——— 6. RAPPORT CONSOLE ———
print("\n========== RAPPORT DÉTAILLÉ ==========\n")
print(f"Date               : {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Total duration     : {total_time/60:.1f} min")
print(f"Dataset shape      : {df.shape}")
print(f"Lookback window    : {LOOKBACK} min ({LOOKBACK} pas)")
print(f"Features retenues  : {feature_cols}")
print(f"Batch size         : {BATCH}")
print(f"Splits (folds)     : {SPLITS}")
print(f"Epochs (max)       : {EPOCHS}")
print(f"Patience           : {PATIENCE}")
print(f"Learning rate      : {LR}")
print(f"Clipnorm           : {CLIP_NORM}")
print("\n—— KPIs par fold ———————————")
print("Fold |    MAE     |   RMSE    | Dir. Acc |   Durée")
print("-" * 49)
for i, (mae, rmse, acc) in enumerate(metrics):
    print(f"{i+1:>4} | {mae:9.2f} | {rmse:8.2f} | {acc:8.2%} | {durations[i]/60:6.1f} min")
print("-" * 49)
means = np.mean(metrics, axis=0)
print(f" Moy | {means[0]:9.2f} | {means[1]:8.2f} | {means[2]:8.2%} | {np.mean(durations)/60:6.1f} min")
print("\n—— Exemples de prédictions par fold ————")
for i, ex in enumerate(preview_preds, 1):
    print(f"Fold {i}: Preds: {ex['preds']} | Reals: {ex['reals']}")
print("\n✅  Entraînement terminé !")
