import pandas as pd
import numpy as np
import talib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Charger les données (remplacer le chemin par vos données)
data = pd.read_csv(r'C:\Users\nicol\Desktop\Python API\BTC_Historical_Data\5m_BTC_Candles\btc_usdt_jan_to_aug_2024_5min.csv')

# Sélectionner les colonnes pertinentes
data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 2. Calculer les indicateurs techniques
def calculate_indicators(df):
    # Moyenne mobile exponentielle (EMA)
    df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
    
    # Bandes de Bollinger
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    
    # ADX (Average Directional Movement Index)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=10)
    
    # RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    
    # OBV (On-Balance Volume)
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    
    # ATR (Average True Range)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=10)
    
    # Moyenne mobile du volume
    df['Volume_Mean'] = df['volume'].rolling(window=20).mean()

    # MFI (Money Flow Index)
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

    # Vortex Indicator (VI)
    df['Vortex_Pos'] = talib.PLUS_DM(df['high'], df['low'], timeperiod=10)
    df['Vortex_Neg'] = talib.MINUS_DM(df['high'], df['low'], timeperiod=10)

    # Chaikin Money Flow (CMF)
    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    # Stochastic RSI (STOCH RSI)
    df['StochRSI'], _ = talib.STOCHRSI(df['close'], timeperiod=10)

    # CCI (Commodity Channel Index)
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=10)

    # MACD (Moving Average Convergence Divergence)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Williams %R
    df['Williams_%R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=10)

    df.fillna(0, inplace=True)  # Remplir les valeurs manquantes avec 0
    return df

# Appliquer les indicateurs techniques
data = calculate_indicators(data)

# Normalisation des données (ajouter tous les indicateurs au processus de normalisation)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume',
                                         'EMA_10', 'BB_upper', 'BB_middle', 'BB_lower', 'ADX', 'RSI', 'OBV', 
                                         'ATR', 'Volume_Mean', 'MFI', 'Vortex_Pos', 'Vortex_Neg', 
                                         'CMF', 'StochRSI', 'CCI', 'MACD', 'MACD_Signal', 'Williams_%R']])

# 3. Fonction pour créer les ensembles de données pour le LSTM
def create_lstm_dataset(data, time_step=10):  
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, :])
        y.append(data[i + time_step, 3])  # Prédiction de la colonne 'close' (indice 3)
    return np.array(X), np.array(y)

# 4. Liste des tailles de fenêtres temporelles à tester
window_sizes = [10, 20, 50, 100]

# 5. Boucle sur les différentes tailles de fenêtres
for window_size in window_sizes:
    print(f"Entraînement du modèle avec une fenêtre de {window_size} périodes...")
    
    # Créer les ensembles de données pour la taille de fenêtre actuelle
    X_train, y_train = create_lstm_dataset(scaled_data, window_size)
    
    # 6. Création du modèle LSTM
    model = Sequential()
    
    # Première couche LSTM avec régularisation L2
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Deuxième couche LSTM
    model.add(LSTM(units=50, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Couche Dense
    model.add(Dense(units=50, kernel_regularizer=l2(0.001)))  
    model.add(Dense(units=1))
    
    # 7. Compilation du modèle
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # 8. Callbacks pour l'entraînement
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    
    # 9. Entraînement du modèle
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # 10. Sauvegarder le modèle avec un nom spécifique
    model_filename = rf'C:\Users\nicol\Desktop\modele_lstm_entrainement_5min_window_{window_size}.h5'
    model.save(model_filename)
    print(f"Modèle sauvegardé sous le nom : {model_filename}")
    
    # 11. Tracer la courbe de perte
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title(f'Courbe de perte pour une fenêtre de {window_size} périodes')
    plt.xlabel('Epochs')
    plt.ylabel('Perte (Loss)')
    plt.legend()
    plt.show()
