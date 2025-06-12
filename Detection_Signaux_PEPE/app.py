import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import os

from backtest_report_generator import generate_html_report, save_report

# ========================
# Configuration des Paramètres
# ========================
DEFAULT_DISPLAY_PARAMS = {
    "triangles": {
        "size": 12,
        "colors": {"red": "darkred", "green": "darkgreen"},
        "y_multiplier": {"red": 1.001, "green": 0.999},
    }
}

# Symbole pour PEPE/USDT perpétuel
SYMBOL = "PEPE/USDT:USDT"
symbol = SYMBOL          # évite NameError plus bas


# Timeframes disponibles et leur durée en minutes
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
TIMEFRAME_MAPPING = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

# -------------------------------------------------
# Initialisation de l'API Bitget via ccxt
# -------------------------------------------------
# -------------------------------------------------
# Initialisation de l'API Bitget via ccxt
# -------------------------------------------------
BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET")
BITGET_API_PASSWORD = os.getenv("BITGET_API_PASSWORD")



if not (BITGET_API_KEY and BITGET_API_SECRET and BITGET_API_PASSWORD):
    raise EnvironmentError(
        "Les variables d'environnement BITGET_API_KEY, BITGET_API_SECRET et BITGET_API_PASSWORD doivent être définies."
    )

exchange = ccxt.bitget(
    {
        "enableRateLimit": True,
        "apiKey": BITGET_API_KEY,
        "secret": BITGET_API_SECRET,
        "password": BITGET_API_PASSWORD,
    }
)

# ========================
# Fonctions de Récupération et de Calcul
# ========================

def fetch_all_data(symbol, timeframe='1H', since=None, until=None, total_limit=672, batch_limit=1000):
    """
    Récupère toutes les données OHLCV en plusieurs requêtes entre since et until.
    """
    all_ohlcv = []
    fetched = 0

    while fetched < total_limit:
        remaining = total_limit - fetched
        limit = min(batch_limit, remaining)
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                print("Aucune donnée supplémentaire récupérée.")
                break
            # Filtrer les données au cas où 'until' serait atteint
            if until:
                ohlcv = [candle for candle in ohlcv if candle[0] <= until]
                if not ohlcv:
                    break
            all_ohlcv.extend(ohlcv)
            fetched += len(ohlcv)
            print(f"Récupéré {fetched} bougies sur {total_limit}.")

            # Mettre à jour le since pour la prochaine requête
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1  # Ajouter 1 ms pour éviter les doublons
            time.sleep(exchange.rateLimit / 1000)  # Respecter le rate limit
        except Exception as e:
            print(f"Erreur lors de la récupération des données OHLCV: {e}")
            break

    # Convertir en DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Supprimer les doublons éventuels
    df = df[~df.index.duplicated(keep='first')]

    print(f"Total bougies récupérées: {len(df)}")
    return df

def get_market_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"Erreur lors de la récupération du prix du marché pour {symbol}: {e}")
        return None

def calculate_ema(df, ema_lengths):
    for length in ema_lengths:
        df[f'ema_{length}'] = df['close'].ewm(span=length, adjust=False).mean()
    return df

def calculate_sum_of_gaps(df):
    ema_lengths = list(range(5, 205, 5))  # EMA5, EMA10, ..., EMA200
    for i in range(1, len(ema_lengths)):
        prev_length = ema_lengths[i-1]
        current_length = ema_lengths[i]
        df[f'gaps_{prev_length}_{current_length}'] = (df[f'ema_{prev_length}'] - df[f'ema_{current_length}']) / df['close'] * 100

    gap_columns = [f'gaps_{ema_lengths[i-1]}_{ema_lengths[i]}' for i in range(1, len(ema_lengths))]
    df['sum_of_gaps_percentage'] = df[gap_columns].sum(axis=1)
    return df

def calculate_average_slope(df, ema_lengths):
    ema_cols = [f'ema_{length}' for length in ema_lengths]
    slopes = df[ema_cols].diff()
    df['average_slope_percentage'] = slopes.mean(axis=1) / df['close'] * 100
    df['average_slope_percentage'] *= 50
    return df

def get_bar_colors_percentage(series):
    colors = ['black'] * len(series)
    for i in range(len(series)):
        value = series.iloc[i]
        if value > 0.7:
            colors[i] = 'red'
        elif value < -0.7:
            colors[i] = 'green'
    return colors

def detect_triangles(df):
    """
    Détecte les triangles rouges et verts basés sur les conditions spécifiées.
    Empêche l'ajout de deux triangles de la même couleur consécutivement.
    """
    triangles = []
    sum_of_gaps = df['sum_of_gaps_percentage']
    avg_slope = df['average_slope_percentage']
    last_triangle_color = None

    for i in range(len(df)):
        # Triangles Rouges
        if i >= 2:
            window_sum_3 = sum_of_gaps.iloc[i-2:i+1].mean()
            if window_sum_3 > 0.2:
                b1 = sum_of_gaps.iloc[i-2]
                b2 = sum_of_gaps.iloc[i-1]
                b3 = sum_of_gaps.iloc[i]
                if b1 < b2 > b3:
                    window_high = sum_of_gaps.iloc[max(0, i-5):i+1]
                    if b2 == window_high.max():
                        increase = (b2 - b1) / abs(b1) * 100 if b1 != 0 else 0
                        decrease = (b2 - b3) / abs(b2) * 100 if b2 != 0 else 0
                        if increase >= 7 and decrease >= 4 and last_triangle_color != 'red':
                            triangles.append({
                                'x': df.index[i-1],
                                'y': df['high'].iloc[i-1] * DEFAULT_DISPLAY_PARAMS['triangles']['y_multiplier']['red'],
                                'color': DEFAULT_DISPLAY_PARAMS['triangles']['colors']['red'],
                                'type': 'red'
                            })
                            last_triangle_color = 'red'
                            continue

        if i >= 2:
            window_sum_3 = sum_of_gaps.iloc[i-2:i+1].mean()
            if window_sum_3 > 1.5:
                b1 = sum_of_gaps.iloc[i-2]
                b2 = sum_of_gaps.iloc[i-1]
                b3 = sum_of_gaps.iloc[i]
                if b1 < b2 > b3:
                    window_high = sum_of_gaps.iloc[max(0, i-5):i+1]
                    if b2 == window_high.max():
                        increase = (b2 - b1) / abs(b1) * 100 if b1 != 0 else 0
                        decrease = (b2 - b3) / abs(b2) * 100 if b2 != 0 else 0
                        if increase >= 5 and decrease >= 2 and last_triangle_color != 'red':
                            triangles.append({
                                'x': df.index[i-1],
                                'y': df['high'].iloc[i-1] * DEFAULT_DISPLAY_PARAMS['triangles']['y_multiplier']['red'],
                                'color': DEFAULT_DISPLAY_PARAMS['triangles']['colors']['red'],
                                'type': 'red'
                            })
                            last_triangle_color = 'red'
                            continue

        if i >= 3:
            window_sum_4 = sum_of_gaps.iloc[i-3:i+1]
            if (window_sum_4.iloc[0] < window_sum_4.iloc[1] > window_sum_4.iloc[2] > window_sum_4.iloc[3]):
                b2 = window_sum_4.iloc[1]
                window_high = sum_of_gaps.iloc[max(0, i-7):i+1]
                if b2 == window_high.max():
                    b1 = window_sum_4.iloc[0]
                    increase = (b2 - b1) / abs(b1) * 100 if b1 != 0 else 0
                    if increase >= 7:
                        slope_window_start = max(0, i-5)
                        slope_window_end = min(len(avg_slope), i+4)
                        slope_window = avg_slope.iloc[slope_window_start:slope_window_end]
                        if len(slope_window) >= 5:
                            b1_slope = slope_window.iloc[2]
                            b2_slope = slope_window.iloc[3]
                            b3_slope = slope_window.iloc[4]
                            if b1_slope < b2_slope > b3_slope and last_triangle_color != 'red':
                                triangles.append({
                                    'x': df.index[i-2],
                                    'y': df['high'].iloc[i-2] * DEFAULT_DISPLAY_PARAMS['triangles']['y_multiplier']['red'],
                                    'color': DEFAULT_DISPLAY_PARAMS['triangles']['colors']['red'],
                                    'type': 'red'
                                })
                                last_triangle_color = 'red'
                                continue

        # Triangles Verts
        if i >= 2:
            window_sum_3 = sum_of_gaps.iloc[i-2:i+1].mean()
            if window_sum_3 < -0.2:
                b1 = sum_of_gaps.iloc[i-2]
                b2 = sum_of_gaps.iloc[i-1]
                b3 = sum_of_gaps.iloc[i]
                if b1 > b2 < b3:
                    window_low = sum_of_gaps.iloc[max(0, i-5):i+1]
                    if b2 == window_low.min():
                        decrease = (b1 - b2) / abs(b1) * 100 if b1 != 0 else 0
                        increase = (b3 - b2) / abs(b2) * 100 if b2 != 0 else 0
                        if decrease >= 7 and increase >= 4 and last_triangle_color != 'green':
                            triangles.append({
                                'x': df.index[i-1],
                                'y': df['low'].iloc[i-1] * DEFAULT_DISPLAY_PARAMS['triangles']['y_multiplier']['green'],
                                'color': DEFAULT_DISPLAY_PARAMS['triangles']['colors']['green'],
                                'type': 'green'
                            })
                            last_triangle_color = 'green'
                            continue

        if i >= 2:
            window_sum_3 = sum_of_gaps.iloc[i-2:i+1].mean()
            if window_sum_3 < -1.5:
                b1 = sum_of_gaps.iloc[i-2]
                b2 = sum_of_gaps.iloc[i-1]
                b3 = sum_of_gaps.iloc[i]
                if b1 > b2 < b3:
                    window_low = sum_of_gaps.iloc[max(0, i-5):i+1]
                    if b2 == window_low.min():
                        decrease = (b1 - b2) / abs(b1) * 100 if b1 != 0 else 0
                        increase = (b3 - b2) / abs(b2) * 100 if b2 != 0 else 0
                        if decrease >= 5 and increase >= 2 and last_triangle_color != 'green':
                            triangles.append({
                                'x': df.index[i-1],
                                'y': df['low'].iloc[i-1] * DEFAULT_DISPLAY_PARAMS['triangles']['y_multiplier']['green'],
                                'color': DEFAULT_DISPLAY_PARAMS['triangles']['colors']['green'],
                                'type': 'green'
                            })
                            last_triangle_color = 'green'
                            continue

        if i >= 3:
            window_sum_4 = sum_of_gaps.iloc[i-3:i+1]
            if (window_sum_4.iloc[0] > window_sum_4.iloc[1] < window_sum_4.iloc[2] < window_sum_4.iloc[3]):
                b2 = window_sum_4.iloc[1]
                window_low = sum_of_gaps.iloc[max(0, i-7):i+1]
                if b2 == window_low.min():
                    b1 = window_sum_4.iloc[0]
                    decrease = (b1 - b2) / abs(b1) * 100 if b1 != 0 else 0
                    if decrease >= 7:
                        slope_window_start = max(0, i-5)
                        slope_window_end = min(len(avg_slope), i+4)
                        slope_window = avg_slope.iloc[slope_window_start:slope_window_end]
                        if len(slope_window) >= 5:
                            b1_slope = slope_window.iloc[2]
                            b2_slope = slope_window.iloc[3]
                            b3_slope = slope_window.iloc[4]
                            if b1_slope > b2_slope < b3_slope and last_triangle_color != 'green':
                                triangles.append({
                                    'x': df.index[i-2],
                                    'y': df['low'].iloc[i-2] * DEFAULT_DISPLAY_PARAMS['triangles']['y_multiplier']['green'],
                                    'color': DEFAULT_DISPLAY_PARAMS['triangles']['colors']['green'],
                                    'type': 'green'
                                })
                                last_triangle_color = 'green'
                                continue

    return triangles

def backtest_strategy(df, triangles):
    balance = 100  # Solde initial
    position = None
    entry_price = 0
    trade_size = 30  # Taille du trade en USDT
    leverage = 25  # Effet de levier
    stop_loss_percent = 0.005  # Stop loss à 0.5%
    trailing_stop_percent = 0.02  # Trailing stop profit à 2%
    fee_per_trade = 0.5  # Frais par trade à la clôture

    trades = []
    balance_over_time = []
    last_triangle_color = None

    df['signal'] = 0
    triangle_indices = [df.index.get_loc(triangle['x']) for triangle in triangles]

    highest_price = 0
    lowest_price = float('inf')

    for idx in range(len(df)):
        timestamp = df.index[idx]
        close_price = df['close'].iloc[idx]
        high_price = df['high'].iloc[idx]
        low_price = df['low'].iloc[idx]

        signal = 0
        if idx in triangle_indices:
            triangle = triangles[triangle_indices.index(idx)]
            signal = 1 if triangle['type'] == 'green' else -1
            df.at[timestamp, 'signal'] = signal

        unrealized_pnl = 0
        if position == 'long':
            unrealized_pnl = (close_price - entry_price) * trade_size * leverage / entry_price
        elif position == 'short':
            unrealized_pnl = (entry_price - close_price) * trade_size * leverage / entry_price

        balance_over_time.append({'timestamp': timestamp, 'balance': balance + unrealized_pnl})

        if position == 'long':
            if high_price > highest_price:
                highest_price = high_price
            trailing_stop_price = highest_price * (1 - trailing_stop_percent)
            if low_price <= trailing_stop_price:
                exit_price = trailing_stop_price
                profit = (exit_price - entry_price) * trade_size * leverage / entry_price - fee_per_trade
                balance += profit
                trades.append({
                    'type': 'trailing_stop',
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_timestamp,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None
                highest_price = 0
                continue

            stop_loss_price = entry_price * (1 - stop_loss_percent)
            if low_price <= stop_loss_price:
                exit_price = stop_loss_price
                profit = (exit_price - entry_price) * trade_size * leverage / entry_price - fee_per_trade
                balance += profit
                trades.append({
                    'type': 'stop_loss',
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_timestamp,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None
                highest_price = 0
                continue

        elif position == 'short':
            if low_price < lowest_price:
                lowest_price = low_price
            trailing_stop_price = lowest_price * (1 + trailing_stop_percent)
            if high_price >= trailing_stop_price:
                exit_price = trailing_stop_price
                profit = (entry_price - exit_price) * trade_size * leverage / entry_price - fee_per_trade
                balance += profit
                trades.append({
                    'type': 'trailing_stop',
                    'direction': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_timestamp,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None
                lowest_price = float('inf')
                continue

            stop_loss_price = entry_price * (1 + stop_loss_percent)
            if high_price >= stop_loss_price:
                exit_price = stop_loss_price
                profit = (entry_price - exit_price) * trade_size * leverage / entry_price - fee_per_trade
                balance += profit
                trades.append({
                    'type': 'stop_loss',
                    'direction': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_timestamp,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None
                lowest_price = float('inf')
                continue

        if signal == 1 and balance >= trade_size:
            if position == 'short':
                exit_price = close_price
                profit = (entry_price - exit_price) * trade_size * leverage / entry_price - fee_per_trade
                balance += profit
                trades.append({
                    'type': 'close_short',
                    'direction': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_timestamp,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None
                lowest_price = float('inf')

            if position is None:
                entry_price = close_price
                entry_timestamp = timestamp
                position = 'long'
                highest_price = close_price
                trades.append({
                    'type': 'open_long',
                    'direction': 'long',
                    'entry_price': entry_price,
                    'timestamp_entry': entry_timestamp
                })

        elif signal == -1 and balance >= trade_size:
            if position == 'long':
                exit_price = close_price
                profit = (exit_price - entry_price) * trade_size * leverage / entry_price - fee_per_trade
                balance += profit
                trades.append({
                    'type': 'close_long',
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_timestamp,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None
                highest_price = 0

            if position is None:
                entry_price = close_price
                entry_timestamp = timestamp
                position = 'short'
                lowest_price = close_price
                trades.append({
                    'type': 'open_short',
                    'direction': 'short',
                    'entry_price': entry_price,
                    'timestamp_entry': entry_timestamp
                })

        if position == 'long' and signal == -1:
            exit_price = close_price
            profit = (exit_price - entry_price) * trade_size * leverage / entry_price - fee_per_trade
            balance += profit
            trades.append({
                'type': 'close_long',
                'direction': 'long',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'timestamp_entry': entry_timestamp,
                'timestamp_exit': timestamp,
                'profit': profit
            })
            position = None
            highest_price = 0

        elif position == 'short' and signal == 1:
            exit_price = close_price
            profit = (entry_price - exit_price) * trade_size * leverage / entry_price - fee_per_trade
            balance += profit
            trades.append({
                'type': 'close_short',
                'direction': 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'timestamp_entry': entry_timestamp,
                'timestamp_exit': timestamp,
                'profit': profit
            })
            position = None
            lowest_price = float('inf')

    return trades, balance_over_time

# ========================
# Initialisation de l'application Dash avec un thème Bootstrap
# ========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server   # <- indispensable pour 'gunicorn app:server'


# ========================
# Définition du layout de l'application
# ========================
app.layout = html.Div([
    # Stores pour partager l'état
    dcc.Store(id='xaxis-store', data={'xaxis.range[0]': None, 'xaxis.range[1]': None}),
    dcc.Store(id='display-params', data=DEFAULT_DISPLAY_PARAMS),

    html.H1(f"Données {symbol} en temps réel - Backtest", style={'textAlign': 'center'}),

    # Options interactives
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Nombre de jours à afficher :'),
                dcc.Dropdown(
                    id='days-dropdown',
                    options=[
                        {'label': '1 Jour', 'value': 1},
                        {'label': '3 Jours', 'value': 3},
                        {'label': '7 Jours', 'value': 7},
                        {'label': '14 Jours', 'value': 14},
                        {'label': '30 Jours', 'value': 30},
                        {'label': '60 Jours', 'value': 60},
                    ],
                    value=7,
                    clearable=False
                ),
            ], width=4),
            dbc.Col([
                html.Label('Timeframe des bougies :'),
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[{'label': tf, 'value': tf} for tf in AVAILABLE_TIMEFRAMES],
                    value='15m',
                    clearable=False
                ),
            ], width=4),
            dbc.Col([
                dbc.Button(
                    "Générer Rapport Détaillé",
                    id="generate-report-btn",
                    color="primary",
                    className="mt-4",
                    style={'width': '100%'}
                ),
                dcc.Download(id="download-report")
            ], width=4),
        ], justify='center', style={'padding': '10px'}),
    ]),

    # Indicateur de chargement pendant la génération du rapport
    dcc.Loading(
        id="loading-report",
        type="default",
        children=html.Div(id="loading-output")
    ),

    # Graphique principal avec les EMA et les triangles
    html.Div([
        dcc.Graph(id='live-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    # Graphique de la somme des écarts en pourcentage (brut) en barres
    html.Div([
        dcc.Graph(id='gaps-percentage-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    # Graphique de la moyenne des pentes en pourcentage (brut) en barres
    html.Div([
        dcc.Graph(id='slope-percentage-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    # Graphique de la balance au cours du temps avec la liste des trades
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='balance-graph', config={'displayModeBar': True})
            ], width=8),
            dbc.Col([
                html.H4("Liste des Trades"),
                dash_table.DataTable(
                    id='trades-table',
                    columns=[
                        {'name': 'Type', 'id': 'type'},
                        {'name': 'Direction', 'id': 'direction'},
                        {'name': 'Prix Entrée', 'id': 'entry_price'},
                        {'name': 'Prix Sortie', 'id': 'exit_price'},
                        {'name': 'Date Entrée', 'id': 'timestamp_entry'},
                        {'name': 'Date Sortie', 'id': 'timestamp_exit'},
                        {'name': 'PnL', 'id': 'profit'},
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    page_size=10
                )
            ], width=4)
        ])
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    # Interval pour les mises à jour
    dcc.Interval(
        id='interval-component',
        interval=15 * 1000,  # Mise à jour toutes les 15 minutes (900000 ms)
        n_intervals=0
    )
], style={'padding': '10px'})


# ========================
# Callback pour Générer et Télécharger le Rapport
# ========================
@app.callback(
    Output("download-report", "data"),
    Input("generate-report-btn", "n_clicks"),
    [
        State('days-dropdown', 'value'),
        State('timeframe-dropdown', 'value'),
    ],
    prevent_initial_call=True
)
def generate_and_download_report(n_clicks, selected_days, selected_timeframe):
    if not n_clicks:
        return None

    try:
        # Calcul du nombre total de bougies à récupérer
        timeframe_in_minutes = TIMEFRAME_MAPPING.get(selected_timeframe, 15)
        total_bougies = selected_days * 24 * (60 // timeframe_in_minutes)
        total_limit = total_bougies

        utc_now = datetime.now(timezone.utc)
        start_time = utc_now - timedelta(days=selected_days)
        since = int(start_time.timestamp() * 1000)
        until = int(utc_now.timestamp() * 1000)

        # Récupération des données
        df = fetch_all_data(symbol, selected_timeframe, since=since, until=until, total_limit=total_limit, batch_limit=1000)
        if df.empty:
            return None

        # Calcul des indicateurs
        ema_lengths = list(range(5, 205, 5))
        df = calculate_ema(df, ema_lengths)
        df = calculate_sum_of_gaps(df)
        df = calculate_average_slope(df, ema_lengths)
        df.fillna(0, inplace=True)

        # Détection des triangles
        triangles = detect_triangles(df)

        # Exécution du backtest
        trades, balance_over_time = backtest_strategy(df, triangles)

        # Génération du rapport HTML
        html_content = generate_html_report(
            df=df,
            trades=trades,
            balance_over_time=balance_over_time,
            triangles=triangles,
            symbol=symbol,
            timeframe=selected_timeframe,
            initial_balance=100
        )

        # Sauvegarder temporairement le rapport
        filename = f"backtest_report_{symbol.replace('/', '_')}_{selected_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        save_report(html_content, filename)

        # Retourner le fichier pour téléchargement
        return dcc.send_file(filename)

    except Exception as e:
        print(f"Erreur lors de la génération du rapport : {e}")
        return None


# Callback pour afficher l'indicateur de chargement
@app.callback(
    Output("loading-output", "children"),
    Input("generate-report-btn", "n_clicks"),
    prevent_initial_call=True
)
def display_loading(n_clicks):
    if n_clicks:
        return "Génération du rapport en cours..."
    return ""


# ========================
# Callback pour Capturer les Changements de Zoom et Mettre à Jour le Store
# ========================
@app.callback(
    Output('xaxis-store', 'data'),
    [
        Input('live-graph', 'relayoutData'),
        Input('gaps-percentage-graph', 'relayoutData'),
        Input('slope-percentage-graph', 'relayoutData'),
        Input('balance-graph', 'relayoutData')
    ],
    [State('xaxis-store', 'data')]
)
def update_store(relayout_live, relayout_gaps_percentage, relayout_slope_percentage, relayout_balance, current_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_store
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        relayoutData = None
        if triggered_id == 'live-graph':
            relayoutData = relayout_live
        elif triggered_id == 'gaps-percentage-graph':
            relayoutData = relayout_gaps_percentage
        elif triggered_id == 'slope-percentage-graph':
            relayoutData = relayout_slope_percentage
        elif triggered_id == 'balance-graph':
            relayoutData = relayout_balance

        if relayoutData and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
            return {
                'xaxis.range[0]': relayoutData['xaxis.range[0]'],
                'xaxis.range[1]': relayoutData['xaxis.range[1]']
            }
        elif relayoutData and 'xaxis.autorange' in relayoutData:
            return {'xaxis.range[0]': None, 'xaxis.range[1]': None}
        else:
            return current_store


# ========================
# Callback pour Mettre à Jour Tous les Graphiques
# ========================
@app.callback(
    [
        Output('live-graph', 'figure'),
        Output('gaps-percentage-graph', 'figure'),
        Output('slope-percentage-graph', 'figure'),
        Output('balance-graph', 'figure'),
        Output('trades-table', 'data')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('xaxis-store', 'data'),
        Input('days-dropdown', 'value'),
        Input('timeframe-dropdown', 'value'),
    ]
)
def update_graph(n, store_data, selected_days, selected_timeframe):
    try:
        # Calcul du nombre total de bougies à récupérer
        timeframe_in_minutes = TIMEFRAME_MAPPING.get(selected_timeframe, 15)
        total_bougies = selected_days * 24 * (60 // timeframe_in_minutes)
        total_limit = total_bougies

        # Calcul des timestamps pour la période sélectionnée
        utc_now = datetime.now(timezone.utc)
        start_time = utc_now - timedelta(days=selected_days)
        since = int(start_time.timestamp() * 1000)
        until = int(utc_now.timestamp() * 1000)

        # Récupération des données pour la période sélectionnée
        df = fetch_all_data(symbol, selected_timeframe, since=since, until=until, total_limit=total_limit, batch_limit=1000)
        current_price = get_market_price(symbol)

        if df.empty or current_price is None:
            return [go.Figure()] * 5

        # Calcul des indicateurs
        ema_lengths = list(range(5, 205, 5))
        df = calculate_ema(df, ema_lengths)
        df = calculate_sum_of_gaps(df)
        df = calculate_average_slope(df, ema_lengths)
        df.fillna(0, inplace=True)

        # Détection des Triangles
        triangles = detect_triangles(df)

        # Exécution du backtest
        trades, balance_over_time = backtest_strategy(df, triangles)

        # Graphique principal avec les EMA
        fig_live = go.Figure()
        fig_live.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=False,
        ))
        colors = [
            'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'grey', 'olive', 'cyan',
            'darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkcyan', 'black',
            'magenta', 'gold', 'darkgrey', 'lightblue', 'lightgreen', 'lightpink', 'lightgrey',
            'teal', 'navy', 'maroon', 'lime', 'coral', 'indigo', 'turquoise', 'tan', 'salmon',
            'orchid', 'chocolate', 'plum', 'skyblue', 'darkolivegreen', 'firebrick', 'sienna',
            'slateblue', 'aquamarine'
        ]
        for i, length in enumerate(ema_lengths):
            fig_live.add_trace(go.Scatter(
                x=df.index,
                y=df[f'ema_{length}'],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=1),
                name=f'EMA {length}',
                hoverinfo='none'
            ))
        if triangles:
            for triangle in triangles:
                fig_live.add_trace(go.Scatter(
                    x=[triangle['x']],
                    y=[triangle['y']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if triangle['type'] == 'red' else 'triangle-down',
                        color=triangle['color'],
                        size=DEFAULT_DISPLAY_PARAMS['triangles']['size']
                    ),
                    name='Signal Triangle',
                    showlegend=False
                ))
        fig_live.update_layout(
            height=800,
            title_text=f"{symbol} - Prix actuel: {current_price:.9f} USDT",
            xaxis_rangeslider_visible=False,
            yaxis_title='',
            xaxis_title='',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            uirevision='constant',
            showlegend=False
        )
        if store_data['xaxis.range[0]'] and store_data['xaxis.range[1]']:
            fig_live.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_live.update_xaxes(range=[df.index[0], df.index[-1]])

        # Graphique de la somme des écarts en pourcentage (brut) en barres
        fig_gaps_percentage = go.Figure()
        colors_gaps_percentage = get_bar_colors_percentage(df['sum_of_gaps_percentage'])
        fig_gaps_percentage.add_trace(go.Bar(
            x=df.index,
            y=df['sum_of_gaps_percentage'],
            marker_color=colors_gaps_percentage,
            name='Somme des Écarts EMA (%)',
            hoverinfo='text',
            hovertext=[f"Date: {x}<br>Somme des Écarts (%): {y:.4f}" for x, y in zip(df.index, df['sum_of_gaps_percentage'])],
            showlegend=False
        ))
        fig_gaps_percentage.update_layout(
            height=300,
            title_text='Somme des Écarts EMA en Pourcentage',
            xaxis_title='',
            yaxis_title='Pourcentage (%)',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            showlegend=False
        )
        if store_data['xaxis.range[0]'] and store_data['xaxis.range[1]']:
            fig_gaps_percentage.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_gaps_percentage.update_xaxes(range=[df.index[0], df.index[-1]])

        # Graphique de la moyenne des pentes en pourcentage (brut) en barres
        fig_slope_percentage = go.Figure()
        colors_slope_percentage = get_bar_colors_percentage(df['average_slope_percentage'])
        fig_slope_percentage.add_trace(go.Bar(
            x=df.index,
            y=df['average_slope_percentage'],
            marker_color=colors_slope_percentage,
            name='Moyenne des Pentes (%)',
            hoverinfo='text',
            hovertext=[f"Date: {x}<br>Moyenne des Pentes (%): {y:.4f}" for x, y in zip(df.index, df['average_slope_percentage'])],
            showlegend=False
        ))
        fig_slope_percentage.update_layout(
            height=300,
            title_text='Moyenne des Pentes en Pourcentage (×50)',
            xaxis_title='',
            yaxis_title='Pourcentage (%)',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            showlegend=False
        )
        if store_data['xaxis.range[0]'] and store_data['xaxis.range[1]']:
            fig_slope_percentage.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_slope_percentage.update_xaxes(range=[df.index[0], df.index[-1]])

        # Graphique de la balance au cours du temps
        fig_balance = go.Figure()
        balance_df = pd.DataFrame(balance_over_time)
        fig_balance.add_trace(go.Scatter(
            x=balance_df['timestamp'],
            y=balance_df['balance'],
            mode='lines',
            name='Balance',
            line=dict(color='blue')
        ))
        for trade in trades:
            if 'timestamp_exit' in trade and trade['timestamp_exit'] in balance_df['timestamp'].values:
                timestamp = trade['timestamp_exit']
                balance_point = balance_df.loc[balance_df['timestamp'] == timestamp, 'balance'].values[0]
                color = 'green' if trade['profit'] > 0 else 'red'
                fig_balance.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[balance_point],
                    mode='markers+text',
                    marker=dict(color=color, size=10),
                    text=[f"PnL: {trade['profit']:.2f}"],
                    textposition='top center',
                    showlegend=False
                ))
        fig_balance.update_layout(
            height=400,
            title_text='Évolution de la Balance au Cours du Temps',
            xaxis_title='Temps',
            yaxis_title='Balance (USDT)',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            showlegend=False
        )
        if store_data['xaxis.range[0]'] and store_data['xaxis.range[1]']:
            fig_balance.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_balance.update_xaxes(range=[balance_df['timestamp'].iloc[0], balance_df['timestamp'].iloc[-1]])

        # Préparer les données pour la table des trades
        trades_table_data = []
        for trade in trades:
            trade_data = {
                'type': trade.get('type', ''),
                'direction': trade.get('direction', ''),
                'entry_price': f"{trade.get('entry_price', 0):.9f}" if 'entry_price' in trade else '',
                'exit_price': f"{trade.get('exit_price', 0):.9f}" if 'exit_price' in trade else '',
                'timestamp_entry': trade.get('timestamp_entry', '').strftime('%Y-%m-%d %H:%M') if 'timestamp_entry' in trade else '',
                'timestamp_exit': trade.get('timestamp_exit', '').strftime('%Y-%m-%d %H:%M') if 'timestamp_exit' in trade else '',
                'profit': f"{trade.get('profit', 0):.2f}" if 'profit' in trade else ''
            }
            trades_table_data.append(trade_data)

        return (
            fig_live,
            fig_gaps_percentage,
            fig_slope_percentage,
            fig_balance,
            trades_table_data
        )
    except Exception as e:
        print(f"Erreur dans update_graph : {e}")
        return [go.Figure()] * 5


# Lancement (local) -------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    print(f"Application en cours d'exécution sur http://0.0.0.0:{port}")
    app.run_server(debug=False, host="0.0.0.0", port=port)
