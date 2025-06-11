import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
from backtest_report_generator_macd import generate_html_report, save_report

# ========================
# Configuration des Paramètres
# ========================
BASE_TRADE_SIZE = 30       # Taille de base en USDT
LEVERAGE = 50              # Effet de levier
FEE_RATE = 0.0001          # 0,01% par opération (round-trip inclus)
PIVOT_WINDOW = 20          # Nombre de bougies pour calculer pivots
ATR_WINDOW = 14            # Période pour l’ATR

ZONE_BULLISH = "Bullish"
ZONE_BEARISH = "Bearish"
ZONE_NEUTRAL = "Neutral"

# Mapping des timeframes à l’alias Pandas pour resampling
RESAMPLE_MAPPING = {
    '1m': '1T',
    '2m': '2T',
    '5m': '5T',
    '15m': '15T',
    '30m': '30T',
    '1h': '1H',
    '2h': '2H',
    '12h': '12H',
    '1d': '1D',
    '1w': '1W'
}

# Fichier CSV local (30m BTC/USDT)
CSV_PATH = r"C:\Users\nicol\Desktop\Python API\BTC_Historical_Data\30m_BTC_Candles\btc_usdt_30min_2024.csv"

# ========================
# Lecture des données CSV
# ========================
df_raw = pd.read_csv(
    CSV_PATH,
    parse_dates=['timestamp'],
    infer_datetime_format=True
)
df_raw.set_index('timestamp', inplace=True)
df_raw.sort_index(inplace=True)

# ========================
# Fonctions d'Indicatrices
# ========================
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    df['ema_26'] = calculate_ema(df['close'], span=26)
    df['ema_30'] = calculate_ema(df['close'], span=30)
    df['macd'] = df['ema_26'] - df['ema_30']
    df['signal_line'] = calculate_ema(df['macd'], span=20)
    return df

def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

def calculate_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    tr = calculate_true_range(df)
    return tr.rolling(window=window, min_periods=1).mean()

def calculate_pivots(df: pd.DataFrame, window: int = PIVOT_WINDOW) -> Tuple[pd.Series, pd.Series]:
    pivot_high = df['high'].rolling(window=window, min_periods=1).max().shift(1)
    pivot_low = df['low'].rolling(window=window, min_periods=1).min().shift(1)
    return pivot_high, pivot_low

def compute_zones(df: pd.DataFrame) -> pd.DataFrame:
    df['atr'] = calculate_atr(df, window=ATR_WINDOW)
    pivot_highs, pivot_lows = calculate_pivots(df, window=PIVOT_WINDOW)
    df['pivot_high'] = pivot_highs
    df['pivot_low'] = pivot_lows
    df['upper_line'] = df['pivot_high'] + df['atr']
    df['lower_line'] = df['pivot_low'] - df['atr']

    def classify_zone(row):
        price = row['close']
        if pd.isna(row['upper_line']) or pd.isna(row['lower_line']):
            return ZONE_NEUTRAL
        if price > row['upper_line']:
            return ZONE_BULLISH
        elif price < row['lower_line']:
            return ZONE_BEARISH
        else:
            return ZONE_NEUTRAL

    df['zone'] = df.apply(classify_zone, axis=1)
    return df

def detect_triangles(df: pd.DataFrame) -> List[dict]:
    """
    Fonction de détection de triangles. Retourne une liste de dictionnaires
    avec clés : 'x', 'y', 'color', 'type'.
    Implémentation basique : on retourne une liste vide pour l'instant.
    """
    return []

# ========================
# Fonction de Backtest
# ========================
def backtest_strategy(df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    balance = 100.0
    position = None          # 'long' ou 'short' ou None
    entry_price = 0.0
    entry_time = None
    trades: List[dict] = []
    balance_over_time: List[dict] = []

    df['macd_diff'] = df['macd'] - df['signal_line']
    df['macd_diff_prev'] = df['macd_diff'].shift(1)

    for idx in range(1, len(df)):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        timestamp = df.index[idx]
        close = row['close']
        zone = row['zone']

        # Calcul PnL latent si position ouverte
        unrealized_pnl = 0.0
        if position == 'long':
            unrealized_pnl = (close - entry_price) * trade_size * LEVERAGE / entry_price
        elif position == 'short':
            unrealized_pnl = (entry_price - close) * trade_size * LEVERAGE / entry_price
        balance_over_time.append({'timestamp': timestamp, 'balance': balance + unrealized_pnl})

        # Croisements MACD
        cross_up = (prev['macd_diff'] < 0) and (row['macd_diff'] > 0)
        cross_down = (prev['macd_diff'] > 0) and (row['macd_diff'] < 0)

        # Taille du trade selon zone
        if zone == ZONE_BULLISH:
            trade_size = BASE_TRADE_SIZE * 1.5
        elif zone == ZONE_BEARISH:
            trade_size = BASE_TRADE_SIZE * 0.5
        else:
            trade_size = BASE_TRADE_SIZE

        # Entrée Long / clôture Short
        if cross_up and (zone != ZONE_BEARISH):
            if position == 'short':
                exit_price = close
                profit = (entry_price - exit_price) * trade_size * LEVERAGE / entry_price
                fee = trade_size * FEE_RATE
                profit -= fee
                balance += profit
                trades.append({
                    'type': 'close_short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_time,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None

            if position is None:
                entry_price = close
                entry_time = timestamp
                position = 'long'
                trades.append({
                    'type': 'open_long',
                    'entry_price': entry_price,
                    'timestamp_entry': timestamp
                })

        # Entrée Short / clôture Long
        elif cross_down and (zone != ZONE_BULLISH):
            if position == 'long':
                exit_price = close
                profit = (exit_price - entry_price) * trade_size * LEVERAGE / entry_price
                fee = trade_size * FEE_RATE
                profit -= fee
                balance += profit
                trades.append({
                    'type': 'close_long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'timestamp_entry': entry_time,
                    'timestamp_exit': timestamp,
                    'profit': profit
                })
                position = None

            if position is None:
                entry_price = close
                entry_time = timestamp
                position = 'short'
                trades.append({
                    'type': 'open_short',
                    'entry_price': entry_price,
                    'timestamp_entry': timestamp
                })

    return trades, balance_over_time

# ========================
# Initialisation de l'application Dash
# ========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Options pour la sélection de la période (en jours)
DAYS_OPTIONS = [
    {'label': '1 Jour', 'value': 1},
    {'label': '3 Jours', 'value': 3},
    {'label': '7 Jours', 'value': 7},
    {'label': '14 Jours', 'value': 14},
    {'label': '30 Jours', 'value': 30},
    {'label': '60 Jours', 'value': 60},
    {'label': '1 An', 'value': 365},
    {'label': '4 Ans', 'value': 365 * 4},
]

# Timeframes disponibles
AVAILABLE_TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '1h', '2h', '12h', '1d', '1w']

# Layout
app.layout = html.Div([
    dcc.Store(id='xaxis-store', data={'xaxis.range[0]': None, 'xaxis.range[1]': None}),

    html.H1("Backtest BTC/USDT - MACD + Zones", style={'textAlign': 'center'}),

    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Nombre de jours à afficher :'),
                dcc.Dropdown(
                    id='days-dropdown',
                    options=DAYS_OPTIONS,
                    value=7,
                    clearable=False
                ),
            ], width=4),
            dbc.Col([
                html.Label('Timeframe des bougies :'),
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[{'label': tf, 'value': tf} for tf in AVAILABLE_TIMEFRAMES],
                    value='30m',
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

    dcc.Loading(
        id="loading-report",
        type="default",
        children=html.Div(id="loading-output")
    ),

    # Graphique Prix
    html.Div([
        dcc.Graph(id='price-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    # Graphique MACD + Signal
    html.Div([
        dcc.Graph(id='macd-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    # Graphique Balance + Table des trades
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

    # Intervalle de mise à jour
    dcc.Interval(
        id='interval-component',
        interval=15 * 60 * 1000,  # 15 minutes
        n_intervals=0
    )
], style={'padding': '10px'})

# ========================
# Callback Génération Rapport
# ========================
@app.callback(
    Output("download-report", "data"),
    Input("generate-report-btn", "n_clicks"),
    State('days-dropdown', 'value'),
    State('timeframe-dropdown', 'value'),
    prevent_initial_call=True
)
def generate_and_download_report(n_clicks, selected_days, selected_timeframe):
    if not n_clicks:
        return None
    try:
        # Filtrer les données brutes selon selected_days
        end_time = df_raw.index.max()
        start_time = end_time - timedelta(days=selected_days)
        df_period = df_raw.loc[start_time:end_time].copy()

        # Resample selon timeframe choisi
        rule = RESAMPLE_MAPPING.get(selected_timeframe, '30T')
        df = df_period.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        if df.empty:
            return None

        df = calculate_macd(df)
        df = compute_zones(df)
        df.fillna(method='ffill', inplace=True)

        triangles = detect_triangles(df)
        trades, balance_over_time = backtest_strategy(df)

        # Utiliser la fonction importée pour générer et sauvegarder le rapport
        html_content = generate_html_report(
            df=df,
            trades=trades,
            balance_over_time=balance_over_time,
            triangles=triangles,
            symbol="BTC/USDT",
            timeframe=selected_timeframe,
            initial_balance=100
        )
        filename = f"backtest_report_btc_{selected_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        save_report(html_content, filename)

        return dcc.send_file(filename)

    except Exception as e:
        print(f"Erreur lors de la génération du rapport : {e}")
        return None

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
# Callback Mise à Jour Graphiques & Table
# ========================
@app.callback(
    [
        Output('price-graph', 'figure'),
        Output('macd-graph', 'figure'),
        Output('balance-graph', 'figure'),
        Output('trades-table', 'data')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('xaxis-store', 'data'),
        Input('days-dropdown', 'value'),
        Input('timeframe-dropdown', 'value')
    ]
)
def update_graphs(n_intervals, store_data, selected_days, selected_timeframe):
    try:
        # Filtrer les données brutes selon selected_days
        end_time = df_raw.index.max()
        start_time = end_time - timedelta(days=selected_days)
        df_period = df_raw.loc[start_time:end_time].copy()

        # Resample selon timeframe choisi
        rule = RESAMPLE_MAPPING.get(selected_timeframe, '30T')
        df = df_period.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        if df.empty:
            return [go.Figure()] * 4

        df = calculate_macd(df)
        df = compute_zones(df)
        df.fillna(method='ffill', inplace=True)

        triangles = detect_triangles(df)
        trades, balance_over_time = backtest_strategy(df)

        # === Figure Prix + Signaux ===
        fig_price = go.Figure()

        fig_price.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=False
        ))

        # Ajouter signaux sur le graphique prix
        for t in trades:
            if t['type'] == 'open_long':
                fig_price.add_trace(go.Scatter(
                    x=[t['timestamp_entry']],
                    y=[t['entry_price']],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=12),
                    name='Entry Long',
                    showlegend=False
                ))
            elif t['type'] == 'close_long':
                fig_price.add_trace(go.Scatter(
                    x=[t['timestamp_exit']],
                    y=[t['exit_price']],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=12),
                    name='Close Long',
                    showlegend=False
                ))
            elif t['type'] == 'open_short':
                fig_price.add_trace(go.Scatter(
                    x=[t['timestamp_entry']],
                    y=[t['entry_price']],
                    mode='markers',
                    marker=dict(symbol='circle', color='red', size=8),
                    name='Entry Short',
                    showlegend=False
                ))
            elif t['type'] == 'close_short':
                fig_price.add_trace(go.Scatter(
                    x=[t['timestamp_exit']],
                    y=[t['exit_price']],
                    mode='markers',
                    marker=dict(symbol='circle', color='green', size=8),
                    name='Close Short',
                    showlegend=False
                ))

        fig_price.update_layout(
            height=600,
            title_text=f"BTC/USDT ({selected_timeframe}) - Prix: {df['close'].iloc[-1]:.2f} USDT",
            xaxis_rangeslider_visible=False,
            yaxis_title='Prix (USDT)',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            uirevision='constant'
        )
        if store_data['xaxis.range[0]'] and store_data['xaxis.range[1]']:
            fig_price.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_price.update_xaxes(range=[df.index[0], df.index[-1]])

        # === Figure MACD + Signal ===
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['macd'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='MACD'
        ))
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['signal_line'],
            mode='lines',
            line=dict(color='orange', width=1),
            name='Signal'
        ))
        fig_macd.update_layout(
            height=300,
            title_text="MACD & Signal",
            xaxis_title='',
            yaxis_title='Valeur',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            uirevision='constant'
        )
        if store_data['xaxis.range[0]'] and store_data['xaxis.range[1]']:
            fig_macd.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_macd.update_xaxes(range=[df.index[0], df.index[-1]])

        # === Figure Balance ===
        balance_df = pd.DataFrame(balance_over_time)
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(
            x=balance_df['timestamp'],
            y=balance_df['balance'],
            mode='lines',
            name='Balance',
            line=dict(color='blue')
        ))
        for t in trades:
            if 'timestamp_exit' in t and t['timestamp_exit'] in balance_df['timestamp'].values:
                ts = t['timestamp_exit']
                bal_pt = balance_df.loc[balance_df['timestamp'] == ts, 'balance'].values[0]
                color = 'green' if t.get('profit', 0) > 0 else 'red'
                fig_balance.add_trace(go.Scatter(
                    x=[ts],
                    y=[bal_pt],
                    mode='markers+text',
                    marker=dict(color=color, size=10),
                    text=[f"{t.get('profit', 0):.2f}"],
                    textposition='top center',
                    showlegend=False
                ))
        fig_balance.update_layout(
            height=400,
            title_text='Évolution de la Balance',
            xaxis_title='Temps',
            yaxis_title='Balance (USDT)',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            uirevision='constant'
        )
        if store_data['xaxis.range[0]'] and store_data['xaxis.range[1]']:
            fig_balance.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_balance.update_xaxes(range=[balance_df['timestamp'].iloc[0], balance_df['timestamp'].iloc[-1]])

        # Préparer données pour la table des trades
        trades_table_data = []
        for t in trades:
            entry_price = t.get('entry_price', 0)
            exit_price = t.get('exit_price', 0)
            entry_ts = t.get('timestamp_entry', "")
            exit_ts = t.get('timestamp_exit', "")
            pnl = t.get('profit', 0)
            trades_table_data.append({
                'type': t['type'],
                'entry_price': f"{entry_price:.2f}",
                'exit_price': f"{exit_price:.2f}" if 'exit_price' in t else "",
                'timestamp_entry': entry_ts.strftime('%Y-%m-%d %H:%M') if isinstance(entry_ts, pd.Timestamp) else entry_ts,
                'timestamp_exit': exit_ts.strftime('%Y-%m-%d %H:%M') if isinstance(exit_ts, pd.Timestamp) else exit_ts,
                'profit': f"{pnl:.2f}" if 'profit' in t else ""
            })

        return fig_price, fig_macd, fig_balance, trades_table_data

    except Exception as e:
        print(f"Erreur dans update_graphs : {e}")
        return [go.Figure()] * 4

# Callback pour capturer le zoom
@app.callback(
    Output('xaxis-store', 'data'),
    [
        Input('price-graph', 'relayoutData'),
        Input('macd-graph', 'relayoutData'),
        Input('balance-graph', 'relayoutData')
    ],
    [State('xaxis-store', 'data')]
)
def update_store(relayout_price, relayout_macd, relayout_balance, current_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_store
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    relayoutData = None
    if triggered_id == 'price-graph':
        relayoutData = relayout_price
    elif triggered_id == 'macd-graph':
        relayoutData = relayout_macd
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

if __name__ == '__main__':
    print("Démarrage de l'application Dash...")
    app.run(debug=True, threaded=False)
