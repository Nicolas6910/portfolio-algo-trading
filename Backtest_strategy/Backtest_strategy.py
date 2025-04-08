import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_table
import dash_bootstrap_components as dbc
import copy
import base64
import io

# ========================
# Configuration des Paramètres
# ========================

DEFAULT_DISPLAY_PARAMS = {
    'squares': {
        'size': 10,
        'colors': {
            'red': 'red',
            'darkred': 'darkred',
            'lightred': 'lightcoral',
            'green': 'green',
            'darkgreen': 'darkgreen',
            'lightgreen': 'lightgreen'
        }
    }
}

neutral_params = {
    'sp1': 2,
    'vp1': 2,
    'sp2': 4,
    'vp2': 1.6,
    'sp3': 6,
    'vp3': 2.4,
    'sp4': 3,
    'vp4': 1,
    'vt1': 3,
    'vt2': 1.6,
    'vt3': 2.4,
    'vt4': 2
}

def generate_zone_params(base_zone, zone_names, increment=0.2, rounding=1):
    zones = {}
    current_zone_params = copy.deepcopy(base_zone)
    zones['neutral'] = current_zone_params

    for zone in zone_names:
        previous_zone = 'neutral' if zone == zone_names[0] else zone_names[zone_names.index(zone)-1]
        previous_zone_params = copy.deepcopy(zones[previous_zone])

        new_zone_params = copy.deepcopy(previous_zone_params)

        for key in new_zone_params.keys():
            if key.startswith('vp') or key.startswith('vt'):
                new_value = round(new_zone_params[key] * (1 + increment), rounding)
                new_zone_params[key] = new_value
        zones[zone] = new_zone_params

    return zones

zone_names = ['lightblue', 'darkblue']
ZONE_PARAMS = generate_zone_params(neutral_params, zone_names)

STOP_LOSS_PERCENTAGE = 0.01  # 1%

def read_csv_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        print(f"Total bougies récupérées: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
    except pd.errors.ParserError:
        print(f"Erreur : Problème de parsing avec le fichier {file_path}.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
    return pd.DataFrame()

def calculate_ema(df, series_name, ema_lengths):
    for length in ema_lengths:
        ema_col = f'{series_name}_ema_{length}'
        df[ema_col] = df[series_name].ewm(span=length, adjust=False).mean()
    return df

def calculate_sum_of_gaps(df):
    ema_lengths = list(range(5, 205, 5))
    gaps_columns = []

    for i in range(1, len(ema_lengths)):
        prev_length = ema_lengths[i-1]
        current_length = ema_lengths[i]
        gap_col = f'gaps_{prev_length}_{current_length}'
        ema_prev = f'close_ema_{prev_length}'
        ema_current = f'close_ema_{current_length}'
        if ema_prev in df.columns and ema_current in df.columns:
            df[gap_col] = (df[ema_prev] - df[ema_current]) / df['close'] * 100
            gaps_columns.append(gap_col)
        else:
            print(f"Colonnes {ema_prev} ou {ema_current} manquantes dans df.")

    if gaps_columns:
        df['sum_of_gaps_percentage'] = df[gaps_columns].sum(axis=1)
    else:
        df['sum_of_gaps_percentage'] = 0

    return df

def calculate_average_slope(df, ema_lengths):
    ema_cols = [f'close_ema_{length}' for length in ema_lengths]
    slopes = df[ema_cols].diff()
    df['average_slope_percentage'] = slopes.mean(axis=1) / df['close'] * 100
    df['average_slope_percentage'] *= 50
    return df

def detect_ema_crossovers(df, series_name, short_length, long_length):
    bearish_crosses = []
    bullish_crosses = []

    ema_short_col = f'{series_name}_ema_{short_length}'
    ema_long_col = f'{series_name}_ema_{long_length}'
    if ema_short_col not in df.columns or ema_long_col not in df.columns:
        print(f"Colonnes {ema_short_col} ou {ema_long_col} manquantes dans df.")
        return bearish_crosses, bullish_crosses

    df['ema_diff'] = df[ema_short_col] - df[ema_long_col]

    df['signal'] = 0
    df.loc[(df['ema_diff'] > 0) & (df['ema_diff'].shift(1) <= 0), 'signal'] = 1  # Bullish
    df.loc[(df['ema_diff'] < 0) & (df['ema_diff'].shift(1) >= 0), 'signal'] = -1  # Bearish

    for idx, row in df.iterrows():
        if row['signal'] == 1:
            y = df['close'].loc[idx]
            explanation = f'Croisement haussier : {ema_short_col} a croisé au-dessus de {ema_long_col}.'
            bullish_crosses.append({
                'x': idx,
                'y': y,
                'color': 'green',
                'type': 'bullish_crossover',
                'explanation': explanation
            })
        elif row['signal'] == -1:
            y = df['close'].loc[idx]
            explanation = f'Croisement baissier : {ema_short_col} a croisé en dessous de {ema_long_col}.'
            bearish_crosses.append({
                'x': idx,
                'y': y,
                'color': 'red',
                'type': 'bearish_crossover',
                'explanation': explanation
            })

    df.drop(['ema_diff', 'signal'], axis=1, inplace=True)
    return bearish_crosses, bullish_crosses

def get_bar_colors_percentage(series):
    colors = ['black'] * len(series)
    for i in range(len(series)):
        value = series.iloc[i]
        if value > 0.7:
            colors[i] = 'red'
        elif value < -0.7:
            colors[i] = 'green'
    return colors

def backtest_strategy(df, points_for_backtest):
    balance = 100
    position = None  # 'long' ou 'short'
    entry_price = 0
    entry_timestamp = None

    # Paramètres initiaux de la position
    initial_trade_size = 50
    leverage = 15
    fee_per_trade = 3

    # Paramètres de la martingale
    MAX_ADDITIONS = 4
    MARTINGALE_STEP = 0.05  # 0.4%

    position_size = 0
    position_cost = 0
    original_entry_price = 0
    martingale_count = 0
    last_added_size = 0

    trades = []
    balance_over_time = []

    # Pas de stop loss : toutes les références à stop_loss_price et STOP_LOSS_PERCENTAGE supprimées.

    points_sorted = sorted(points_for_backtest, key=lambda x: x['x'])
    points_dict = {point['x']: point for point in points_sorted}

    for timestamp, row in df.iterrows():
        price = row['close']

        # Martingale si une position est ouverte
        if position:
            if position == 'long':
                while martingale_count < MAX_ADDITIONS and price <= original_entry_price * (1 - (martingale_count + 1) * MARTINGALE_STEP):
                    new_add_size = last_added_size * 2
                    position_size += new_add_size
                    position_cost += new_add_size * price
                    entry_price = position_cost / position_size
                    martingale_count += 1
                    last_added_size = new_add_size
                    trades.append({
                        'type': 'add_long',
                        'direction': 'long',
                        'entry_price': entry_price,
                        'exit_price': None,
                        'timestamp_entry': timestamp,
                        'timestamp_exit': None,
                        'profit': None,
                        'explanation': "Rajout à la position martingale (long)",
                        'zone': df.loc[entry_timestamp, 'zone'] if entry_timestamp in df.index else 'Unknown'
                    })
            elif position == 'short':
                while martingale_count < MAX_ADDITIONS and price >= original_entry_price * (1 + (martingale_count + 1) * MARTINGALE_STEP):
                    new_add_size = last_added_size * 2
                    position_size += new_add_size
                    position_cost += new_add_size * price
                    entry_price = position_cost / position_size
                    martingale_count += 1
                    last_added_size = new_add_size
                    trades.append({
                        'type': 'add_short',
                        'direction': 'short',
                        'entry_price': entry_price,
                        'exit_price': None,
                        'timestamp_entry': timestamp,
                        'timestamp_exit': None,
                        'profit': None,
                        'explanation': "Rajout à la position martingale (short)",
                        'zone': df.loc[entry_timestamp, 'zone'] if entry_timestamp in df.index else 'Unknown'
                    })

        # Vérification des signaux
        point = points_dict.get(timestamp, None)
        if point:
            is_triangle = point.get('is_triangle', False)
            is_inverse = False

            if position:
                if position == 'long' and point['color'] == 'red':
                    is_inverse = True
                    explanation = "Point inverse détecté : Signal rouge pendant une position longue."
                elif position == 'short' and point['color'] == 'green':
                    is_inverse = True
                    explanation = "Point inverse détecté : Signal vert pendant une position courte."

                if is_inverse or is_triangle:
                    # Clôturer la position
                    exit_price = price
                    if position == 'long':
                        profit = (exit_price - entry_price) * position_size * leverage / entry_price - fee_per_trade
                        trade_type = 'close_long'
                        direction = 'long'
                    else:
                        profit = (entry_price - exit_price) * position_size * leverage / entry_price - fee_per_trade
                        trade_type = 'close_short'
                        direction = 'short'

                    balance += profit
                    trades.append({
                        'type': trade_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'timestamp_entry': entry_timestamp,
                        'timestamp_exit': timestamp,
                        'profit': profit,
                        'explanation': explanation if is_inverse else point.get('explanation', ''),
                        'zone': df.loc[entry_timestamp, 'zone'] if entry_timestamp in df.index else 'Unknown'
                    })

                    # Réinitialiser la position et la martingale
                    position = None
                    position_size = 0
                    position_cost = 0
                    martingale_count = 0
                    last_added_size = 0
                    original_entry_price = 0
                    entry_price = 0
                    balance_over_time.append({'timestamp': timestamp, 'balance': balance})

                    # Si c'est un signal (pas un triangle), ouvrir une nouvelle position
                    if not is_triangle:
                        if point['color'] == 'green':
                            position = 'long'
                            original_entry_price = price
                            entry_price = price
                            entry_timestamp = timestamp
                            position_size = initial_trade_size
                            position_cost = entry_price * position_size
                            last_added_size = position_size
                            martingale_count = 0
                            trades.append({
                                'type': 'open_long',
                                'direction': 'long',
                                'entry_price': entry_price,
                                'exit_price': None,
                                'timestamp_entry': entry_timestamp,
                                'timestamp_exit': None,
                                'profit': None,
                                'explanation': point.get('explanation', ''),
                                'zone': df.loc[timestamp, 'zone'] if timestamp in df.index else 'Unknown'
                            })
                        elif point['color'] == 'red':
                            position = 'short'
                            original_entry_price = price
                            entry_price = price
                            entry_timestamp = timestamp
                            position_size = initial_trade_size
                            position_cost = entry_price * position_size
                            last_added_size = position_size
                            martingale_count = 0
                            trades.append({
                                'type': 'open_short',
                                'direction': 'short',
                                'entry_price': entry_price,
                                'exit_price': None,
                                'timestamp_entry': entry_timestamp,
                                'timestamp_exit': None,
                                'profit': None,
                                'explanation': point.get('explanation', ''),
                                'zone': df.loc[timestamp, 'zone'] if timestamp in df.index else 'Unknown'
                            })
                    continue

            # Si aucune position n'est ouverte, ouvrir une position si ce n'est pas un triangle
            if not position and not is_triangle:
                if point['color'] == 'green':
                    position = 'long'
                    original_entry_price = price
                    entry_price = price
                    entry_timestamp = timestamp
                    position_size = initial_trade_size
                    position_cost = entry_price * position_size
                    last_added_size = position_size
                    martingale_count = 0
                    trades.append({
                        'type': 'open_long',
                        'direction': 'long',
                        'entry_price': entry_price,
                        'exit_price': None,
                        'timestamp_entry': entry_timestamp,
                        'timestamp_exit': None,
                        'profit': None,
                        'explanation': point.get('explanation', ''),
                        'zone': df.loc[timestamp, 'zone'] if timestamp in df.index else 'Unknown'
                    })
                elif point['color'] == 'red':
                    position = 'short'
                    original_entry_price = price
                    entry_price = price
                    entry_timestamp = timestamp
                    position_size = initial_trade_size
                    position_cost = entry_price * position_size
                    last_added_size = position_size
                    martingale_count = 0
                    trades.append({
                        'type': 'open_short',
                        'direction': 'short',
                        'entry_price': entry_price,
                        'exit_price': None,
                        'timestamp_entry': entry_timestamp,
                        'timestamp_exit': None,
                        'profit': None,
                        'explanation': point.get('explanation', ''),
                        'zone': df.loc[timestamp, 'zone'] if timestamp in df.index else 'Unknown'
                    })

        balance_over_time.append({'timestamp': timestamp, 'balance': balance})

    # Clôturer la position restante à la fin
    if position:
        exit_price = df['close'].iloc[-1]
        timestamp = df.index[-1]
        if position == 'long':
            profit = (exit_price - entry_price) * position_size * leverage / entry_price - fee_per_trade
            trade_type = 'close_long'
            direction = 'long'
        else:
            profit = (entry_price - exit_price) * position_size * leverage / entry_price - fee_per_trade
            trade_type = 'close_short'
            direction = 'short'
        balance += profit
        trades.append({
            'type': trade_type,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'timestamp_entry': entry_timestamp,
            'timestamp_exit': timestamp,
            'profit': profit,
            'explanation': "Clôture finale de la position à la fin des données.",
            'zone': df.loc[entry_timestamp, 'zone'] if entry_timestamp in df.index else 'Unknown'
        })
        balance_over_time.append({'timestamp': timestamp, 'balance': balance})
        position = None
        position_size = 0
        position_cost = 0
        martingale_count = 0
        last_added_size = 0
        original_entry_price = 0
        entry_price = 0

    return trades, balance_over_time

def filter_triangles_after_signals(signal_points, triangle_points):
    if triangle_points.empty or not signal_points:
        return pd.DataFrame()
    filtered_triangles = []
    signal_times = [point['x'] for point in signal_points]
    signal_times_series = pd.Series(signal_times)
    last_signal_time = None
    for idx, triangle_row in triangle_points.iterrows():
        triangle_time = idx
        signals_before_triangle = signal_times_series[signal_times_series <= triangle_time]
        if not signals_before_triangle.empty:
            last_signal = signals_before_triangle.iloc[-1]
        else:
            last_signal = None
        if last_signal != last_signal_time:
            filtered_triangles.append(triangle_row)
            last_signal_time = last_signal
        else:
            continue
    if filtered_triangles:
        filtered_triangles_df = pd.DataFrame(filtered_triangles)
    else:
        filtered_triangles_df = pd.DataFrame()
    return filtered_triangles_df

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the server variable

app.layout = html.Div([
    dcc.Store(id='xaxis-store', data={'xaxis.range[0]': None, 'xaxis.range[1]': None}),
    dcc.Store(id='uploaded-data', data=None),
    dcc.Store(id='display-params', data=DEFAULT_DISPLAY_PARAMS),

    html.H1(f"Données PEPE/USDT en temps réel - Backtest", style={'textAlign': 'center'}),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Glissez-déposez ou ',
                html.A('Sélectionnez un fichier CSV')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
    ], style={'padding': '10px'}),

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
                    options=[
                        {'label': '1 minute', 'value': '1m'}
                    ],
                    value='1m',
                    clearable=False
                ),
            ], width=4),
        ], justify='center', style={'padding': '10px'}),
    ]),

    html.Div([
        dcc.Graph(id='live-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    html.Div([
        dcc.Graph(id='gaps-percentage-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

    html.Div([
        dcc.Graph(id='slope-percentage-graph', config={'displayModeBar': True})
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),

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
                        {'name': 'Type', 'id': 'Type de Trade'},
                        {'name': 'Direction', 'id': 'Direction'},
                        {'name': 'Prix Entrée', 'id': 'Prix d\'Entrée'},
                        {'name': 'Prix Sortie', 'id': 'Prix de Sortie'},
                        {'name': 'Date Entrée', 'id': 'Date d\'Entrée'},
                        {'name': 'Date Sortie', 'id': 'Date de Sortie'},
                        {'name': 'PnL', 'id': 'PnL'},
                        {'name': 'Zone', 'id': 'Zone'},
                        {'name': 'Explications', 'id': 'Explications'},
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    page_size=10
                )
            ], width=4)
        ])
    ], style={'width': '100%', 'display': 'block', 'padding': '10px'}),
], style={'padding': '10px'})

@app.callback(
    Output('xaxis-store', 'data'),
    [
        Input('live-graph', 'relayoutData'),
        Input('gaps-percentage-graph', 'relayoutData'),
        Input('slope-percentage-graph', 'relayoutData'),
        Input('balance-graph', 'relayoutData'),
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

@app.callback(
    Output('uploaded-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def handle_file_upload(contents, filename, last_modified):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), parse_dates=['timestamp'], index_col='timestamp')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df.to_json(date_format='iso', orient='split')
    except Exception as e:
        print(f"Erreur lors de l'analyse du fichier CSV : {e}")
        return None

@app.callback(
    [
        Output('live-graph', 'figure'),
        Output('gaps-percentage-graph', 'figure'),
        Output('slope-percentage-graph', 'figure'),
        Output('balance-graph', 'figure'),
        Output('trades-table', 'data')
    ],
    [
        Input('xaxis-store', 'data'),
        Input('days-dropdown', 'value'),
        Input('timeframe-dropdown', 'value'),
        Input('uploaded-data', 'data')
    ]
)
def update_graph(store_data, selected_days, selected_timeframe, uploaded_data):
    try:
        gaps_short_ema = 10
        gaps_long_ema = 50
        slope_short_ema = 10
        slope_long_ema = 50

        if uploaded_data is not None:
            df = pd.read_json(uploaded_data, orient='split')
        else:
            csv_file_path = r'C:\Users\nicol\Desktop\Python API\PEPE_Historical_Data\PEPE_Historical_Data_2024_Jan_Oct\pepe_usdt_february_2024.csv'
            print(f"Lecture des données depuis le fichier CSV : {csv_file_path}")
            df = read_csv_data(csv_file_path)
            print(f"Forme des données récupérées : {df.shape}")

        if df.empty:
            print("Le fichier CSV est vide ou mal formaté.")
            return [go.Figure()] * 5

        selected_timeframe = '1m'  # On reste sur 1m

        total_bougies_display = selected_days * 24 * 60
        MAX_EMA_LENGTH = 200
        extra_bougies = MAX_EMA_LENGTH * 5
        total_bougies = total_bougies_display + extra_bougies

        df_display = df.iloc[-total_bougies:].copy()

        ema_lengths_main = list(range(5, 205, 5))
        df_display = calculate_ema(df_display, 'close', ema_lengths_main)

        df_display = calculate_sum_of_gaps(df_display)
        df_display = calculate_average_slope(df_display, ema_lengths_main)

        df_display = calculate_ema(df_display, 'sum_of_gaps_percentage', [gaps_short_ema, gaps_long_ema])
        df_display = calculate_ema(df_display, 'average_slope_percentage', [slope_short_ema, slope_long_ema])

        df_display.fillna(0, inplace=True)

        ema_cols = [f'close_ema_{length}' for length in ema_lengths_main]
        df_display['average_ema'] = df_display[ema_cols].mean(axis=1)

        bearish_crosses = []
        bullish_crosses = []

        cross_bearish_gaps, cross_bullish_gaps = detect_ema_crossovers(df_display, 'sum_of_gaps_percentage', gaps_short_ema, gaps_long_ema)
        bearish_crosses.extend(cross_bearish_gaps)
        bullish_crosses.extend(cross_bullish_gaps)

        cross_bearish_slope, cross_bullish_slope = detect_ema_crossovers(df_display, 'average_slope_percentage', slope_short_ema, slope_long_ema)
        bearish_crosses.extend(cross_bearish_slope)
        bullish_crosses.extend(cross_bullish_slope)

        points = bearish_crosses + bullish_crosses
        points_sorted = sorted(points, key=lambda x: x['x'])

        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display['close'],
            mode='lines',
            name='Prix de Clôture',
            line=dict(color='blue'),
            showlegend=False
        ))

        df_1h = df_display.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum',
            'sum_of_gaps_percentage': 'mean',
            'average_slope_percentage': 'mean'
        })

        zones = []
        if df_1h.empty:
            df_display['zone'] = 'neutral'
        else:
            df_1h['pct_change'] = df_1h['close'].pct_change() * 100
            df_1h['abs_pct_change'] = df_1h['pct_change'].abs()

            current_zone = None
            zone_start = None
            stabilization_counter = 0

            for idx, row in df_1h.iterrows():
                pct_change = row['pct_change']
                abs_pct_change = row['abs_pct_change']

                if current_zone is None:
                    if 5 <= pct_change < 10:
                        current_zone = 'lightblue'
                        zone_start = idx + timedelta(hours=1)
                        stabilization_counter = 0
                    elif pct_change >= 10:
                        current_zone = 'darkblue'
                        zone_start = idx + timedelta(hours=1)
                        stabilization_counter = 0
                else:
                    if 0 <= abs_pct_change <= 2:
                        stabilization_counter += 1
                        if current_zone == 'lightblue' and stabilization_counter >= 5:
                            zone_end = idx
                            zones.append({
                                'type': current_zone,
                                'start': zone_start,
                                'end': zone_end
                            })
                            current_zone = None
                            zone_start = None
                            stabilization_counter = 0
                        elif current_zone == 'darkblue' and stabilization_counter >= 10:
                            zone_end = idx
                            zones.append({
                                'type': current_zone,
                                'start': zone_start,
                                'end': zone_end
                            })
                            current_zone = None
                            zone_start = None
                            stabilization_counter = 0
                    else:
                        stabilization_counter = 0

            if current_zone is not None and zone_start is not None:
                zone_end = df_1h.index[-1]
                zones.append({
                    'type': current_zone,
                    'start': zone_start,
                    'end': zone_end
                })

            color_map = {
                'lightblue': 'rgba(173, 216, 230, 1)',
                'darkblue': 'rgba(0, 0, 139, 1)',
            }

            for zone in zones:
                fig_live.add_shape(
                    type='rect',
                    xref='x',
                    yref='paper',
                    x0=zone['start'],
                    y0=0,
                    x1=zone['end'],
                    y1=1,
                    fillcolor=color_map[zone['type']],
                    opacity=0.3,
                    layer='below',
                    line_width=0
                )

            df_display['zone'] = 'neutral'
            for zone in zones:
                df_display.loc[(df_display.index >= zone['start']) & (df_display.index <= zone['end']), 'zone'] = zone['type']

        def get_zone_params(zone):
            return ZONE_PARAMS.get(zone, ZONE_PARAMS['neutral'])

        red_dot_points = pd.DataFrame()
        green_dot_points = pd.DataFrame()

        for zone_type in ['neutral', 'lightblue', 'darkblue']:
            zone_df = df_display[df_display['zone'] == zone_type]
            params = get_zone_params(zone_type)

            sp1 = params['sp1']; vp1 = params['vp1']
            sp2 = params['sp2']; vp2 = params['vp2']
            sp3 = params['sp3']; vp3 = params['vp3']
            sp4 = params['sp4']; vp4 = params['vp4']

            zone_df['price_above_avg_ema'] = zone_df['close'] > zone_df['average_ema']
            zone_df['price_cross_down'] = (zone_df['price_above_avg_ema'].shift(1) == True) & (zone_df['price_above_avg_ema'] == False)
            zone_df['price_cross_up'] = (zone_df['price_above_avg_ema'].shift(1) == False) & (zone_df['price_above_avg_ema'] == True)

            zone_df['cond_red_1'] = zone_df['average_slope_percentage'] <= zone_df['average_slope_percentage'].shift(sp1) - vp1
            zone_df['cond_red_2'] = zone_df['average_slope_percentage'] <= zone_df['average_slope_percentage'].shift(sp2) - vp2
            zone_df['cond_red_3'] = zone_df['average_slope_percentage'] <= zone_df['average_slope_percentage'].shift(sp3) - vp3
            zone_df['cond_red_4'] = zone_df['average_slope_percentage'] <= zone_df['average_slope_percentage'].shift(sp4) - vp4

            zone_df['condition_red'] = zone_df['price_cross_down'] & (
                zone_df['cond_red_1'] | zone_df['cond_red_2'] | zone_df['cond_red_3'] | zone_df['cond_red_4']
            )

            zone_df['cond_green_1'] = zone_df['average_slope_percentage'] >= zone_df['average_slope_percentage'].shift(sp1) + vp1
            zone_df['cond_green_2'] = zone_df['average_slope_percentage'] >= zone_df['average_slope_percentage'].shift(sp2) + vp2
            zone_df['cond_green_3'] = zone_df['average_slope_percentage'] >= zone_df['average_slope_percentage'].shift(sp3) + vp3
            zone_df['cond_green_4'] = zone_df['average_slope_percentage'] >= zone_df['average_slope_percentage'].shift(sp4) + vp4

            zone_df['condition_green'] = zone_df['price_cross_up'] & (
                zone_df['cond_green_1'] | zone_df['cond_green_2'] | zone_df['cond_green_3'] | zone_df['cond_green_4']
            )

            red_points = zone_df[zone_df['condition_red']]
            green_points = zone_df[zone_df['condition_green']]

            red_dot_points = pd.concat([red_dot_points, red_points])
            green_dot_points = pd.concat([green_dot_points, green_points])

        def alternate_colors(red_points, green_points):
            combined = pd.concat([red_points.assign(color='red'), green_points.assign(color='green')])
            combined = combined.sort_index()
            filtered = []
            last_color = None
            for idx, row in combined.iterrows():
                current_color = row['color']
                if current_color != last_color:
                    filtered.append(row)
                    last_color = current_color
            return pd.DataFrame(filtered)

        filtered_points = alternate_colors(red_dot_points, green_dot_points)

        def get_point_explanation(row):
            zone_type = row['zone']
            params = get_zone_params(zone_type)
            sp1 = params['sp1']; vp1 = params['vp1']
            sp2 = params['sp2']; vp2 = params['vp2']
            sp3 = params['sp3']; vp3 = params['vp3']
            sp4 = params['sp4']; vp4 = params['vp4']

            explanations = []
            if row['color'] == 'red':
                explanations.append("Croisement du prix avec la moyenne EMA à la baisse.")
                if row['cond_red_1']:
                    explanations.append(f"La pente a diminué d'au moins {vp1}% sur {sp1} bougie(s).")
                if row['cond_red_2']:
                    explanations.append(f"La pente a diminué d'au moins {vp2}% sur {sp2} bougie(s).")
                if row['cond_red_3']:
                    explanations.append(f"La pente a diminué d'au moins {vp3}% sur {sp3} bougie(s).")
                if row['cond_red_4']:
                    explanations.append(f"La pente a diminué d'au moins {vp4}% sur {sp4} bougie(s).")
            elif row['color'] == 'green':
                explanations.append("Croisement du prix avec la moyenne EMA à la hausse.")
                if row['cond_green_1']:
                    explanations.append(f"La pente a augmenté d'au moins {vp1}% sur {sp1} bougie(s).")
                if row['cond_green_2']:
                    explanations.append(f"La pente a augmenté d'au moins {vp2}% sur {sp2} bougie(s).")
                if row['cond_green_3']:
                    explanations.append(f"La pente a augmenté d'au moins {vp3}% sur {sp3} bougie(s).")
                if row['cond_green_4']:
                    explanations.append(f"La pente a augmenté d'au moins {vp4}% sur {sp4} bougie(s).")
            return "<br>".join(explanations)

        if not filtered_points.empty:
            fig_live.add_trace(go.Scatter(
                x=filtered_points.index,
                y=filtered_points['close'],
                mode='markers',
                marker=dict(color=filtered_points['color'], size=10, symbol='circle'),
                name='Points',
                text=filtered_points.apply(get_point_explanation, axis=1),
                hoverinfo='text',
                showlegend=False
            ))

        points_for_backtest = []
        for idx, row in filtered_points.iterrows():
            point = {
                'x': idx,
                'y': row['close'],
                'color': row['color'],
                'type': 'signal',
                'explanation': get_point_explanation(row),
                'is_triangle': False
            }
            points_for_backtest.append(point)

        triangle_up_points = pd.DataFrame()
        triangle_down_points = pd.DataFrame()

        for zone_type in ['neutral', 'lightblue', 'darkblue']:
            zone_df = df_display[df_display['zone'] == zone_type]
            params = get_zone_params(zone_type)

            vt1 = params['vt1']
            vt2 = params['vt2']
            vt3 = params['vt3']
            vt4 = params['vt4']

            if len(zone_df) < 6:
                continue

            zone_df['cond_triangle_1_up'] = (
                (zone_df['average_slope_percentage'] > 0) &
                (zone_df['average_slope_percentage'].shift(1) > 0) &
                (zone_df['average_slope_percentage'].shift(2) > 0) &
                ((zone_df['average_slope_percentage'].shift(2) - zone_df['average_slope_percentage'].shift(1)) >= vt1) &
                ((zone_df['average_slope_percentage'].shift(1) - zone_df['average_slope_percentage']) >= vt1)
            )

            zone_df['cond_triangle_2_up'] = (
                (zone_df['average_slope_percentage'] > 0) &
                (zone_df['average_slope_percentage'].shift(1) > 0) &
                (zone_df['average_slope_percentage'].shift(2) > 0) &
                (zone_df['average_slope_percentage'].shift(3) > 0) &
                ((zone_df['average_slope_percentage'].shift(3) - zone_df['average_slope_percentage'].shift(2)) >= vt2) &
                ((zone_df['average_slope_percentage'].shift(1) - zone_df['average_slope_percentage']) >= vt2)
            )

            zone_df['cond_triangle_3_up'] = (
                (zone_df['average_slope_percentage'] > 0) &
                (zone_df['average_slope_percentage'].shift(1) > 0) &
                (zone_df['average_slope_percentage'].shift(2) > 0) &
                (zone_df['average_slope_percentage'].shift(3) > 0) &
                (zone_df['average_slope_percentage'].shift(4) > 0) &
                ((zone_df['average_slope_percentage'].shift(1) - zone_df['average_slope_percentage']) >= vt3) &
                ((zone_df['average_slope_percentage'].shift(4) - zone_df['average_slope_percentage'].shift(3)) >= vt3)
            )

            zone_df['cond_triangle_4_up'] = (
                (zone_df['average_slope_percentage'] > 0) &
                (zone_df['average_slope_percentage'].shift(1) > 0) &
                (zone_df['average_slope_percentage'].shift(2) > 0) &
                (zone_df['average_slope_percentage'].shift(3) > 0) &
                (zone_df['average_slope_percentage'].shift(4) > 0) &
                (zone_df['average_slope_percentage'].shift(5) > 0) &
                ((zone_df['average_slope_percentage'].shift(1) - zone_df['average_slope_percentage']) >= vt4) &
                ((zone_df['average_slope_percentage'].shift(5) - zone_df['average_slope_percentage'].shift(4)) >= vt4)
            )

            zone_df['cond_triangle_1_down'] = (
                (zone_df['average_slope_percentage'] < 0) &
                (zone_df['average_slope_percentage'].shift(1) < 0) &
                (zone_df['average_slope_percentage'].shift(2) < 0) &
                ((zone_df['average_slope_percentage'].shift(2) - zone_df['average_slope_percentage'].shift(1)) <= -vt1) &
                ((zone_df['average_slope_percentage'].shift(1) - zone_df['average_slope_percentage']) <= -vt1)
            )

            zone_df['cond_triangle_2_down'] = (
                (zone_df['average_slope_percentage'] < 0) &
                (zone_df['average_slope_percentage'].shift(1) < 0) &
                (zone_df['average_slope_percentage'].shift(2) < 0) &
                (zone_df['average_slope_percentage'].shift(3) < 0) &
                ((zone_df['average_slope_percentage'].shift(3) - zone_df['average_slope_percentage'].shift(2)) <= -vt2) &
                ((zone_df['average_slope_percentage'].shift(1) - zone_df['average_slope_percentage']) <= -vt2)
            )

            zone_df['cond_triangle_3_down'] = (
                (zone_df['average_slope_percentage'] < 0) &
                (zone_df['average_slope_percentage'].shift(1) < 0) &
                (zone_df['average_slope_percentage'].shift(2) < 0) &
                (zone_df['average_slope_percentage'].shift(3) < 0) &
                (zone_df['average_slope_percentage'].shift(4) < 0) &
                ((zone_df['average_slope_percentage'] - zone_df['average_slope_percentage'].shift(1)) <= -vt3) &
                ((zone_df['average_slope_percentage'].shift(3) - zone_df['average_slope_percentage'].shift(4)) <= -vt3)
            )

            zone_df['cond_triangle_4_down'] = (
                (zone_df['average_slope_percentage'] < 0) &
                (zone_df['average_slope_percentage'].shift(1) < 0) &
                (zone_df['average_slope_percentage'].shift(2) < 0) &
                (zone_df['average_slope_percentage'].shift(3) < 0) &
                (zone_df['average_slope_percentage'].shift(4) < 0) &
                (zone_df['average_slope_percentage'].shift(5) < 0) &
                ((zone_df['average_slope_percentage'] - zone_df['average_slope_percentage'].shift(1)) <= -vt4) &
                ((zone_df['average_slope_percentage'].shift(4) - zone_df['average_slope_percentage'].shift(5)) <= -vt4)
            )

            zone_df['triangle_up'] = (
                zone_df['cond_triangle_1_up'] | zone_df['cond_triangle_2_up'] |
                zone_df['cond_triangle_3_up'] | zone_df['cond_triangle_4_up']
            )

            zone_df['triangle_down'] = (
                zone_df['cond_triangle_1_down'] | zone_df['cond_triangle_2_down'] |
                zone_df['cond_triangle_3_down'] | zone_df['cond_triangle_4_down']
            )

            triangle_up_points = pd.concat([triangle_up_points, zone_df[zone_df['triangle_up']]])
            triangle_down_points = pd.concat([triangle_down_points, zone_df[zone_df['triangle_down']]])

        if not triangle_up_points.empty:
            triangle_up_points['type'] = 'triangle_up'
        if not triangle_down_points.empty:
            triangle_down_points['type'] = 'triangle_down'

        triangle_points = pd.concat([triangle_up_points, triangle_down_points]).sort_index()
        filtered_triangles = filter_triangles_after_signals(points_for_backtest, triangle_points)

        triangle_up_filtered = filtered_triangles[filtered_triangles['type'] == 'triangle_up'] if not filtered_triangles.empty else pd.DataFrame()
        triangle_down_filtered = filtered_triangles[filtered_triangles['type'] == 'triangle_down'] if not filtered_triangles.empty else pd.DataFrame()

        if not triangle_up_filtered.empty:
            fig_live.add_trace(go.Scatter(
                x=triangle_up_filtered.index,
                y=triangle_up_filtered['high'] * 1.02,
                mode='markers',
                marker=dict(symbol='triangle-up', color='blue', size=12),
                name='Triangle Up',
                text=["Conditions Triangle Up satisfaites" for _ in triangle_up_filtered.index],
                hoverinfo='text',
                showlegend=False
            ))
            for idx, row in triangle_up_filtered.iterrows():
                points_for_backtest.append({
                    'x': idx,
                    'y': row['close'],
                    'color': 'blue',
                    'type': 'triangle_up',
                    'is_triangle': True,
                    'explanation': "Signal de clôture de position (Triangle Up)"
                })

        if not triangle_down_filtered.empty:
            fig_live.add_trace(go.Scatter(
                x=triangle_down_filtered.index,
                y=triangle_down_filtered['low'] * 0.98,
                mode='markers',
                marker=dict(symbol='triangle-down', color='orange', size=12),
                name='Triangle Down',
                text=["Conditions Triangle Down satisfaites" for _ in triangle_down_filtered.index],
                hoverinfo='text',
                showlegend=False
            ))
            for idx, row in triangle_down_filtered.iterrows():
                points_for_backtest.append({
                    'x': idx,
                    'y': row['close'],
                    'color': 'orange',
                    'type': 'triangle_down',
                    'is_triangle': True,
                    'explanation': "Signal de clôture de position (Triangle Down)"
                })

        points_for_backtest = sorted(points_for_backtest, key=lambda x: x['x'])

        trades, balance_over_time = backtest_strategy(df_display, points_for_backtest)
        if not balance_over_time:
            final_balance = 100
        else:
            final_balance = balance_over_time[-1]['balance']
        print(f"\nBalance finale du backtest : {final_balance:.2f} USDT")

        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            string_columns = ['type', 'direction', 'zone', 'explanation']
            trades_df[string_columns] = trades_df[string_columns].fillna('-')
            numerical_columns = ['profit', 'entry_price', 'exit_price']
            trades_df[numerical_columns] = trades_df[numerical_columns].fillna(0)
            trades_df = trades_df[['type', 'direction', 'entry_price', 'exit_price', 'zone', 'profit', 'timestamp_entry', 'timestamp_exit', 'explanation']]
            trades_df.rename(columns={
                'type': 'Type de Trade',
                'direction': 'Direction',
                'entry_price': 'Prix d\'Entrée',
                'exit_price': 'Prix de Sortie',
                'zone': 'Zone',
                'profit': 'PnL',
                'timestamp_entry': 'Date d\'Entrée',
                'timestamp_exit': 'Date de Sortie',
                'explanation': 'Explications'
            }, inplace=True)
        else:
            trades_df = pd.DataFrame(columns=['Type de Trade', 'Direction', 'Prix d\'Entrée', 'Prix de Sortie', 'Zone', 'PnL', 'Date d\'Entrée', 'Date de Sortie', 'Explications'])

        fig_live.update_layout(
            height=800,
            title_text=f"PEPE/USDT - Données Historiques",
            xaxis_rangeslider_visible=False,
            yaxis_title='Prix de Clôture',
            xaxis_title='',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=50, b=50),
            uirevision='constant',
            showlegend=False,
            yaxis=dict(
                range=[df_display['close'].min() * 0.99, df_display['close'].max() * 1.05]
            )
        )

        if store_data and 'xaxis.range[0]' in store_data and 'xaxis.range[1]' in store_data:
            fig_live.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_live.update_xaxes(range=[df_display.index[0], df_display.index[-1]])

        fig_gaps_percentage = go.Figure()
        colors_gaps_percentage = get_bar_colors_percentage(df_display['sum_of_gaps_percentage'])

        fig_gaps_percentage.add_trace(go.Bar(
            x=df_display.index,
            y=df_display['sum_of_gaps_percentage'],
            marker_color=colors_gaps_percentage,
            name='Somme des Écarts EMA (%)',
            hoverinfo='text',
            hovertext=[f"Date: {x}<br>Somme des Écarts (%): {y:.4f}" for x, y in zip(df_display.index, df_display['sum_of_gaps_percentage'])],
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

        if store_data and 'xaxis.range[0]' in store_data and 'xaxis.range[1]' in store_data:
            fig_gaps_percentage.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_gaps_percentage.update_xaxes(range=[df_display.index[0], df_display.index[-1]])

        fig_slope_percentage = go.Figure()
        colors_slope_percentage = get_bar_colors_percentage(df_display['average_slope_percentage'])

        fig_slope_percentage.add_trace(go.Bar(
            x=df_display.index,
            y=df_display['average_slope_percentage'],
            marker_color=colors_slope_percentage,
            name='Moyenne des Pentes (%)',
            hoverinfo='text',
            hovertext=[f"Date: {x}<br>Moyenne des Pentes (%): {y:.4f}" for x, y in zip(df_display.index, df_display['average_slope_percentage'])],
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

        if store_data and 'xaxis.range[0]' in store_data and 'xaxis.range[1]' in store_data:
            fig_slope_percentage.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        else:
            fig_slope_percentage.update_xaxes(range=[df_display.index[0], df_display.index[-1]])

        fig_balance = go.Figure()
        balance_df = pd.DataFrame(balance_over_time)
        fig_balance.add_trace(go.Scatter(
            x=balance_df['timestamp'],
            y=balance_df['balance'],
            mode='lines',
            name='Balance',
            line=dict(color='blue')
        ))

        fig_balance.update_layout(
            height=400,
            title_text='Évolution de la Balance au Cours du Temps',
            xaxis_title='Temps',
            yaxis_title='Balance (USDT)',
            hovermode='x unified',
            margin=dict(l=10, r=50, t=50, b=50),
            showlegend=False,
            yaxis=dict(
                range=[balance_df['balance'].min() * 0.95, balance_df['balance'].max() * 1.05]
            )
        )

        if store_data and 'xaxis.range[0]' in store_data and 'xaxis.range[1]' in store_data and len(balance_df) > 0:
            fig_balance.update_xaxes(range=[store_data['xaxis.range[0]'], store_data['xaxis.range[1]']])
        elif len(balance_df) > 0:
            fig_balance.update_xaxes(range=[balance_df['timestamp'].iloc[0], balance_df['timestamp'].iloc[-1]])

        # Ajouter des marqueurs pour les trades et additions martingale dans le graphique de la balance
        # Points de fermeture de trade
        for trade in trades:
            if 'timestamp_exit' in trade and isinstance(trade['timestamp_exit'], pd.Timestamp):
                ts = trade['timestamp_exit']
                bal_point = balance_df.loc[balance_df['timestamp'] == ts, 'balance'].values
                if len(bal_point) > 0:
                    bal_point = bal_point[0]
                    color = 'green' if float(trade['profit']) > 0 else 'red'
                    fig_balance.add_trace(go.Scatter(
                        x=[ts],
                        y=[bal_point],
                        mode='markers+text',
                        marker=dict(color=color, size=10),
                        text=[f"PnL: {float(trade['profit']):.2f}"],
                        textposition='top center',
                        showlegend=False
                    ))

        # Points add_long (vert) et add_short (rouge)
        add_long_trades = trades_df[trades_df['Type de Trade'] == 'add_long']
        add_short_trades = trades_df[trades_df['Type de Trade'] == 'add_short']

        def get_balance_for_timestamp(ts):
            row = balance_df[balance_df['timestamp'] == ts]
            if not row.empty:
                return row['balance'].values[0]
            else:
                return None

        if not add_long_trades.empty:
            add_long_y = [get_balance_for_timestamp(ts) for ts in add_long_trades['Date d\'Entrée']]
            fig_balance.add_trace(go.Scatter(
                x=add_long_trades['Date d\'Entrée'],
                y=add_long_y,
                mode='markers',
                marker=dict(color='green', size=10, symbol='circle'),
                name='Add Long',
                text=add_long_trades['Explications'],
                hoverinfo='text',
                showlegend=True
            ))

        if not add_short_trades.empty:
            add_short_y = [get_balance_for_timestamp(ts) for ts in add_short_trades['Date d\'Entrée']]
            fig_balance.add_trace(go.Scatter(
                x=add_short_trades['Date d\'Entrée'],
                y=add_short_y,
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle'),
                name='Add Short',
                text=add_short_trades['Explications'],
                hoverinfo='text',
                showlegend=True
            ))

        trades_table_data = trades_df.to_dict('records')

        return (
            fig_live,
            fig_gaps_percentage,
            fig_slope_percentage,
            fig_balance,
            trades_table_data
        )
    except Exception as e:
        print(f"Erreur dans update_graph: {e}")
        return [go.Figure()] * 5

if __name__ == '__main__':
    print("Démarrage de l'application...")
    print(f"Veuillez ouvrir votre navigateur à l'adresse : http://127.0.0.1:8050/")
    app.run_server(debug=True, threaded=False)
