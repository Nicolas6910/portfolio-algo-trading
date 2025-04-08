import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*'H' is deprecated.*")

def calculate_custom_macd(df: pd.DataFrame, fast_length: int = 26, slow_length: int = 30, signal_length: int = 20) -> pd.DataFrame:
    fast_ema = df['close'].ewm(span=fast_length, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_length, adjust=False).mean()
    df['MACD'] = macd = fast_ema - slow_ema
    df['MACD_Signal'] = macd.ewm(span=signal_length, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def trendlines_with_breaks(df: pd.DataFrame, length: int = 15, mult: float = 1) -> pd.DataFrame:
    df['Pivot_High'] = df['high'].rolling(window=2*length+1, center=True).apply(lambda x: 1 if x[length] == max(x) else 0, raw=True)
    df['Pivot_Low'] = df['low'].rolling(window=2*length+1, center=True).apply(lambda x: 1 if x[length] == min(x) else 0, raw=True)
    tr = pd.DataFrame({'hl': df['high'] - df['low'], 'hc': np.abs(df['high'] - df['close'].shift(1)), 'lc': np.abs(df['low'] - df['close'].shift(1))})
    df['ATR'] = tr.max(axis=1).rolling(window=length).mean()
    df['slope'] = df['ATR'] / length * mult
    upper, lower, upos, dnos = df['close'].iloc[0], df['close'].iloc[0], 0, 0
    slope_ph, slope_pl = 0, 0
    upper_list, lower_list, upos_list, dnos_list = [], [], [], []
    for i, row in df.iterrows():
        if row['Pivot_High'] == 1:
            slope_ph, upper = row['slope'], row['high']
        else:
            upper -= slope_ph if slope_ph != 0 else 0
        if row['Pivot_Low'] == 1:
            slope_pl, lower = row['slope'], row['low']
        else:
            lower += slope_pl if slope_pl != 0 else 0
        upos = 0 if row['Pivot_High'] == 1 else (1 if slope_ph != 0 and row['close'] > upper - slope_ph * length else upos)
        dnos = 0 if row['Pivot_Low'] == 1 else (1 if slope_pl != 0 and row['close'] < lower + slope_pl * length else dnos)
        upper_list.append(upper)
        lower_list.append(lower) 
        upos_list.append(upos)
        dnos_list.append(dnos)
    df['upper'], df['lower'], df['upos'], df['dnos'] = upper_list, lower_list, upos_list, dnos_list
    df['Zone'] = np.where(df['upos'] > df['upos'].shift(1).fillna(0), 'Bullish', 
                          np.where(df['dnos'] > df['dnos'].shift(1).fillna(0), 'Bearish', 'Neutral'))
    return df

def trading_strategy(df: pd.DataFrame) -> Tuple[List[Dict], Dict, float, float, float, int, float]:
    position, balance = None, 0
    trades, yearly_results = [], {year: {'pnl': 0, 'trades': 0, 'winning_trades': 0} for year in df.index.year.unique()}
    total_max_profit, total_max_loss, total_trades, total_winning_trades = float('-inf'), float('inf'), 0, 0
    base_trade_size, leverage = 10, 50
    fee_percentage = 0.0001
    for i in range(1, len(df)):
        current_price, zone = df['close'].iloc[i], df['Zone'].iloc[i]
        prev_macd, prev_signal = df['MACD'].iloc[i-1], df['MACD_Signal'].iloc[i-1] 
        macd, signal = df['MACD'].iloc[i], df['MACD_Signal'].iloc[i]
        current_timestamp = df.index[i]
        current_year = current_timestamp.year
        trade_size = base_trade_size * (1.5 if zone == 'Bullish' else 0.5 if zone == 'Bearish' else 1)
        fee = trade_size * fee_percentage
        if prev_macd < prev_signal and macd > signal and zone != 'Bearish':
            if position == 'short':
                pnl = trade_size * leverage * (entry_price - current_price) / entry_price - 2 * fee
                balance += pnl
                trades.append({'action': 'close_short', 'price': current_price, 'pnl': pnl, 'timestamp': current_timestamp, 'balance': balance})
                yearly_results[current_year]['pnl'] += pnl
                yearly_results[current_year]['trades'] += 1
                if pnl > 0:
                    yearly_results[current_year]['winning_trades'] += 1
                    total_winning_trades += 1
                total_max_profit, total_max_loss = max(pnl, total_max_profit), min(pnl, total_max_loss)
                total_trades += 1
            position, entry_price = 'long', current_price
            balance -= fee
            trades.append({'action': 'open_long', 'price': current_price, 'pnl': -fee, 'timestamp': current_timestamp, 'balance': balance})
            yearly_results[current_year]['pnl'] -= fee
        elif prev_macd > prev_signal and macd < signal and zone != 'Bullish':
            if position == 'long':
                pnl = trade_size * leverage * (current_price - entry_price) / entry_price - 2 * fee
                balance += pnl
                trades.append({'action': 'close_long', 'price': current_price, 'pnl': pnl, 'timestamp': current_timestamp, 'balance': balance})
                yearly_results[current_year]['pnl'] += pnl
                yearly_results[current_year]['trades'] += 1
                if pnl > 0:
                    yearly_results[current_year]['winning_trades'] += 1
                    total_winning_trades += 1
                total_max_profit, total_max_loss = max(pnl, total_max_profit), min(pnl, total_max_loss)
                total_trades += 1
            position, entry_price = 'short', current_price
            balance -= fee
            trades.append({'action': 'open_short', 'price': current_price, 'pnl': -fee, 'timestamp': current_timestamp, 'balance': balance})
            yearly_results[current_year]['pnl'] -= fee
    if position:
        pnl = (trade_size * leverage * (current_price - entry_price) / entry_price if position == 'long' 
               else trade_size * leverage * (entry_price - current_price) / entry_price) - 2 * fee
        balance += pnl
        trades.append({'action': f'close_{position}', 'price': current_price, 'pnl': pnl, 'timestamp': df.index[-1], 'balance': balance})
        yearly_results[current_year]['pnl'] += pnl
        yearly_results[current_year]['trades'] += 1
        if pnl > 0:
            yearly_results[current_year]['winning_trades'] += 1
            total_winning_trades += 1
        total_max_profit, total_max_loss = max(pnl, total_max_profit), min(pnl, total_max_loss)
        total_trades += 1
    winrate = (total_winning_trades / total_trades * 100) if total_trades > 0 else 0
    return trades, yearly_results, balance, total_max_profit, total_max_loss, total_trades, winrate

def visualize_trades_and_indicators(data: pd.DataFrame, trades: List[Dict], total_pnl: float) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=('Prix et Trades', 'Évolution de la Balance', 'MACD'))
    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Prix'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['upper'], mode='lines', name='Upper Trend', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['lower'], mode='lines', name='Lower Trend', line=dict(color='red')), row=1, col=1)
    long_entries = [trade for trade in trades if trade['action'] == 'open_long']
    short_entries = [trade for trade in trades if trade['action'] == 'open_short']
    fig.add_trace(go.Scatter(x=[trade['timestamp'] for trade in long_entries], y=[trade['price'] for trade in long_entries],
                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Long Entry'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[trade['timestamp'] for trade in short_entries], y=[trade['price'] for trade in short_entries],
                             mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Short Entry'), row=1, col=1)
    balance_evolution = pd.Series([trade['balance'] for trade in trades], index=pd.to_datetime([trade['timestamp'] for trade in trades]))
    fig.add_trace(go.Scatter(x=balance_evolution.index, y=balance_evolution.values, mode='lines', name='Balance', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal', line=dict(color='orange')), row=3, col=1)
    colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], marker_color=colors, name='MACD Histogram'), row=3, col=1)
    fig.update_layout(title=f'Graphique des trades, Balance et Indicateurs (Balance finale : {total_pnl:.2f} USDT)', 
                      height=1200, width=1200, xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_yaxes(title_text='Prix (USDT)', row=1, col=1)
    fig.update_yaxes(title_text='Balance (USDT)', row=2, col=1)
    fig.update_yaxes(title_text='MACD', row=3, col=1)
    return fig

def run_backtest(data_file: str):
    try:
        data = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')
        required_columns = {'open', 'high', 'low', 'close'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Le fichier de données doit contenir les colonnes suivantes: {required_columns}")
        data = calculate_custom_macd(data)
        data_4h = data.resample('4H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        data_4h = trendlines_with_breaks(data_4h)
        data = data.merge(data_4h[['Zone', 'upper', 'lower']], left_index=True, right_index=True, how='left')
        data['Zone'] = data['Zone'].ffill().fillna('Neutral')
        data[['upper', 'lower']] = data[['upper', 'lower']].ffill()
        trades, yearly_results, total_pnl, total_max_profit, total_max_loss, total_trades, winrate = trading_strategy(data)
        fig = visualize_trades_and_indicators(data, trades, total_pnl)
        fig.write_html("trades_indicators_visualization.html")
        return trades, yearly_results, total_pnl, total_max_profit, total_max_loss, total_trades, winrate
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return [], {}, 0.0, 0.0, 0.0, 0, 0.0

if __name__ == "__main__":
    data_file = r'C:\\Users\\nicol\\Desktop\\Python API\\BTC_Historical_Data\\30m_BTC_Candles\\btc_usdt_30min_2024.csv'
    trades, yearly_results, total_pnl, total_max_profit, total_max_loss, total_trades, winrate = run_backtest(data_file)
    if trades:
        print(f"\nBalance finale : {total_pnl:.2f} USDT")
        print(f"Profit maximal : {total_max_profit:.2f} USDT")
        print(f"Perte maximale : {total_max_loss:.2f} USDT")
        print(f"Nombre total de trades : {total_trades}")
        print(f"Winrate global : {winrate:.2f}%")
        print("\nLe graphique des trades, de la balance et des indicateurs a été sauvegardé dans 'trades_indicators_visualization.html'")
        print("\nRésultats annuels :")
        for year, results in yearly_results.items():
            winrate_year = (results['winning_trades'] / results['trades'] * 100) if results['trades'] > 0 else 0
            print(f"{year}: PnL = {results['pnl']:.2f} USDT, Trades = {results['trades']}, Winrate = {winrate_year:.2f}%")
        total_pnl_check = sum(results['pnl'] for results in yearly_results.values())
        print(f"\nSomme des PnL annuels : {total_pnl_check:.2f} USDT")
    else:
        print("Le backtest n'a pas pu être exécuté en raison d'une erreur.")
