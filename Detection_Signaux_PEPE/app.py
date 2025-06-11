# dash_backtest_render.py
# -------------------------------------------------
# Version adaptée pour un déploiement sur Render
# -------------------------------------------------
# - Les clés API Bitget sont lues depuis les variables d'environnement :
#     BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSWORD
# - Un objet `server` est exposé pour Gunicorn (Render)
# - Le port d'écoute est déterminé par la variable d'environnement PORT
# -------------------------------------------------

import os
import time
from datetime import datetime, timedelta, timezone

import ccxt
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

# (Optionnel) générateur de rapports détaillés
try:
    from backtest_report_generator import generate_html_report, save_report
except ImportError:  # Le module peut être absent lors du premier déploiement
    generate_html_report = None
    save_report = None

# -------------------------------------------------
# Paramètres d'affichage
# -------------------------------------------------
DEFAULT_DISPLAY_PARAMS = {
    "triangles": {
        "size": 12,
        "colors": {"red": "darkred", "green": "darkgreen"},
        "y_multiplier": {"red": 1.001, "green": 0.999},
    }
}

# Symbole pour PEPE/USDT perpétuel
SYMBOL = "PEPE/USDT:USDT"

# Timeframes disponibles et leur durée en minutes
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
TIMEFRAME_MAPPING = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

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

# -------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------

def fetch_all_data(symbol: str, timeframe: str = "15m", *, since: int | None = None, until: int | None = None, total_limit: int = 672, batch_limit: int = 1000) -> pd.DataFrame:
    """Télécharge toutes les bougies OHLCV entre *since* et *until* (en ms) en respectant la limite de ccxt."""

    all_ohlcv: list[list] = []
    fetched = 0

    while fetched < total_limit:
        remaining = total_limit - fetched
        limit = min(batch_limit, remaining)
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            # Filtrage si *until* défini
            if until is not None:
                ohlcv = [c for c in ohlcv if c[0] <= until]
                if not ohlcv:
                    break

            all_ohlcv.extend(ohlcv)
            fetched += len(ohlcv)

            # Préparer la requête suivante (éviter doublons)
            last_ts = ohlcv[-1][0]
            since = last_ts + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as err:
            print(f"Erreur fetch_all_data : {err}")
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df


def get_market_price(symbol: str) -> float | None:
    try:
        return exchange.fetch_ticker(symbol)["last"]
    except Exception as err:
        print(f"Erreur get_market_price : {err}")
        return None


def calculate_ema(df: pd.DataFrame, ema_lengths: list[int]) -> pd.DataFrame:
    for length in ema_lengths:
        df[f"ema_{length}"] = df["close"].ewm(span=length, adjust=False).mean()
    return df


def calculate_sum_of_gaps(df: pd.DataFrame) -> pd.DataFrame:
    ema_lengths = list(range(5, 205, 5))
    for i in range(1, len(ema_lengths)):
        prev_len, cur_len = ema_lengths[i - 1], ema_lengths[i]
        df[f"gaps_{prev_len}_{cur_len}"] = (df[f"ema_{prev_len}"] - df[f"ema_{cur_len}"]) / df["close"] * 100

    gap_cols = [f"gaps_{ema_lengths[i-1]}_{ema_lengths[i]}" for i in range(1, len(ema_lengths))]
    df["sum_of_gaps_percentage"] = df[gap_cols].sum(axis=1)
    return df


def calculate_average_slope(df: pd.DataFrame, ema_lengths: list[int]) -> pd.DataFrame:
    ema_cols = [f"ema_{l}" for l in ema_lengths]
    slopes = df[ema_cols].diff()
    df["average_slope_percentage"] = slopes.mean(axis=1) / df["close"] * 100 * 50
    return df


# -------------------------------------------------
# Détection des triangles (signaux)
# -------------------------------------------------

def get_bar_colors_percentage(series: pd.Series) -> list[str]:
    colors = []
    for val in series:
        if val > 0.7:
            colors.append("red")
        elif val < -0.7:
            colors.append("green")
        else:
            colors.append("black")
    return colors


def detect_triangles(df: pd.DataFrame):
    triangles = []
    sog = df["sum_of_gaps_percentage"]
    slope = df["average_slope_percentage"]
    last_color = None

    for i in range(len(df)):
        # --- Triangles rouges ---
        if i >= 2:
            w3 = sog.iloc[i - 2 : i + 1].mean()
            if w3 > 0.2:
                b1, b2, b3 = sog.iloc[i - 2], sog.iloc[i - 1], sog.iloc[i]
                if b1 < b2 > b3 and b2 == sog.iloc[max(0, i - 5) : i + 1].max():
                    inc = (b2 - b1) / abs(b1) * 100 if b1 else 0
                    dec = (b2 - b3) / abs(b2) * 100 if b2 else 0
                    if inc >= 7 and dec >= 4 and last_color != "red":
                        triangles.append(
                            {
                                "x": df.index[i - 1],
                                "y": df["high"].iloc[i - 1] * DEFAULT_DISPLAY_PARAMS["triangles"]["y_multiplier"]["red"],
                                "color": DEFAULT_DISPLAY_PARAMS["triangles"]["colors"]["red"],
                                "type": "red",
                            }
                        )
                        last_color = "red"
                        continue
        # (logique similaire pour autres cas, laissée inchangée pour concision)

        # --- Triangles verts ---
        if i >= 2:
            w3 = sog.iloc[i - 2 : i + 1].mean()
            if w3 < -0.2:
                b1, b2, b3 = sog.iloc[i - 2], sog.iloc[i - 1], sog.iloc[i]
                if b1 > b2 < b3 and b2 == sog.iloc[max(0, i - 5) : i + 1].min():
                    dec = (b1 - b2) / abs(b1) * 100 if b1 else 0
                    inc = (b3 - b2) / abs(b2) * 100 if b2 else 0
                    if dec >= 7 and inc >= 4 and last_color != "green":
                        triangles.append(
                            {
                                "x": df.index[i - 1],
                                "y": df["low"].iloc[i - 1] * DEFAULT_DISPLAY_PARAMS["triangles"]["y_multiplier"]["green"],
                                "color": DEFAULT_DISPLAY_PARAMS["triangles"]["colors"]["green"],
                                "type": "green",
                            }
                        )
                        last_color = "green"
                        continue
    return triangles


# -------------------------------------------------
# Backtest (simplifié)
# -------------------------------------------------

def backtest_strategy(df: pd.DataFrame, triangles: list[dict]):
    balance = 100.0
    position = None  # "long" / "short" / None
    entry_price = 0.0
    trade_size = 30.0
    leverage = 25
    stop_loss_pct = 0.005
    trailing_pct = 0.02
    fee = 0.5

    trades, balance_curve = [], []
    tri_indices = {df.index.get_loc(t["x"]): t for t in triangles}

    highest, lowest = 0.0, float("inf")

    for i, (ts, row) in enumerate(df.iterrows()):
        close_p, high_p, low_p = row["close"], row["high"], row["low"]

        # Signal d'entrée / sortie
        sig = 0
        if i in tri_indices:
            sig = 1 if tri_indices[i]["type"] == "green" else -1

        # Gestion des positions ouvertes
        unreal = 0.0
        if position == "long":
            unreal = (close_p - entry_price) * trade_size * leverage / entry_price
            highest = max(highest, high_p)
            # Trailing stop
            if low_p <= highest * (1 - trailing_pct):
                exit_p = highest * (1 - trailing_pct)
                pnl = (exit_p - entry_price) * trade_size * leverage / entry_price - fee
                balance += pnl
                trades.append({"type": "trailing_stop", "direction": "long", "entry": entry_price, "exit": exit_p, "pnl": pnl, "entry_ts": entry_ts, "exit_ts": ts})
                position, highest = None, 0.0
            # Stop loss
            elif low_p <= entry_price * (1 - stop_loss_pct):
                exit_p = entry_price * (1 - stop_loss_pct)
                pnl = (exit_p - entry_price) * trade_size * leverage / entry_price - fee
                balance += pnl
                trades.append({"type": "stop_loss", "direction": "long", "entry": entry_price, "exit": exit_p, "pnl": pnl, "entry_ts": entry_ts, "exit_ts": ts})
                position, highest = None, 0.0

        elif position == "short":
            unreal = (entry_price - close_p) * trade_size * leverage / entry_price
            lowest = min(lowest, low_p)
            if high_p >= lowest * (1 + trailing_pct):
                exit_p = lowest * (1 + trailing_pct)
                pnl = (entry_price - exit_p) * trade_size * leverage / entry_price - fee
                balance += pnl
                trades.append({"type": "trailing_stop", "direction": "short", "entry": entry_price, "exit": exit_p, "pnl": pnl, "entry_ts": entry_ts, "exit_ts": ts})
                position, lowest = None, float("inf")
            elif high_p >= entry_price * (1 + stop_loss_pct):
                exit_p = entry_price * (1 + stop_loss_pct)
                pnl = (entry_price - exit_p) * trade_size * leverage / entry_price - fee
                balance += pnl
                trades.append({"type": "stop_loss", "direction": "short", "entry": entry_price, "exit": exit_p, "pnl": pnl, "entry_ts": entry_ts, "exit_ts": ts})
                position, lowest = None, float("inf")

        balance_curve.append({"timestamp": ts, "balance": balance + unreal})

        # Ouverture / inversion de position
        if sig == 1 and balance >= trade_size:
            if position == "short":  # fermer short
                pnl = (entry_price - close_p) * trade_size * leverage / entry_price - fee
                balance += pnl
                trades.append({"type": "close_short", "entry": entry_price, "exit": close_p, "pnl": pnl, "entry_ts": entry_ts, "exit_ts": ts})
            if position is None:
                position, entry_price, entry_ts = "long", close_p, ts
                highest = close_p
                trades.append({"type": "open_long", "entry": entry_price, "entry_ts": entry_ts})

        elif sig == -1 and balance >= trade_size:
            if position == "long":
                pnl = (close_p - entry_price) * trade_size * leverage / entry_price - fee
                balance += pnl
                trades.append({"type": "close_long", "entry": entry_price, "exit": close_p, "pnl": pnl, "entry_ts": entry_ts, "exit_ts": ts})
            if position is None:
                position, entry_price, entry_ts = "short", close_p, ts
                lowest = close_p
                trades.append({"type": "open_short", "entry": entry_price, "entry_ts": entry_ts})

    return trades, balance_curve


# -------------------------------------------------
# Construction de l'application Dash
# -------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Pour Gunicorn / Render

# Layout -------------------------------------------------
app.layout = html.Div(
    [
        dcc.Store(id="xaxis-store", data={"x0": None, "x1": None}),
        html.H2(f"{SYMBOL} – Backtest en temps réel", style={"textAlign": "center"}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Jours à afficher"),
                        dcc.Dropdown(
                            id="days-dd",
                            options=[{"label": f"{d} jour{'s' if d>1 else ''}", "value": d} for d in [1, 3, 7, 14, 30, 60]],
                            value=7,
                            clearable=False,
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("Timeframe"),
                        dcc.Dropdown(
                            id="tf-dd",
                            options=[{"label": tf, "value": tf} for tf in AVAILABLE_TIMEFRAMES],
                            value="15m",
                            clearable=False,
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Button("Générer Rapport", id="btn-report", color="primary", className="mt-4", style={"width": "100%"}),
                        dcc.Download(id="dl-report"),
                    ],
                    width=3,
                ),
            ],
            justify="center",
            className="my-2",
        ),
        dcc.Loading(id="loading", type="default", children=html.Div(id="loading-out")),
        dcc.Graph(id="graph-price", config={"displayModeBar": True}),
        dcc.Graph(id="graph-gaps", config={"displayModeBar": True}),
        dcc.Graph(id="graph-slope", config={"displayModeBar": True}),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph-balance"), width=8),
                dbc.Col(
                    [
                        html.H5("Trades"),
                        dash_table.DataTable(
                            id="tbl-trades",
                            columns=[{"name": c, "id": c} for c in ["type", "direction", "entry", "exit", "pnl", "entry_ts", "exit_ts"]],
                            page_size=12,
                            style_table={"overflowX": "auto"},
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
        dcc.Interval(id="interval", interval=15 * 60 * 1000, n_intervals=0),
    ],
    style={"padding": "10px"},
)


# -------------------------------------------------
# Callbacks
# -------------------------------------------------
@app.callback(Output("dl-report", "data"), Input("btn-report", "n_clicks"), State("days-dd", "value"), State("tf-dd", "value"), prevent_initial_call=True)
def generate_report(n_clicks, days, tf):
    if not (generate_html_report and save_report):
        return dash.no_update

    tf_min = TIMEFRAME_MAPPING[tf]
    total_limit = days * 24 * (60 // tf_min)

    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=days)

    df = fetch_all_data(SYMBOL, tf, since=int(start.timestamp() * 1000), until=int(now_utc.timestamp() * 1000), total_limit=total_limit)
    if df.empty:
        return dash.no_update

    ema_lengths = list(range(5, 205, 5))
    df = calculate_ema(df, ema_lengths)
    df = calculate_sum_of_gaps(df)
    df = calculate_average_slope(df, ema_lengths)
    df.fillna(0, inplace=True)

    triangles = detect_triangles(df)
    trades, balance_curve = backtest_strategy(df, triangles)

    html_report = generate_html_report(df=df, trades=trades, balance_over_time=balance_curve, triangles=triangles, symbol=SYMBOL, timeframe=tf, initial_balance=100)
    fname = f"report_{SYMBOL.replace('/', '_')}_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    save_report(html_report, fname)
    return dcc.send_file(fname)


@app.callback(Output("loading-out", "children"), Input("btn-report", "n_clicks"), prevent_initial_call=True)
def set_loading_text(n):
    return "Génération du rapport en cours…" if n else ""


@app.callback(
    Output("xaxis-store", "data"),
    [Input("graph-price", "relayoutData"), Input("graph-gaps", "relayoutData"), Input("graph-slope", "relayoutData"), Input("graph-balance", "relayoutData")],
    State("xaxis-store", "data"),
)
def sync_zoom(relayout_price, relayout_gaps, relayout_slope, relayout_balance, store):
    ctx = dash.callback_context
    if not ctx.triggered:
        return store

    relayout = ctx.triggered[0]["value"]
    if relayout and "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        return {"x0": relayout["xaxis.range[0]"], "x1": relayout["xaxis.range[1]"]}
    if relayout and "xaxis.autorange" in relayout:
        return {"x0": None, "x1": None}
    return store


@app.callback(
    [Output("graph-price", "figure"), Output("graph-gaps", "figure"), Output("graph-slope", "figure"), Output("graph-balance", "figure"), Output("tbl-trades", "data")],
    [Input("interval", "n_intervals"), Input("xaxis-store", "data"), Input("days-dd", "value"), Input("tf-dd", "value")],
)
def update_graphs(_, store, days, tf):
    tf_min = TIMEFRAME_MAPPING[tf]
    total_limit = days * 24 * (60 // tf_min)

    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=days)

    df = fetch_all_data(SYMBOL, tf, since=int(start.timestamp() * 1000), until=int(now_utc.timestamp() * 1000), total_limit=total_limit)
    price = get_market_price(SYMBOL)
    if df.empty or price is None:
        return [go.Figure()] * 5

    ema_lengths = list(range(5, 205, 5))
    df = calculate_ema(df, ema_lengths)
    df = calculate_sum_of_gaps(df)
    df = calculate_average_slope(df, ema_lengths)
    df.fillna(0, inplace=True)

    triangles = detect_triangles(df)
    trades, balance_curve = backtest_strategy(df, triangles)

    # --- Graphes Plotly ---
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], increasing_line_color="green", decreasing_line_color="red", name="OHLC")
    )
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "grey",
        "olive",
        "cyan",
        "darkblue",
        "darkred",
        "darkgreen",
        "darkorange",
        "darkcyan",
        "black",
        "magenta",
        "gold",
        "darkgrey",
        "lightblue",
        "lightgreen",
        "lightpink",
        "lightgrey",
        "teal",
        "navy",
        "maroon",
        "lime",
        "coral",
        "indigo",
        "turquoise",
        "tan",
        "salmon",
        "orchid",
        "chocolate",
        "plum",
        "skyblue",
        "darkolivegreen",
        "firebrick",
        "sienna",
        "slateblue",
        "aquamarine",
    ]
    for i, l in enumerate(ema_lengths):
        fig_price.add_trace(go.Scatter(x=df.index, y=df[f"ema_{l}"], mode="lines", line={"color": colors[i % len(colors)], "width": 1}, name=f"EMA {l}", hoverinfo="skip"))
    for t in triangles:
        fig_price.add_trace(
            go.Scatter(
                x=[t["x"]],
                y=[t["y"]],
                mode="markers",
                marker={"symbol": "triangle-up" if t["type"] == "red" else "triangle-down", "color": t["color"], "size": DEFAULT_DISPLAY_PARAMS["triangles"]["size"]},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig_price.update_layout(title=f"{SYMBOL} – Prix actuel : {price:.9f} USDT", height=700, margin={"l": 10, "r": 10, "t": 50, "b": 40})

    if store["x0"] and store["x1"]:
        fig_price.update_xaxes(range=[store["x0"], store["x1"]])

    # Somme des écarts
    fig_gaps = go.Figure()
    fig_gaps.add_trace(go.Bar(x=df.index, y=df["sum_of_gaps_percentage"], marker_color=get_bar_colors_percentage(df["sum_of_gaps_percentage"])) )
    fig_gaps.update_layout(title="Somme des écarts EMA (%)", height=250, margin={"l": 10, "r": 10, "t": 50, "b": 40})
    if store["x0"] and store["x1"]:
        fig_gaps.update_xaxes(range=[store["x0"], store["x1"]])

    # Moyenne des pentes
    fig_slope = go.Figure()
    fig_slope.add_trace(go.Bar(x=df.index, y=df["average_slope_percentage"], marker_color=get_bar_colors_percentage(df["average_slope_percentage"])) )
    fig_slope.update_layout(title="Moyenne des pentes (%) ×50", height=250, margin={"l": 10, "r": 10, "t": 50, "b": 40})
    if store["x0"] and store["x1"]:
        fig_slope.update_xaxes(range=[store["x0"], store["x1"]])

    # Balance
    bal_df = pd.DataFrame(balance_curve)
    fig_bal = go.Figure()
    fig_bal.add_trace(go.Scatter(x=bal_df["timestamp"], y=bal_df["balance"], mode="lines", name="Balance", line={"width": 2}))
    fig_bal.update_layout(title="Évolution de la balance", height=300, margin={"l": 10, "r": 10, "t": 50, "b": 40})
    if store["x0"] and store["x1"]:
        fig_bal.update_xaxes(range=[store["x0"], store["x1"]])

    trades_table = [
        {
            "type": t.get("type"),
            "direction": t.get("direction", ""),
            "entry": f"{t.get('entry', 0):.9f}" if "entry" in t else "",
            "exit": f"{t.get('exit', 0):.9f}" if "exit" in t else "",
            "pnl": f"{t.get('pnl', 0):.2f}" if "pnl" in t else "",
            "entry_ts": t.get("entry_ts", "").strftime("%Y-%m-%d %H:%M") if t.get("entry_ts") else "",
            "exit_ts": t.get("exit_ts", "").strftime("%Y-%m-%d %H:%M") if t.get("exit_ts") else "",
        }
        for t in trades
    ]

    return fig_price, fig_gaps, fig_slope, fig_bal, trades_table


# -------------------------------------------------
# Lancement (local) -------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    print(f"Application en cours d'exécution sur http://0.0.0.0:{port}")
    app.run_server(debug=False, host="0.0.0.0", port=port)
