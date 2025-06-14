import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def calculate_performance_metrics(trades, balance_over_time, initial_balance=100):
    """
    Calcule tous les KPIs de performance nécessaires pour le rapport
    """
    if not trades or not balance_over_time:
        return {}
    
    # Convertir en DataFrame pour faciliter les calculs
    trades_df = pd.DataFrame([t for t in trades if 'profit' in t])
    balance_df = pd.DataFrame(balance_over_time)
    
    # Calculs de base
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit'] > 0])
    losing_trades = len(trades_df[trades_df['profit'] < 0])
    
    # Métriques de performance
    metrics = {
        'total_net_profit': balance_df['balance'].iloc[-1] - initial_balance if len(balance_df) > 0 else 0,
        'total_gross_profit': trades_df[trades_df['profit'] > 0]['profit'].sum() if winning_trades > 0 else 0,
        'total_gross_loss': trades_df[trades_df['profit'] < 0]['profit'].sum() if losing_trades > 0 else 0,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'average_win': trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0,
        'average_loss': trades_df[trades_df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0,
        'largest_win': trades_df[trades_df['profit'] > 0]['profit'].max() if winning_trades > 0 else 0,
        'largest_loss': trades_df[trades_df['profit'] < 0]['profit'].min() if losing_trades > 0 else 0,
    }
    
    # Profit Factor
    if metrics['total_gross_loss'] != 0:
        metrics['profit_factor'] = abs(metrics['total_gross_profit'] / metrics['total_gross_loss'])
    else:
        metrics['profit_factor'] = float('inf') if metrics['total_gross_profit'] > 0 else 0
    
    # Ratio Gain/Perte
    if metrics['average_loss'] != 0:
        metrics['win_loss_ratio'] = abs(metrics['average_win'] / metrics['average_loss'])
    else:
        metrics['win_loss_ratio'] = float('inf') if metrics['average_win'] > 0 else 0
    
    # Drawdown
    running_balance = balance_df['balance'].values
    running_max = np.maximum.accumulate(running_balance)
    drawdown = (running_balance - running_max) / running_max * 100
    metrics['max_drawdown'] = drawdown.min()
    metrics['max_drawdown_value'] = (running_balance - running_max).min()
    
    # Durée moyenne des trades
    if total_trades > 0:
        trades_with_duration = []
        for trade in trades:
            if 'timestamp_entry' in trade and 'timestamp_exit' in trade and trade.get('timestamp_exit'):
                duration = (trade['timestamp_exit'] - trade['timestamp_entry']).total_seconds() / 3600  # en heures
                trades_with_duration.append(duration)
        
        if trades_with_duration:
            metrics['average_trade_duration'] = np.mean(trades_with_duration)
        else:
            metrics['average_trade_duration'] = 0
    else:
        metrics['average_trade_duration'] = 0
    
    # Calcul des rendements journaliers pour Sharpe et Sortino
    if len(balance_df) > 1:
        balance_df['returns'] = balance_df['balance'].pct_change()
        daily_returns = balance_df['returns'].dropna()
        
        if len(daily_returns) > 0:
            # Sharpe Ratio (annualisé)
            avg_return = daily_returns.mean()
            std_return = daily_returns.std()
            metrics['sharpe_ratio'] = (avg_return / std_return * np.sqrt(252)) if std_return != 0 else 0
            
            # Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            metrics['sortino_ratio'] = (avg_return / downside_std * np.sqrt(252)) if downside_std != 0 else 0
            
            # Calmar Ratio
            annual_return = (balance_df['balance'].iloc[-1] / initial_balance) ** (252 / len(balance_df)) - 1
            metrics['calmar_ratio'] = (annual_return / abs(metrics['max_drawdown']) * 100) if metrics['max_drawdown'] != 0 else 0
            
            # Volatilité
            metrics['volatility'] = std_return * np.sqrt(252) * 100
        else:
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0
            metrics['calmar_ratio'] = 0
            metrics['volatility'] = 0
    else:
        metrics['sharpe_ratio'] = 0
        metrics['sortino_ratio'] = 0
        metrics['calmar_ratio'] = 0
        metrics['volatility'] = 0
    
    return metrics

def generate_trade_details_json(trades):
    """
    Génère les données des trades au format JSON pour le tableau interactif
    """
    trade_details = []
    for i, trade in enumerate(trades):
        if 'profit' in trade:  # Trade fermé
            trade_detail = {
                'id': i + 1,
                'type': trade.get('type', ''),
                'direction': trade.get('direction', ''),
                'entry_time': trade.get('timestamp_entry').strftime('%Y-%m-%d %H:%M:%S') if trade.get('timestamp_entry') else '',
                'exit_time': trade.get('timestamp_exit').strftime('%Y-%m-%d %H:%M:%S') if trade.get('timestamp_exit') else '',
                'entry_price': float(trade.get('entry_price', 0)),
                'exit_price': float(trade.get('exit_price', 0)),
                'profit': float(trade.get('profit', 0)),
                'profit_percent': (float(trade.get('profit', 0)) / 30) * 100,  # Basé sur la taille du trade de 30 USDT
                'duration_hours': 0
            }
            
            # Calcul de la durée
            if trade.get('timestamp_entry') and trade.get('timestamp_exit'):
                duration = (trade['timestamp_exit'] - trade['timestamp_entry']).total_seconds() / 3600
                trade_detail['duration_hours'] = round(duration, 2)
            
            trade_details.append(trade_detail)
    
    return trade_details

def generate_performance_by_period(trades, balance_over_time):
    """
    Génère les données de performance par période (jour, semaine, mois)
    """
    if not trades or not balance_over_time:
        return {}, {}, {}
    
    # Convertir en DataFrame
    balance_df = pd.DataFrame(balance_over_time)
    balance_df['date'] = pd.to_datetime(balance_df['timestamp'])
    balance_df.set_index('date', inplace=True)
    
    # Performance journalière
    daily_perf = balance_df.resample('D').last()
    daily_perf['returns'] = daily_perf['balance'].pct_change() * 100
    daily_perf['profit'] = daily_perf['balance'].diff()
    
    # Performance hebdomadaire
    weekly_perf = balance_df.resample('W').last()
    weekly_perf['returns'] = weekly_perf['balance'].pct_change() * 100
    weekly_perf['profit'] = weekly_perf['balance'].diff()
    
    # Performance mensuelle
    monthly_perf = balance_df.resample('M').last()
    monthly_perf['returns'] = monthly_perf['balance'].pct_change() * 100
    monthly_perf['profit'] = monthly_perf['balance'].diff()
    
    return daily_perf, weekly_perf, monthly_perf

def generate_html_report(df, trades, balance_over_time, triangles, symbol, timeframe, initial_balance=100):
    """
    Génère le rapport HTML complet du backtest
    """
    # Calculer les métriques
    metrics = calculate_performance_metrics(trades, balance_over_time, initial_balance)
    trade_details = generate_trade_details_json(trades)
    
    # Dates de début et fin
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    
    # Générer les données pour les graphiques
    balance_df = pd.DataFrame(balance_over_time)
    equity_curve_data = [
        {'x': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 'y': row['balance']} 
        for _, row in balance_df.iterrows()
    ]
    
    # Calcul du drawdown
    running_balance = balance_df['balance'].values
    running_max = np.maximum.accumulate(running_balance)
    drawdown = (running_balance - running_max) / running_max * 100
    drawdown_data = [
        {'x': balance_df.iloc[i]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 'y': drawdown[i]} 
        for i in range(len(drawdown))
    ]
    
    # Distribution des profits/pertes
    profit_distribution = []
    for trade in trades:
        if 'profit' in trade:
            profit_distribution.append(float(trade['profit']))
    
    # Performance par période
    daily_perf, weekly_perf, monthly_perf = generate_performance_by_period(trades, balance_over_time)
    
    # Statistiques des trades par type
    trade_stats = {
        'long': {'count': 0, 'wins': 0, 'losses': 0, 'total_profit': 0},
        'short': {'count': 0, 'wins': 0, 'losses': 0, 'total_profit': 0}
    }
    
    for trade in trades:
        if 'profit' in trade and 'direction' in trade:
            direction = trade['direction']
            if direction in trade_stats:
                trade_stats[direction]['count'] += 1
                trade_stats[direction]['total_profit'] += trade['profit']
                if trade['profit'] > 0:
                    trade_stats[direction]['wins'] += 1
                else:
                    trade_stats[direction]['losses'] += 1
    
    # Calcul des statistiques par type de sortie
    exit_stats = {
        'close_long': 0,
        'close_short': 0,
        'stop_loss': 0,
        'trailing_stop': 0
    }
    
    for trade in trades:
        if 'type' in trade and trade['type'] in exit_stats:
            exit_stats[trade['type']] += 1
    
    # Heatmap mensuelle
    heatmap_data = []
    for date, row in monthly_perf.iterrows():
        if pd.notna(row['returns']):
            heatmap_data.append({
                'date': date.strftime('%Y-%m'),
                'value': round(row['returns'], 2)
            })
    
    # Génération du HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Backtest - {symbol} {timeframe}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css">
    
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- Custom CSS -->
    <style>
        :root {{
            --primary-color: #2c3e50;
            --success-color: #27ae60;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --info-color: #3498db;
            --dark-bg: #1a1a1a;
            --light-bg: #ecf0f1;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #333;
        }}
        
        .navbar {{
            background-color: var(--primary-color);
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
        }}
        
        .card {{
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
            margin-bottom: 20px;
            transition: transform 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,.15);
        }}
        
        .metric-card {{
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        .positive {{
            color: var(--success-color);
        }}
        
        .negative {{
            color: var(--danger-color);
        }}
        
        .section-title {{
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 10px;
            margin-bottom: 30px;
            color: var(--primary-color);
        }}
        
        .tooltip-custom {{
            cursor: help;
            text-decoration: underline dotted;
        }}
        
        .table-container {{
            max-height: 500px;
            overflow-y: auto;
        }}
        
        .summary-box {{
            background-color: #f8f9fa;
            border-left: 4px solid var(--info-color);
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
        }}
        
        #navbar-content {{
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        
        .trade-row-positive {{
            background-color: rgba(39, 174, 96, 0.1);
        }}
        
        .trade-row-negative {{
            background-color: rgba(231, 76, 60, 0.1);
        }}
        
        .dark-mode {{
            background-color: var(--dark-bg);
            color: #f8f9fa;
        }}
        
        .dark-mode .card {{
            background-color: #2c2c2c;
            color: #f8f9fa;
        }}
        
        .dark-mode .table {{
            color: #f8f9fa;
        }}
        
        @media print {{
            .no-print {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-dark navbar-expand-lg" id="navbar-content">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-chart-line"></i> Rapport de Backtest
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="#resume">Résumé</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#kpis">KPIs</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#graphiques">Graphiques</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#trades">Trades</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#analyse">Analyse</a>
                    </li>
                </ul>
                <div class="ms-auto">
                    <button class="btn btn-outline-light no-print" onclick="window.print()">
                        <i class="fas fa-print"></i> Imprimer
                    </button>
                    <button class="btn btn-outline-light no-print" onclick="exportToCSV()">
                        <i class="fas fa-download"></i> Export CSV
                    </button>
                    <button class="btn btn-outline-light no-print" onclick="toggleDarkMode()">
                        <i class="fas fa-moon"></i>
                    </button>
                </div>
            </div>
        </div>
    </nav>
    
    <!-- Container principal -->
    <div class="container-fluid mt-4">
        <!-- En-tête -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h1 class="card-title">Stratégie de Trading - Détection de Triangles</h1>
                        <div class="row">
                            <div class="col-md-3">
                                <p><strong>Marché:</strong> {symbol}</p>
                            </div>
                            <div class="col-md-3">
                                <p><strong>Période:</strong> {start_date} - {end_date}</p>
                            </div>
                            <div class="col-md-3">
                                <p><strong>Timeframe:</strong> {timeframe}</p>
                            </div>
                            <div class="col-md-3">
                                <p><strong>Version:</strong> 1.0</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Résumé exécutif -->
        <section id="resume">
            <h2 class="section-title">Résumé Exécutif</h2>
            <div class="row">
                <div class="col-12">
                    <div class="summary-box">
                        <h4>Performance Globale</h4>
                        <p>La stratégie a généré un profit net de <strong>{metrics['total_net_profit']:.2f} USDT</strong> 
                        sur un capital initial de {initial_balance} USDT, soit un rendement de 
                        <strong>{(metrics['total_net_profit']/initial_balance*100):.2f}%</strong>.</p>
                        <p>Avec un taux de réussite de <strong>{metrics['win_rate']:.1f}%</strong> sur 
                        <strong>{metrics['total_trades']}</strong> trades, la stratégie montre une 
                        {'performance positive' if metrics['total_net_profit'] > 0 else 'performance négative'} 
                        avec un drawdown maximum de <strong>{abs(metrics['max_drawdown']):.2f}%</strong>.</p>
                        <div class="mt-3">
                            <h5>Points Forts:</h5>
                            <ul>
                                <li>Profit factor: {metrics['profit_factor']:.2f}</li>
                                <li>Ratio gain/perte: {metrics['win_loss_ratio']:.2f}</li>
                                <li>Nombre de triangles détectés: {len(triangles)}</li>
                            </ul>
                            <h5>Points Faibles:</h5>
                            <ul>
                                <li>Drawdown maximum: {metrics['max_drawdown']:.2f}%</li>
                                <li>Volatilité: {metrics['volatility']:.2f}%</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Paramètres du backtest -->
        <section id="parametres" class="mt-5">
            <h2 class="section-title">Paramètres du Backtest</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Paramètres de Trading</h5>
                            <ul>
                                <li>Capital initial: {initial_balance} USDT</li>
                                <li>Taille des positions: 30 USDT</li>
                                <li>Effet de levier: 25x</li>
                                <li>Stop loss: 0.5%</li>
                                <li>Trailing stop: 2%</li>
                                <li>Frais par trade: 0.5 USDT</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Paramètres d'Analyse</h5>
                            <ul>
                                <li>EMAs utilisées: 5 à 200 (par paliers de 5)</li>
                                <li>Seuil triangles rouges: >0.2 ou >1.5</li>
                                <li>Seuil triangles verts: <-0.2 ou <-1.5</li>
                                <li>Fenêtre de détection: 3-4 barres</li>
                                <li>Multiplicateur de pente: 50</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- KPIs -->
        <section id="kpis" class="mt-5">
            <h2 class="section-title">Indicateurs de Performance Clés</h2>
            <div class="row">
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Profit Net Total</div>
                        <div class="metric-value {'positive' if metrics['total_net_profit'] > 0 else 'negative'}">
                            {metrics['total_net_profit']:.2f} USDT
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Taux de Réussite</div>
                        <div class="metric-value">{metrics['win_rate']:.1f}%</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{metrics['profit_factor']:.2f}</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Drawdown Max</div>
                        <div class="metric-value negative">{metrics['max_drawdown']:.2f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{metrics['sortino_ratio']:.2f}</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Calmar Ratio</div>
                        <div class="metric-value">{metrics['calmar_ratio']:.2f}</div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6 mb-3">
                    <div class="metric-card">
                        <div class="metric-label">Volatilité Annuelle</div>
                        <div class="metric-value">{metrics['volatility']:.1f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body">
                            <h5>Détails des Performances</h5>
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <tbody>
                                        <tr>
                                            <td>Nombre total de trades</td>
                                            <td><strong>{metrics['total_trades']}</strong></td>
                                            <td>Gain moyen</td>
                                            <td class="positive"><strong>{metrics['average_win']:.2f} USDT</strong></td>
                                        </tr>
                                        <tr>
                                            <td>Trades gagnants</td>
                                            <td class="positive"><strong>{metrics['winning_trades']}</strong></td>
                                            <td>Perte moyenne</td>
                                            <td class="negative"><strong>{metrics['average_loss']:.2f} USDT</strong></td>
                                        </tr>
                                        <tr>
                                            <td>Trades perdants</td>
                                            <td class="negative"><strong>{metrics['losing_trades']}</strong></td>
                                            <td>Plus gros gain</td>
                                            <td class="positive"><strong>{metrics['largest_win']:.2f} USDT</strong></td>
                                        </tr>
                                        <tr>
                                            <td>Durée moyenne des trades</td>
                                            <td><strong>{metrics['average_trade_duration']:.1f} heures</strong></td>
                                            <td>Plus grosse perte</td>
                                            <td class="negative"><strong>{metrics['largest_loss']:.2f} USDT</strong></td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Graphiques -->
        <section id="graphiques" class="mt-5">
            <h2 class="section-title">Graphiques Interactifs</h2>
            
            <!-- Courbe d'équité -->
            <div class="chart-container">
                <h4>Courbe d'Équité</h4>
                <div id="equityCurve"></div>
            </div>
            
            <!-- Drawdown -->
            <div class="chart-container">
                <h4>Drawdown dans le Temps</h4>
                <div id="drawdownChart"></div>
            </div>
            
            <!-- Distribution des profits/pertes -->
            <div class="row">
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Distribution des Profits/Pertes</h4>
                        <div id="profitDistribution"></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Répartition des Types de Trades</h4>
                        <div id="tradeTypeDistribution"></div>
                    </div>
                </div>
            </div>
            
            <!-- Heatmap mensuelle -->
            <div class="chart-container">
                <h4>Performance Mensuelle</h4>
                <div id="monthlyHeatmap"></div>
            </div>
        </section>
        
        <!-- Analyse détaillée des trades -->
        <section id="trades" class="mt-5">
            <h2 class="section-title">Analyse Détaillée des Trades</h2>
            <div class="card">
                <div class="card-body">
                    <div class="table-responsive">
                        <table id="tradesTable" class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Type</th>
                                    <th>Direction</th>
                                    <th>Date Entrée</th>
                                    <th>Date Sortie</th>
                                    <th>Prix Entrée</th>
                                    <th>Prix Sortie</th>
                                    <th>Profit (USDT)</th>
                                    <th>Profit (%)</th>
                                    <th>Durée (h)</th>
                                </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Statistiques par type de trade -->
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5>Performance par Direction</h5>
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Direction</th>
                                        <th>Nombre</th>
                                        <th>Gagnants</th>
                                        <th>Perdants</th>
                                        <th>Profit Total</th>
                                        <th>Win Rate</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Long</td>
                                        <td>{trade_stats['long']['count']}</td>
                                        <td class="positive">{trade_stats['long']['wins']}</td>
                                        <td class="negative">{trade_stats['long']['losses']}</td>
                                        <td class="{'positive' if trade_stats['long']['total_profit'] > 0 else 'negative'}">{trade_stats['long']['total_profit']:.2f}</td>
                                        <td>{(trade_stats['long']['wins'] / trade_stats['long']['count'] * 100) if trade_stats['long']['count'] > 0 else 0:.1f}%</td>
                                    </tr>
                                    <tr>
                                        <td>Short</td>
                                        <td>{trade_stats['short']['count']}</td>
                                        <td class="positive">{trade_stats['short']['wins']}</td>
                                        <td class="negative">{trade_stats['short']['losses']}</td>
                                        <td class="{'positive' if trade_stats['short']['total_profit'] > 0 else 'negative'}">{trade_stats['short']['total_profit']:.2f}</td>
                                        <td>{(trade_stats['short']['wins'] / trade_stats['short']['count'] * 100) if trade_stats['short']['count'] > 0 else 0:.1f}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5>Répartition des Sorties</h5>
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Type de Sortie</th>
                                        <th>Nombre</th>
                                        <th>Pourcentage</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Signal Opposé (Long)</td>
                                        <td>{exit_stats['close_long']}</td>
                                        <td>{(exit_stats['close_long'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0:.1f}%</td>
                                    </tr>
                                    <tr>
                                        <td>Signal Opposé (Short)</td>
                                        <td>{exit_stats['close_short']}</td>
                                        <td>{(exit_stats['close_short'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0:.1f}%</td>
                                    </tr>
                                    <tr>
                                        <td>Stop Loss</td>
                                        <td>{exit_stats['stop_loss']}</td>
                                        <td>{(exit_stats['stop_loss'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0:.1f}%</td>
                                    </tr>
                                    <tr>
                                        <td>Trailing Stop</td>
                                        <td>{exit_stats['trailing_stop']}</td>
                                        <td>{(exit_stats['trailing_stop'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0:.1f}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Analyse et Conclusions -->
        <section id="analyse" class="mt-5">
            <h2 class="section-title">Analyse et Conclusions</h2>
            
            <!-- Alertes et anomalies -->
            <div class="card mb-4">
                <div class="card-body">
                    <h4>Alertes et Anomalies</h4>
                    <div class="alert alert-info">
                        <h5>Trades Outliers Détectés</h5>
                        <ul>
                            <li>Plus gros gain: {metrics['largest_win']:.2f} USDT ({(metrics['largest_win'] / metrics['average_win']):.1f}x le gain moyen)</li>
                            <li>Plus grosse perte: {metrics['largest_loss']:.2f} USDT ({(metrics['largest_loss'] / metrics['average_loss']):.1f}x la perte moyenne)</li>
                        </ul>
                    </div>
                    {f'''<div class="alert alert-warning">
                        <h5>Points d'Attention</h5>
                        <ul>
                            <li>Drawdown maximum atteint: {metrics['max_drawdown']:.2f}%</li>
                            <li>Ratio de Sharpe {'faible' if metrics['sharpe_ratio'] < 1 else 'acceptable' if metrics['sharpe_ratio'] < 2 else 'bon'}: {metrics['sharpe_ratio']:.2f}</li>
                        </ul>
                    </div>''' if metrics['max_drawdown'] < -20 or metrics['sharpe_ratio'] < 1 else ''}
                </div>
            </div>
            
            <!-- Benchmark -->
            <div class="card mb-4">
                <div class="card-body">
                    <h4>Benchmark</h4>
                    <p>Comparaison avec une stratégie Buy & Hold:</p>
                    <ul>
                        <li>Prix initial: {df['close'].iloc[0]:.9f} USDT</li>
                        <li>Prix final: {df['close'].iloc[-1]:.9f} USDT</li>
                        <li>Performance Buy & Hold: {((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100):.2f}%</li>
                        <li>Performance de la stratégie: {(metrics['total_net_profit'] / initial_balance * 100):.2f}%</li>
                        <li>Alpha généré: {((metrics['total_net_profit'] / initial_balance * 100) - ((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100)):.2f}%</li>
                    </ul>
                </div>
            </div>
            
            <!-- Conclusions et recommandations -->
            <div class="card">
                <div class="card-body">
                    <h4>Conclusions et Axes d'Amélioration</h4>
                    <h5>Synthèse des Résultats</h5>
                    <p>La stratégie de détection de triangles a généré un {'profit' if metrics['total_net_profit'] > 0 else 'perte'} de 
                    {abs(metrics['total_net_profit']):.2f} USDT sur la période testée, avec un taux de réussite de {metrics['win_rate']:.1f}%.</p>
                    
                    <h5>Points Forts</h5>
                    <ul>
                        {'<li>Profit factor supérieur à 1.5, indiquant une bonne rentabilité</li>' if metrics['profit_factor'] > 1.5 else ''}
                        {'<li>Ratio gain/perte favorable</li>' if metrics['win_loss_ratio'] > 1.5 else ''}
                        {'<li>Taux de réussite élevé</li>' if metrics['win_rate'] > 55 else ''}
                        <li>Système de trailing stop efficace pour protéger les gains</li>
                    </ul>
                    
                    <h5>Limites Identifiées</h5>
                    <ul>
                        {'<li>Drawdown important nécessitant une meilleure gestion du risque</li>' if metrics['max_drawdown'] < -15 else ''}
                        {'<li>Sharpe ratio faible indiquant un rendement ajusté au risque insuffisant</li>' if metrics['sharpe_ratio'] < 1 else ''}
                        {'<li>Nombre de trades relativement faible pour une validation statistique robuste</li>' if metrics['total_trades'] < 50 else ''}
                        <li>Dépendance aux conditions de marché spécifiques</li>
                    </ul>
                    
                    <h5>Recommandations d'Amélioration</h5>
                    <ol>
                        <li><strong>Optimisation des paramètres:</strong> Tester différentes valeurs de seuils pour la détection des triangles</li>
                        <li><strong>Gestion du risque:</strong> Implémenter une taille de position dynamique basée sur la volatilité</li>
                        <li><strong>Filtres additionnels:</strong> Ajouter des conditions de volume ou de momentum</li>
                        <li><strong>Diversification:</strong> Tester sur d'autres paires et timeframes</li>
                        <li><strong>Machine Learning:</strong> Explorer l'utilisation d'algorithmes d'apprentissage pour améliorer la détection</li>
                    </ol>
                </div>
            </div>
        </section>
    </div>
    
    <!-- Footer -->
    <footer class="mt-5 py-4 bg-dark text-white text-center">
        <p>Rapport généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | © 2024 Backtest Trading System</p>
    </footer>
    
    <!-- Scripts -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
    
    <script>
        // Données pour les graphiques
        const equityData = {json.dumps(equity_curve_data)};
        const drawdownData = {json.dumps(drawdown_data)};
        const profitDistribution = {json.dumps(profit_distribution)};
        const tradeDetails = {json.dumps(trade_details)};
        const heatmapData = {json.dumps(heatmap_data)};
        
        // Configuration des graphiques
        const config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        // Courbe d'équité
        const equityTrace = {{
            x: equityData.map(d => d.x),
            y: equityData.map(d => d.y),
            type: 'scatter',
            mode: 'lines',
            name: 'Balance',
            line: {{color: '#3498db', width: 2}}
        }};
        
        Plotly.newPlot('equityCurve', [equityTrace], {{
            title: 'Évolution du Capital',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Balance (USDT)'}},
            hovermode: 'x unified'
        }}, config);
        
        // Drawdown
        const drawdownTrace = {{
            x: drawdownData.map(d => d.x),
            y: drawdownData.map(d => d.y),
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            name: 'Drawdown',
            line: {{color: '#e74c3c'}},
            fillcolor: 'rgba(231, 76, 60, 0.3)'
        }};
        
        Plotly.newPlot('drawdownChart', [drawdownTrace], {{
            title: 'Drawdown',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Drawdown (%)'}},
            hovermode: 'x unified'
        }}, config);
        
        // Distribution des profits
        const profitHist = {{
            x: profitDistribution,
            type: 'histogram',
            name: 'Distribution',
            marker: {{
                color: profitDistribution.map(v => v > 0 ? '#27ae60' : '#e74c3c')
            }},
            nbinsx: 20
        }};
        
        Plotly.newPlot('profitDistribution', [profitHist], {{
            title: 'Distribution des Profits/Pertes',
            xaxis: {{title: 'Profit (USDT)'}},
            yaxis: {{title: 'Fréquence'}},
            bargap: 0.1
        }}, config);
        
        // Répartition des types de trades
        const tradeTypePie = {{
            values: [{trade_stats['long']['count']}, {trade_stats['short']['count']}],
            labels: ['Long', 'Short'],
            type: 'pie',
            marker: {{
                colors: ['#3498db', '#e67e22']
            }}
        }};
        
        Plotly.newPlot('tradeTypeDistribution', [tradeTypePie], {{
            title: 'Répartition Long/Short'
        }}, config);
        
        // Heatmap mensuelle (si données disponibles)
        if (heatmapData.length > 0) {{
            const months = heatmapData.map(d => d.date);
            const values = heatmapData.map(d => d.value);
            
            const heatmapTrace = {{
                z: [values],
                x: months,
                y: ['Performance'],
                type: 'heatmap',
                colorscale: [
                    [0, '#e74c3c'],
                    [0.5, '#ecf0f1'],
                    [1, '#27ae60']
                ],
                zmid: 0
            }};
            
            Plotly.newPlot('monthlyHeatmap', [heatmapTrace], {{
                title: 'Performance Mensuelle (%)',
                xaxis: {{title: 'Mois'}},
                yaxis: {{title: ''}},
                height: 200
            }}, config);
        }}
        
        // DataTable pour les trades
        $(document).ready(function() {{
            const table = $('#tradesTable').DataTable({{
                data: tradeDetails,
                columns: [
                    {{ data: 'id' }},
                    {{ data: 'type' }},
                    {{ data: 'direction' }},
                    {{ data: 'entry_time' }},
                    {{ data: 'exit_time' }},
                    {{ data: 'entry_price', render: $.fn.dataTable.render.number(',', '.', 9) }},
                    {{ data: 'exit_price', render: $.fn.dataTable.render.number(',', '.', 9) }},
                    {{ 
                        data: 'profit',
                        render: function(data, type, row) {{
                            const color = data > 0 ? 'positive' : 'negative';
                            return `<span class="${{color}}">${{parseFloat(data).toFixed(2)}}</span>`;
                        }}
                    }},
                    {{ 
                        data: 'profit_percent',
                        render: function(data, type, row) {{
                            const color = data > 0 ? 'positive' : 'negative';
                            return `<span class="${{color}}">${{parseFloat(data).toFixed(2)}}%</span>`;
                        }}
                    }},
                    {{ data: 'duration_hours', render: $.fn.dataTable.render.number(',', '.', 2) }}
                ],
                pageLength: 25,
                order: [[0, 'desc']],
                language: {{
                    url: '//cdn.datatables.net/plug-ins/1.11.5/i18n/fr-FR.json'
                }},
                createdRow: function(row, data, dataIndex) {{
                    if (data.profit > 0) {{
                        $(row).addClass('trade-row-positive');
                    }} else {{
                        $(row).addClass('trade-row-negative');
                    }}
                }}
            }});
        }});
        
        // Fonction d'export CSV
        function exportToCSV() {{
            let csv = 'ID,Type,Direction,Date Entrée,Date Sortie,Prix Entrée,Prix Sortie,Profit,Profit %,Durée\\n';
            tradeDetails.forEach(trade => {{
                csv += `${{trade.id}},${{trade.type}},${{trade.direction}},${{trade.entry_time}},${{trade.exit_time}},${{trade.entry_price}},${{trade.exit_price}},${{trade.profit}},${{trade.profit_percent}},${{trade.duration_hours}}\\n`;
            }});
            
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('hidden', '');
            a.setAttribute('href', url);
            a.setAttribute('download', 'trades_export.csv');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }}
        
        // Dark mode toggle
        function toggleDarkMode() {{
            document.body.classList.toggle('dark-mode');
            const icon = document.querySelector('.fa-moon');
            if (document.body.classList.contains('dark-mode')) {{
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            }} else {{
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }}
        }}
        
        // Smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    return html_content

# Fonction pour sauvegarder le rapport
def save_report(html_content, filename="backtest_report.html"):
    """
    Sauvegarde le rapport HTML dans un fichier
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Rapport sauvegardé dans {filename}")
    return filename