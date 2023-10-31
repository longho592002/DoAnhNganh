import MetaTrader5 as mt5
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import pandas_ta as ta
from mt5_funcs import get_symbol_name, TIMEFRAMES, TIMEFRAME_DICT
import ta

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

symbol_dropdown = html.Div([
    html.P('Symbol:'),
    dcc.Dropdown(
        id='symbol-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in get_symbol_name()],
        value='XAUUSDm'
    )
])

timeframe_dropdown = html.Div([
    html.P('Timeframe:'),
    dcc.Dropdown(
        id='timeframe-dropdown',
        options=[{'label': timeframe, 'value': timeframe} for timeframe in TIMEFRAMES],
        value='M5'
    )
])

num_bars_input = html.Div([
    html.P('Number of Candles'),
    dbc.Input(id='num-bar-input', type='number', value='500')
])

app.layout = html.Div([
    html.H1('Real Time Charts'),

    dbc.Row([
        dbc.Col(symbol_dropdown),
        dbc.Col(timeframe_dropdown),
        dbc.Col(num_bars_input)
    ]),
    html.Hr(),
    dcc.Interval(id='update', interval=200),
    html.Div(id='page-content'),
    html.Div(id='sub-content')
], style={'margin-left': '5%', 'margin-right': '5%', 'margin-top': '20px'})


@app.callback(
    Output('page-content', 'children'),
    Output('sub-content', 'children'),
    Input('update', 'n_intervals'),
    State('symbol-dropdown', 'value'), State('timeframe-dropdown', 'value'), State('num-bar-input', 'value')
)
def update_ohlc_chart(interval, symbol, timeframe, num_bars):
    timeframe_str = timeframe
    timeframe = TIMEFRAME_DICT[timeframe]
    num_bars = int(num_bars)
    mt5.initialize()
    login = 122573284
    password = '123456789Az@'
    server = 'Exness-MT5Trial7'
    mt5.login(login, password, server)

    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # df['RSI'] = df.ta.rsi(length=14)
    df['RSI1'] = ta.momentum.RSIIndicator(close=((df['high'] + df['low']) / 2).abs(), window=5).rsi()
    df['RSI2'] = ta.momentum.RSIIndicator(close=((df['high'] + df['low']) / 2).abs(), window=13).rsi()
    df['RSI3'] = ta.momentum.RSIIndicator(close=((df['high'] + df['low']) / 2).abs(), window=34).rsi()
    df['RSI'] = (df['RSI1'] + df['RSI2'] + df['RSI3']) / 3
    df['pivot'] = df.apply(lambda x: pivotid(df, x.name, 5, 5), axis=1)
    df['RSIpivot'] = df.apply(lambda x: RSIpivotID(df, x.name, 5, 5), axis=1)
    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    df['RSIpointpos'] = df.apply(lambda row: RSIpointpos(row), axis=1)
    df['divSignal2'] = df.apply(lambda row: divsignal2(df, row, 30), axis=1)
    df_signal = df[(~df['divSignal2'].isna()) & (df['divSignal2'] != 0) & (~df['pointpos'].isna()) & (~df['RSIpointpos'].isna())]
    # df_signal = df[(~df['divSignal2'].isna()) & (df['divSignal2'] != 0)]
    print(df_signal)
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Candlestick(x=np.array(df['time']),
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close']))
    fig.add_scatter(
        x=np.array(df['time']), y=df['pointpos'], mode="markers", marker=dict(size=10, color="blue"),
        name="pivot",
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=np.array(df['time']),
        y=df['RSI']
    ), row=2, col=1)
    fig.add_scatter(
        x=np.array(df['time']), y=df['RSIpointpos'], mode="markers", marker=dict(size=10, color="blue"),
        name="pivot",
        row=2, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(yaxis={'side': 'right'})
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    return [
        html.H2(id='chart-details', children=f'{symbol} - {timeframe_str}'),
        dcc.Graph(figure=fig, config={'displayModeBar': False})
    ]


def myRSI(price, n=20):
    delta = price['close'].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(window=n).mean()
    RolDown = dDown.rolling(window=n).mean().abs()

    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    return rsi


def pivotid(df1, l, n1, n2):
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0

    pividlow = 1
    pividhigh = 1
    for i in range(l - n1, l + n2 + 1):
        if (df1.low[l] > df1.low[i]):
            pividlow = 0
        if (df1.high[l] < df1.high[i]):
            pividhigh = 0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0


def RSIpivotID(df1, l, n1, n2):
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0
    pividlow = 1
    pividhigh = 1
    for i in range(l - n1, l + n2 + 1):
        if (df1.RSI[l] > df1.RSI[i]):
            pividlow = 0
        if (df1.RSI[l] < df1.RSI[i]):
            pividhigh = 0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0


def pointpos(x):
    if x['pivot'] == 1:
        return x['low'] - 1e-3
    elif x['pivot'] == 2:
        return x['high'] + 1e-3
    else:
        return np.nan


def RSIpointpos(x):
    if x['RSIpivot'] == 1:
        return x['RSI'] - 1
    elif x['RSIpivot'] == 2:
        return x['RSI'] + 1
    else:
        return np.nan


def divsignal2(df, x, nbackcandles):
    backcandles = nbackcandles
    candleid = int(x.name)

    closp = np.array([])
    xxclos = np.array([])

    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])

    maximRSI = np.array([])
    minimRSI = np.array([])
    xxminRSI = np.array([])
    xxmaxRSI = np.array([])

    for i in range(candleid - backcandles, candleid + 1):
        closp = np.append(closp, df.iloc[i].close)
        xxclos = np.append(xxclos, i)
        if df.iloc[i].pivot == 1:
            minim = np.append(minim, df.iloc[i].low)
            xxmin = np.append(xxmin, df.iloc[i].name)
        if df.iloc[i].pivot == 2:
            maxim = np.append(maxim, df.iloc[i].high)
            xxmax = np.append(xxmax, df.iloc[i].name)
        if df.iloc[i].RSIpivot == 1:
            minimRSI = np.append(minimRSI, df.iloc[i].RSI)
            xxminRSI = np.append(xxminRSI, df.iloc[i].name)
        if df.iloc[i].RSIpivot == 2:
            maximRSI = np.append(maximRSI, df.iloc[i].RSI)
            xxmaxRSI = np.append(xxmaxRSI, df.iloc[i].name)

    slclos, interclos = np.polyfit(xxclos, closp, 1)

    if slclos > 1e-4 and (maximRSI.size < 2 or maxim.size < 2):
        return 0
    if slclos < -1e-4 and (minimRSI.size < 2 or minim.size < 2):
        return 0
    if slclos > 1e-4:
        if maximRSI[-1] < maximRSI[-2] and maxim[-1] > maxim[-2]:
            if((df.iloc[i].RSIpointpos <= 50)):
                return 1
            else:
                return 2
    elif slclos < -1e-4:
        if minimRSI[-1] > minimRSI[-2] and minim[-1] < minim[-2]:
            if ((df.iloc[i].RSIpointpos > 50)):
                return 2
            else:
                return 1
    else:
        return 0


if __name__ == '__main__':
    app.run_server(port=8051)
 # su dung thuat toan rsi cai tien de du bao gia vang trong thoi gian thuc