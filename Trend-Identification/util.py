import random
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

output_notebook()

def seed(n):
    random.seed(n)
    np.random.seed(n)

def read_candles():
    df = pd.read_csv('../Dataset/output/SP500.csv')
    df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
    global dfs
    dfs = {name: sdf.set_index('Date').sort_index() for name, sdf in df.groupby('Name')}

def read_candles_one(stock):
    return dfs[stock].drop(['Name'], axis=1)

def plot_candles(df, showit=True):
    p = figure(width=800, height=400, x_axis_type='datetime')
    inc = df.close > df.open
    dec = df.open > df.close
    w = 12 * 60 * 60 * 1000
    p.segment(df.index, df.high, df.index, df.low, color='black')
    p.vbar(df.index[inc], w, df.open[inc], df.close[inc], fill_color='green', line_color='green')
    p.vbar(df.index[dec], w, df.open[dec], df.close[dec], fill_color='red', line_color='red')
    if showit:
        show(p)
    else:
        return p

def plot_trend(df, showit=True):
    p = figure(width=800, height=400, x_axis_type='datetime')
    p.line(df.index, df.adjclose, color='black')
    p.line(df.index, df.trend, color='blue')
    if showit:
        show(p)
    else:
        return p
