import numpy as np
import pandas as pd

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

from matplotlib import pyplot as plt
import plotly.express as px

assets = ["BBAS3.SA", "PETR4.SA", "EVEN3.SA", "CPLE6.SA", "IVVB11.SA"]

start = "2022-03-01"
prices = pdr.get_data_yahoo(assets, start=start)['Adj Close']

returns = prices.pct_change().dropna()

weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

returns_portifolio = returns.assign(portifolio=returns.dot(weights))

acum_index = (1+returns_portifolio).cumprod()

acum = acum_index.reset_index(level=0)

N = len(acum_index)
annualized_return = acum_index.iloc[-1,] ** (252 / N) - 1
sd_annualized = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252
port_sd_annualized = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
annualized_risk = pd.concat([pd.Series({'portfolio' : port_sd_annualized}), sd_annualized])
annualized_risk_return = pd.DataFrame({'Risco Anualizado' : annualized_risk,
              'Retorno Anualizado' : annualized_return})
px.scatter(annualized_risk_return,
           x = 'Risco Anualizado',
           y = 'Retorno Anualizado',
           color =  annualized_risk_return.index,
           size = np.array([5, 5, 5, 5, 5, 5, 5])
           )