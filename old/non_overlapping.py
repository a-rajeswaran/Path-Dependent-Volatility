import pandas as pd
import numpy as np
import math
import utils
import plotly.express as px


vspx = pd.read_excel("data/VIXSPX.xlsx")
rendements_test = vspx["Return"].to_numpy()
nbr_lags = 94
nbr_rows_test = len(rendements_test) // nbr_lags
vix = rendements_test[: nbr_rows_test * nbr_lags]
X = vix.reshape(nbr_rows_test, nbr_lags)
Y_test = vspx["VIX"][nbr_lags +1 :nbr_lags + 1 + nbr_rows_test].to_numpy()

print(len(X), len(Y_test))




"""
fig = px.line(vspx, x="Date", y="VIX", title="VIX")
fig.show()
"""



