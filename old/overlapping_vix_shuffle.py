import pandas as pd
import numpy as np
import math
import utils
import plotly.express as px
import plotly.graph_objects as go
from numpy.lib.stride_tricks import sliding_window_view

vol_label = "VIX"
df = pd.read_excel("data/VIXSPX.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
slice_index = int(0.8 * len(df))

rendements = df["Return"].to_numpy()
rendements2 = df["Return 2"].to_numpy()

nbr_lags = 1000
nbr_rows = len(rendements) - nbr_lags + 1


X_df = sliding_window_view(rendements, window_shape=nbr_lags)
X_df2 = sliding_window_view(rendements2, window_shape=nbr_lags)
Y_df = df[vol_label][nbr_lags + 1 : nbr_lags + nbr_rows + 2].to_numpy()
Dates_df = df["Date"][nbr_lags + 1 : nbr_lags + nbr_rows + 2]

matching_rows = min(X_df.shape[0], len(Y_df))
X_df, X_df2, Y_df, Dates_df = X_df[:matching_rows], X_df2[:matching_rows], Y_df[:matching_rows], Dates_df[:matching_rows]

print(f"X : {X_df.shape[0]}, X2 : {X_df2.shape[0]}, Y : {len(Y_df)}, Dates : {len(Dates_df)}")


"""
nbr_shuffles = int(0.3 * len(df))
for _ in range(nbr_shuffles):
    indices = np.random.permutation(matching_rows)
    X_df = X_df[indices]
    X_df2 = X_df2[indices]
    Y_df = Y_df[indices]
    Dates_df = Dates_df.iloc[indices]
"""


####### Training #############


X = X_df[:slice_index]
X2 = X_df2[:slice_index]
Y = Y_df[:slice_index]
Dates = Dates_df[:slice_index]


linear_estimate = utils.train_linear_model(X,Y)

Y_pred = linear_estimate.predict(X)
Y_residual = [(y - y_pred) for y, y_pred in zip(Y, Y_pred)]
mse = sum(y**2 for y in Y_residual) / len(Y_residual)

linear_estimate2 = utils.train_linear_model(X2, Y_residual)
Y_pred2 = linear_estimate2.predict(X2)
Y_pred_concat = [(y1 + y2) for y1, y2 in zip(Y_pred, Y_pred2)]

mse2 = sum((y - y_pred)**2 for y, y_pred in zip(Y, Y_pred_concat)) / len(Y)

print(f"TRAINING -> MSE : {mse}, MSE2 : {mse2}")



fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=Dates, y=Y, mode='lines', name=vol_label))
fig1.add_trace(go.Scatter(x=Dates, y=Y_pred, mode='lines', name=f'{vol_label} predicted'))
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=Dates, y=Y_residual, mode='lines', name=f'Residual {vol_label}'))
fig2.add_trace(go.Scatter(x=Dates, y=Y_pred2, mode='lines', name=f'Residual predicted {vol_label}'))
fig2.show()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=Dates, y=Y, mode='lines', name=vol_label))
fig3.add_trace(go.Scatter(x=Dates, y=Y_pred_concat, mode='lines', name=f'{vol_label} predicted after 2 regressions'))
fig3.show()



######### Testing #################


X_test = X_df[slice_index:]
X_test2 = X_df2[slice_index:]
Y_test = Y_df[slice_index:]
Dates_test = Dates_df[slice_index:]


Y_pred_test = linear_estimate.predict(X_test)
Y_residual_test = [(y - y_pred) for y, y_pred in zip(Y_test, Y_pred_test)]
mse_test = sum(y**2 for y in Y_residual_test) / len(Y_residual_test)


Y_pred_test2 = linear_estimate2.predict(X_test2)
Y_pred_concat_test = [(y1 + y2) for y1, y2 in zip(Y_pred_test, Y_pred_test2)]

mse_test2 = sum((y - y_pred)**2 for y, y_pred in zip(Y_test, Y_pred_concat_test)) / len(Y_test)

print(f" TEST -> MSE : {mse_test}, MSE2 : {mse_test2}")



fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=Dates_test, y=Y_test, mode='lines', name= vol_label))
fig4.add_trace(go.Scatter(x=Dates_test, y=Y_pred_test, mode='lines', name=f"{vol_label} predicted"))
fig4.show()

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=Dates_test, y=Y_residual_test, mode='lines', name=f'Residual {vol_label}'))
fig5.add_trace(go.Scatter(x=Dates_test, y=Y_pred_test2, mode='lines', name=f'Residual {vol_label} predicted'))
fig5.show()

fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=Dates_test, y=Y_test, mode='lines', name=vol_label))
fig6.add_trace(go.Scatter(x=Dates_test, y=Y_pred_concat_test, mode='lines', name=f'{vol_label} predicted after 2 regressions'))
fig6.show()

