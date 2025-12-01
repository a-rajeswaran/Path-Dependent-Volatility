import pandas as pd
import numpy as np
import math
import utils
import plotly.express as px
import plotly.graph_objects as go
from numpy.lib.stride_tricks import sliding_window_view

vol_label = "VIX"
df = pd.read_excel("data/VIXSPX.xlsx")
slice_index = int(0.8 * len(df))
df_train, df_test = df.iloc[:slice_index], df.iloc[slice_index:]


####### Training #############
rendements = df_train["Return"].to_numpy()
rendements2 = df_train["Return 2"].to_numpy()

nbr_lags = 1000
nbr_rows = len(rendements) - nbr_lags + 1
X = sliding_window_view(rendements, window_shape=nbr_lags)
X2 = sliding_window_view(rendements2, window_shape=nbr_lags)
Y = df_train[vol_label][nbr_lags + 1 : nbr_lags + nbr_rows + 2].to_numpy()
Dates = df_train["Date"].to_numpy()
matching_rows = min(X.shape[0], len(Y))
X, X2, Y = X[:matching_rows], X2[:matching_rows], Y[:matching_rows]

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
fig1.update_layout(
    title=f"{vol_label} Prediction on Training Set",
    xaxis_title="Date",
    yaxis_title="Volatility",
    template="plotly_white"
)
fig1.write_image(f"{vol_label}_prediction_train_set.png", width=800, height=600)
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

rendements_test = df_test["Return"].to_numpy()
rendements_test2 = df_test["Return 2"].to_numpy()


nbr_rows_test = len(rendements_test) - nbr_lags + 1
X_test = sliding_window_view(rendements_test, window_shape=nbr_lags)
X_test2 = sliding_window_view(rendements_test2, window_shape=nbr_lags)
Y_test = df_test[vol_label][nbr_lags + 1 : nbr_lags + nbr_rows_test + 2].to_numpy()
Dates_test = df_test["Date"].to_numpy()
matching_rows_test = min(X_test.shape[0], len(Y_test))
X_test, X_test2, Y_test = X_test[:matching_rows_test], X_test2[:matching_rows_test], Y_test[:matching_rows_test]

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
fig4.update_layout(
    title=f"{vol_label} Prediction on Test Set",
    xaxis_title="Date",
    yaxis_title="Volatility",
    template="plotly_white"
)
fig4.write_image(f"{vol_label}_prediction_test_set.png", width=800, height=600)
fig4.show()

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=Dates_test, y=Y_residual_test, mode='lines', name=f'Residual {vol_label}'))
fig5.add_trace(go.Scatter(x=Dates_test, y=Y_pred_test2, mode='lines', name=f'Residual {vol_label} predicted'))
fig5.show()

fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=Dates_test, y=Y_test, mode='lines', name=vol_label))
fig6.add_trace(go.Scatter(x=Dates_test, y=Y_pred_concat_test, mode='lines', name=f'{vol_label} predicted after 2 regressions'))
fig6.show()


fig7 = go.Figure()
fig7.add_trace(go.Scatter(
    y=linear_estimate.coef_,
    mode='lines',
    name='Weights'
))

fig7.update_layout(
    title=f"Optimal kernel for {vol_label}",
    xaxis_title="Date",
    yaxis_title="Weight value",
    template="plotly_white"
)

fig7.write_image(f"{vol_label}_optimal_kernel.png", width=800, height=600)

fig7.show()


