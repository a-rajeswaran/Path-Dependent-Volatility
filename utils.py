import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim


def calc_exp_prediction(input_vector, lambd):
    return sum(math.exp(lambd * x) for x in input_vector)


def loss(x_train, y_train, lambd, c):
    mse = 0
    for i in range(len(x_train) - 1001):
        input_vector = x_train[i : 1000 + i]
        y = y_train[1001 + i]
        y_pred = sum(math.exp(lambd * x) for x in input_vector) + c
        mse += (y - y_pred)**2
    return mse

def test_model(x_test, x_test_2, lambd1, lambd2, c):
    y_pred_vector = []
    for i in range(len(x_test) - 1001):
        input_vector = x_test[i : 1000 + i]
        y_pred = sum(math.exp(lambd1 * x) for x in input_vector) + np.sqrt(sum(math.exp(lambd2 * (x)) for x in x_test_2)) + c
        y_pred_vector.append(y_pred)
    return y_pred_vector


def train_linear_model(X, Y):
    if ((X.shape[0] != len(Y)) or (X.shape[0] < X.shape[1])):
        print("Error training linear model")
    model = LinearRegression()
    model.fit(X, Y)
    return model




def loss_function(integral_X, X, Y, beta, dt):
    Y_pred = X * beta[1:]
    integrals = torch.trapz(Y_pred, dx=dt, dim=1)
    residuals = integral_X * (Y - integrals - beta[0])
    loss = torch.abs(torch.sum(residuals))
    return loss

def train_loss_function(integral_x, X, Y, dt=0.01, lr=1e-2, steps=1000):
    M = X.shape[1]
    beta = torch.randn(M + 1, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    for step in range(steps):            
        optimizer.zero_grad()
        loss = loss_function(integral_x, X, Y, beta, dt)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step} | Loss = {loss.item():.6f}")
    return beta
    


#######################################################################################################


class LinearModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

def train_custom_loss(X, Y, loss_fn, lr=0.01, epochs=1000):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    model = LinearModel(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, Y)
        loss.backward()
        optimizer.step()

    return model


def smoothing_operator(B1, l):
    def phi_d(L,i):
        h=1/len(L)
        s1=0
        s2=0
        for j in range(i):
            s1=s1+np.exp(-j*h/np.sqrt(l))*L[j]*h
            s2=s2+np.exp(j*h/np.sqrt(l))*L[j]*h
        return (-np.exp(i*h/np.sqrt(l))/(2*np.sqrt(l))*s1+np.exp(-i*h/np.sqrt(l))/(2*np.sqrt(l))*s2)

    def compute_K(L):
        k1=0
        k2=0
        h=1/len(L)
        for j in range(len(L)):
            k1=k1+np.exp(2*j*h/np.sqrt(l))
            k2=k2+np.exp(-2*j*h/np.sqrt(l))
        return k1,k2

    k1,k2=compute_K(B1)
    alpha=k1*k2-(len(B1)**2)


    def compute_coeff(L):
        A=0
        B=0
        n=len(L)
        h=1/n
        for i in range(n):
            A=A+(k2*np.exp(i*h/np.sqrt(l))-n*np.exp(-i*h/np.sqrt(l)))*(L[i]-phi_d(L,i))
            B=B+(k1*np.exp(-i*h/np.sqrt(l))-n*np.exp(i*h/np.sqrt(l)))*(L[i]-phi_d(L,i))
        A=A/alpha
        B=B/alpha
        return A,B

    B2=[]
    A,B=compute_coeff(B1)
    for i in range(len(B1)):
        h=1/len(B1)
        c=phi_d(B1,i)+A*np.exp(i*h/np.sqrt(l))+B*np.exp(-i*h/np.sqrt(l))
        B2.append(c)