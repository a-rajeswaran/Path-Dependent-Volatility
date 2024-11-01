# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:53:49 2024

@author: a-raj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_linear_estimator(X,Y):
    X_T=np.transpose(X)
    B=np.dot(X_T,X)
    B=np.linalg.inv(B)
    B=np.dot(B,X_T)
    B=np.dot(B,Y)
    return B

def compute_Omega(X_train, Y_train, E):
    X_T=np.transpose(X_train)
    B=np.dot(X_T,X_train)
    B=np.linalg.inv(B)
    eps=0
    for k in E:
        eps=eps+(k**2)
    eps=eps/(len(Y_train)-X_train.shape[1])
    B=B*eps
    return B

df=pd.read_csv("epex2020.csv",sep=";")
df.iloc[:, 1] = df.iloc[:, 1].str.replace(',', '.')
df.iloc[:, 1] = df.iloc[:, 1].astype(float)

cons=pd.read_csv("consommation2020.csv")
Y=[]
for i in range(len(cons)):
    Y.append(cons.iloc[i][0])
    
Cov=[]
X=[]
for i in range(365):
    ligne=[]
    compteur=0
    for j in range(48):
        ligne.append(Y[23+i*48+j])
    X.append(ligne)
    
    
X=np.array(X) 
Y_train, Y_test=df.iloc[1:200,1],df.iloc[200:,1]
Y_train=np.array(Y_train).astype(float)
Y_test=np.array(Y_test).astype(float)
X_train, X_test=X[:199], X[199:]
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))




B1=compute_linear_estimator(X_train,Y_train)
print(len(B1))
B0,B1=B1[:1],B1[1:]


l=0.0006999

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

plt.plot(B1, label="kernel linéaire")
plt.plot(B2, label="kernel lissé")
plt.legend()
plt.show()

Y_prev=[]
b0=np.mean(Y_train)
for i in range(len(Y_test)):
    y=0
    for j in range(48):
        y=y+B2[j]*X_test[i][j]
    y=y+b0
    Y_prev.append(abs(y))
    
plt.plot(Y_test, label="EPEX réalisé")
plt.plot(Y_prev, label="EPEX prédit")
plt.legend()
plt.show()


mse=0
for k in range(len(Y_prev)):
    mse=mse+(Y_prev[k]-Y_test[k])**2
mse=mse/len(Y_test)
    
def signe(nombre):
    if nombre > 0:
        return 1
    elif nombre < 0:
        return -1
    else:
        return 0
    
    
hit_ratio=0   
for k in range(1,len(Y_test)):
    if(signe(Y_prev[k]-Y_prev[k-1])==signe(Y_test[k]-Y_test[k-1])):
        hit_ratio+=1
hit_ratio=hit_ratio/len(Y_test)

benef=0
for j in range(len(Y_test)):
    benef+=-Y_test[j]+Y_prev[j]


    
        




    
    
