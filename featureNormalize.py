import numpy as np
from numpy.matlib import repmat

def featureNormalize(X):
    """
        Функция позволяет вычислить нормализованную версию матрицы
        объекты-признаки X со средним значением для каждого признака
        равным 0 и среднеквадратическим отклонением равным 1
    """

    X_norm = X
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    mu = np.mean(X_norm, 0)
    X_norm = X_norm - mu
    sigma = np.std(X_norm, 0)
    X_norm = X_norm/sigma
    return X_norm, mu, sigma
