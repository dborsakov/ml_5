import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Функция позволяет выполнить градиентный спуск для поиска
        параметров модели theta, используя матрицу объекты-признаки X,
        вектор меток y, параметр сходимости alpha и число итераций
        алгоритма num_iters
    """

    J_history = []
    m = y.shape[0]
    xTrans = X.transpose()
    for i in range(num_iters):
        h = np.dot(X, theta)
        loss = h - y
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
