import numpy as np

def normalEqn(X, y):
    """
        Функция позволяет вычислить параметры модели для линейной
        регресии с использованием нормальных уравнений
    """

    theta = 0
    theta = np.linalg.pinv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return theta
