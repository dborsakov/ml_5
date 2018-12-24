import matplotlib.pyplot as plt

def plotData(data):
    """
        Функция позволяет выполнить визуализацию данных в декартовой
        системе координат с подписанным осями (численность населения
        и прибыль)
        """
    plt.figure()
    plt.plot(data[:,0], data[:,1], 'rx', markersize=5, label='Training Data')
    plt.legend(loc='upper right', shadow=True, fontsize=12, numpoints=1)
    plt.xlabel('Численность населения * 10К')
    plt.ylabel('Прибыль в * $10К')
    plt.grid()
