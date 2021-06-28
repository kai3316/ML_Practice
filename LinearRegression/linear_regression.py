import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.array([1, 2, 4, 6, 8])
    y = np.array([2, 5, 7, 8, 9])
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denominator = 0.0
    numerator = 0.0
    for x_i, y_i in zip(x, y):
        numerator += (x_i - x_mean) * (y_i - y_mean)  # 按照a的公式得到分子
        denominator += (x_i - x_mean) ** 2  # 按照a的公式得到分母
    a = numerator / denominator  # 得到a
    b = y_mean - a * x_mean  # 得到b
    y_predict = a * x + b
    plt.scatter(x, y, color='b')
    plt.plot(x, y_predict, color='r')
    plt.xlabel('管子的长度', fontproperties='simHei', fontsize=15)
    plt.ylabel('收费', fontproperties='simHei', fontsize=15)
    plt.show()
