import numpy as np


def getXmean(X_train):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))  # 将图片从二维展开为一维
    mean_image = np.mean(X_train, axis=0)  # 求出训练集所有图片每个像素位置上的平均值
    return mean_image


def centralized(X_test, mean_image):
    X_test = np.reshape(X_test, (X_test.shape[0], -1))  # 将图片从二维展开为一维
    X_test = X_test.astype(np.float)
    X_test -= mean_image  # 减去均值图像，实现零均值化
    return X_test


def createDataSet():
    group = np.array([[0.5, 1.5], [1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5], [1.1, 1.0]])
    labels = np.array(['B', 'A', 'A', 'B', 'B', 'A'])
    return group, labels