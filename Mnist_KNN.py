import numpy as np
from matplotlib import pyplot as plt

from KNN import kNN_classify

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from utility import getXmean, centralized

batch_size = 100
# MNIST dataset
train_dataset = dsets.MNIST(root='/ml/pymnist',  # 选择数据的根目录
                            train=True,  # 选择训练集
                            transform=None,  # 不考虑使用任何数据预处理
                            download=True)  # 从网络上download图片
test_dataset = dsets.MNIST(root='/ml/pymnist',  # 选择数据的根目录
                           train=False,  # 选择测试集
                           transform=None,  # 不考虑使用任何数据预处理
                           download=True)  # 从网络上download图片
# 加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# import matplotlib.pyplot as plt
# digit = train_loader.dataset.data[0]
# plt.imshow(digit, cmap=plt.cm.gray)
# print(train_loader.dataset.targets[0].numpy())
# plt.show()

X_train = train_loader.dataset.data.numpy()  # 需要转为numpy矩阵
mean_image = getXmean(X_train)
plt.imshow(mean_image.reshape((28, 28)), cmap=plt.cm.gray)
plt.show()
X_train = centralized(X_train, mean_image)
X_train = X_train.reshape(X_train.shape[0], 28 * 28)  # 需要reshape之后才能放入knn分类器

y_train = train_loader.dataset.targets.numpy()

X_test = test_loader.dataset.data[:1000].numpy()
X_test = centralized(X_test, mean_image)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

y_test = test_loader.dataset.targets[:1000].numpy()

num_test = y_test.shape[0]
y_test_pred = kNN_classify(5, 'M', X_train, y_train, X_test)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
