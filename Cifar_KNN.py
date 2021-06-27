import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from KNN import kNN_classify
from utility import getXmean, centralized

batch_size = 100
# Cifar10 dataset
train_dataset = datasets.CIFAR10(root='/ml/pycifar',  # 选择数据的根目录
                                 train=True,  # 选择训练集
                                 download=True)  # 从网络上download图片
test_dataset = datasets.CIFAR10(root='/ml/pycifar',  # 选择数据的根目录
                                train=False,  # 选择测试集
                                download=True)  # 从网络上download图片
# 加载数据

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


X_train = train_loader.dataset.data
mean_image = getXmean(X_train)
print(type(mean_image.reshape((32, 32, 3))))
plt.imshow(mean_image.reshape((32, 32, 3)).astype(int))
plt.show()
X_train = centralized(X_train, mean_image)
y_train = train_loader.dataset.targets
X_test = test_loader.dataset.data[:100]
X_test = centralized(X_test, mean_image)
y_test = test_loader.dataset.targets[:100]
num_test = len(y_test)
y_test_pred = kNN_classify(6, 'M', X_train, y_train, X_test)  # 这里并没有使用封装好的类
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
