import operator

import numpy as np
import torch
import torchvision.datasets as datasets
from utility import getXmean, centralized
import matplotlib.pyplot as plt


class Knn:

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.Xtr = X_train
        self.ytr = y_train

    def predict(self, k, dis, X_test):
        assert dis == 'E' or dis == 'M', 'dis must E or M'
        num_test = X_test.shape[0]  # 测试样本的数量
        labellist = []
        # 使用欧拉公式作为距离度量
        if (dis == 'E'):
            for i in range(num_test):
                distances = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i], (self.Xtr.shape[0], 1))) ** 2), axis=1))
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                classCount = {}
                for i in topK:
                    classCount[self.ytr[i]] = classCount.get(self.ytr[i], 0) + 1
                sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)

        # 使用曼哈顿公式作为距离度量
        if (dis == 'M'):
            for i in range(num_test):
                # 按照列的方向相加，其实就是行相加
                distances = np.sum(np.abs(self.Xtr - np.tile(X_test[i], (self.Xtr.shape[0], 1))), axis=1)
                nearest_k = np.argsort(distances)
                topK = nearest_k[:k]
                classCount = {}
                for i in topK:
                    classCount[self.ytr[i]] = classCount.get(self.ytr[i], 0) + 1
                sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)



batch_size = 100
# MNIST dataset
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
X_train = X_train.reshape(X_train.shape[0], -1)
mean_image = getXmean(X_train)
X_train = centralized(X_train, mean_image)
y_train = train_loader.dataset.targets
y_train = np.array(y_train)
X_test = test_loader.dataset.data
X_test = X_test.reshape(X_test.shape[0], -1)
X_test = centralized(X_test, mean_image)
y_test = test_loader.dataset.targets
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20]  # k的值一般选择1~20以内
num_training = X_train.shape[0]
X_train_folds = []
y_train_folds = []
indices = np.array_split(np.arange(num_training), indices_or_sections=num_folds)  # 把下标分成5个部分
for i in indices:
    X_train_folds.append(X_train[i])
    y_train_folds.append(y_train[i])
k_to_accuracies = {}
for k in k_choices:
    # 进行交叉验证
    acc = []
    for i in range(num_folds):
        x = X_train_folds[0:i] + X_train_folds[i + 1:]  # 训练集不包括验证集
        x = np.concatenate(x, axis=0)  # 使用concatenate将4个训练集拼在一起
        # print(x.shape)
        y = y_train_folds[0:i] + y_train_folds[i + 1:]

        y = np.concatenate(y)  # 对label使用同样的操作
        test_x = X_train_folds[i]  # 单独拿出验证集
        test_y = y_train_folds[i]

        classifier = Knn()  # 定义model
        classifier.fit(x, y)  # 将训练集读入
        # dist = classifier.compute_distances_no_loops(test_x)  # 计算距离矩阵
        y_pred = classifier.predict(k, 'M', test_x)  # 预测结果
        accuracy = np.mean(y_pred == test_y)  # 计算准确率
        acc.append(accuracy)
k_to_accuracies[k] = acc  # 计算交叉验证的平均准确率
# 输出准确度
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


# plot the raw observations


for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
