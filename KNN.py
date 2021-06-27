import operator
import numpy as np
import matplotlib.pyplot as plt

from utility import createDataSet


def kNN_classify(k, dis, train_data, train_label, Y_test):
    assert dis == 'E' or dis == 'M', 'dis must E or M，E代表欧拉距离，M代表曼哈顿距离'
    num_test = Y_test.shape[0]  # 测试样本的数量
    label_list = []
    '''
    使用欧拉公式作为距离度量
    '''

    for i in range(num_test):
        # 实现欧拉距离公式
        if dis == 'E':
            distances = np.sqrt(np.sum(((train_data - np.tile(Y_test[i], (train_data.shape[0], 1))) ** 2), axis=1))
        if dis == 'M':
            distances = np.sum(np.abs(train_data - np.tile(Y_test[i], (train_data.shape[0], 1))), axis=1)
        nearest_k = np.argsort(distances)  # 距离由小到大进行排序，并返回index值
        topK = nearest_k[:k]  # 选取前k个距离
        classCount = {}
        for j in topK:  # 统计每个类别的个数
            classCount[train_label[j]] = classCount.get(train_label[j], 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        label_list.append(sortedClassCount[0][0])
    return np.array(label_list)




if __name__ == '__main__':
    group, labels = createDataSet()
    y_prediction = kNN_classify(1, 'M', group, labels, np.array([[1.0, 2.1], [0.4, 2.0]]))
    print(y_prediction)
    plt.scatter(group[labels == 'A', 0], group[labels == 'A', 1], color='r', marker='*')  # 对于类别为A的数据集我们使用红色六角形表示
    plt.scatter(group[labels == 'B', 0], group[labels == 'B', 1], color='g', marker='+')  # 对于类别为B的数据集我们使用绿色十字形表示
    plt.show()
