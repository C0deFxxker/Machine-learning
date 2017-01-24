from numpy import *
import operator

# k近邻算法（输入向量，训练样本集，标签向量，最近邻居的数目）
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = square(diffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqrt(sqDistances)
    arrDistances = zeros((distances.shape[0]))
    for i in range(distances.shape[0]):
        arrDistances[i] = distances[i][0]
    # 将距离计算结果排序
    sortedDistindicies = argsort(arrDistances)
    # 统计前k个结果中，各个分类的出现频率
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistindicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 将分类按出现频率降序排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    # 频率出现最高的分类，就认为是预测结果
    return sortedClassCount[0][0]

# 归一化数据（矩阵）
def autoNorm(dataSet:matrix):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals
