import kNN
from numpy import *

def load_data(filepath):
    file = open(filepath)
    data_set = []
    data_set_result = []
    for each_line in file:
        each_line = each_line.rstrip()
        elems = each_line.split(",")
        data_set.append(elems[:-1])
        data_set_result.append(int(elems[-1]))
    file.close()
    return data_set, data_set_result

def create_train_data_set():
    return load_data("optdigits.tra")

def create_test_data_set():
    return load_data("optdigits.tes")

(train_data_set, train_data_set_result) = create_train_data_set()
(test_data_set, test_result) = create_test_data_set()

train_count = len(train_data_set)
test_count = len(test_data_set)
data_set = zeros((train_count+test_count, 64))
data_set_result = []
for i in range(train_count):
    data_set[i] = train_data_set[i]
for i in range(test_count):
    data_set[i+train_count] = test_data_set[i]

data_set_result.extend(train_data_set_result)
data_set_result.extend(test_result)

success_count = 0
for i in range(test_count):
    # k近邻算法（输入向量，训练样本集，标签向量，最近邻居的数目）
    test = kNN.classify0(mat(test_data_set[i]).astype("float64"), mat(train_data_set).astype("float64"), train_data_set_result, 2)
    if test != test_result[i]:
        print("第%d次测试：结果错误，测试结果为%d，正确答案是%d\n\t测试数：%d, 正确数：%d, 失败数：%d，正确率：%f"
              % (i + 1, test, test_result[i], i + 1, success_count, (i+1)-success_count, success_count/(i+1)))
    else:
        success_count += 1
        print("第%d次测试：结果正确，测试结果为%d\n\t测试数：%d, 正确数：%d, 失败数：%d，正确率：%f"
              % (i + 1, test, i + 1, success_count, (i + 1) - success_count, success_count / (i + 1)))