from math import log

# 计算香农熵，用于评估一个结果集的复杂程度（结果越大，则resultSet中的元素越混乱）
def calc_shannon_ent(resultSet):
    num_entries = len(resultSet)
    labelCounts = {}
    for each_result in resultSet:
        if each_result not in labelCounts.keys():
            labelCounts[each_result] = 0
        labelCounts[each_result] += 1
    shannon_ent = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key])/num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

# 从数据集中筛选满足条件data_set[:,axis] == value，对应的结果值（结果集为result_set）
def fetch_result_with_feature(data_set, axis, value, result_set):
    ret_data_set = []
    for i in range(len(data_set)):
        if data_set[i][axis] == value:
            ret_data_set.append(result_set[i])
    return ret_data_set

# 筛选axis特征值为value的数据集出来，并删去axis特征
def split_data_set(data_set, axis, value, result_set):
    ret_data_set = []
    ret_result_set = []
    for i in range(len(data_set)):
        each_data = data_set[i]
        if each_data[axis] == value:
            l = each_data[:axis]
            l.extend(each_data[axis+1:])
            ret_data_set.append(l)
            ret_result_set.append(result_set[i])
    return ret_data_set, ret_result_set

def choose_best_feature_to_split(data_set, result_set):
    num_features = len(data_set[0])
    base_entropy = calc_shannon_ent(result_set)
    base_info_gain = 0.0
    base_feature = -1
    # 尝试每种特征划分方式，找出香农熵减少最多的特征划分方式为最优划分
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        # 计算每种划分方式的信息熵
        for value in unique_vals:
            sub_result_set = fetch_result_with_feature(data_set, i, value, result_set)
            prob = len(sub_result_set) / float(len(result_set))
            new_entropy += prob * calc_shannon_ent(sub_result_set)
        # 计算每种特征划分方式的香农熵收益
        info_gain = base_entropy - new_entropy
        # 选取信息增益最大的那种划分方式的特征作为最优划分特征
        if info_gain > base_info_gain:
            base_info_gain = info_gain
            base_feature = i
    return base_feature

# 当数据集的所有特征都划分完毕后，某些叶子节点中的依然存在多种类型
# 则我们定义这个叶子的类型为该叶子节点下的类型出现频率最高的那种类型
def majority_class(class_list):
    class_count = {}
    max_count = 0
    ret_class = class_list[0]
    for vote in class_list:
        # 计算每种类型的出现次数
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
        # 记录出现频率最高的类型
        if class_count[vote] > max_count:
            max_count = class_count[vote]
            ret_class = vote
    return ret_class

# 构造决策树（labels为数据集data_set中每个特征对应的名称）
def create_tree(data_set, result_set, labels):
    # 如果该节点下剩下都是同一类型，则不用再继续划分
    if result_set.count(result_set[0]) == len(result_set):
        return result_set[0]
    # 如果划分到该节点时，已没有更多特征值能继续划分，则抽取出现频率最高的类型代表该节点的类型
    if len(data_set[0]) == 0:
        return majority_class(result_set)
    # 选择最优划分特征
    best_feat = choose_best_feature_to_split(data_set, result_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label:{}}
    del labels[best_feat]
    example = [example[best_feat] for example in data_set]
    unique_vals = set(example)
    # 枚举最优划分特征的特征值，划分出多个树节点
    for value in unique_vals:
        sub_data_set, sub_result_set = split_data_set(data_set, best_feat, value, result_set)
        my_tree[best_feat_label][value] = create_tree(sub_data_set, sub_result_set, labels[:])
    return my_tree
