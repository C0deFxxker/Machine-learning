from numpy import *

# 返回data_set中出现出现过的所有词条的集合(set)
def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)   # 并集
    return list(vocab_set)

# 将文档词条转为词集模型向量
# def set_of_words_2_vec(vocab_list, input_set):
#     return_vec = [0] * len(vocab_list)
#     for word in input_set:
#         if word in vocab_list:
#             return_vec[vocab_list.index(word)] = 1
#     return return_vec

# 将文档词条转为词袋模型向量
def bag_of_words_2_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec

# 朴素贝叶斯分类器训练函数（参数：文档矩阵[已转为词袋模型向量]，文档类别标签向量）
def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    pAbusive = sum(train_category) / float(num_train_docs)  # 计算侮辱性文档出现概率( p1, p0 = 1 - p1 )
    p0_num = ones(num_words);  p1_num = ones(num_words)   # 用于记录侮辱性和正常性文档中词条出现的次数+1（分子，+1是为了避免后面分类计算概率是乘0导致整个结果都为0的情况）
    # 算每个词条的出现总次数
    p0_denom = 2.0; p1_denom = 2.0                          # 记录两类文档的所有词条总数+2（分母，因为分子做了+1处理，分母也进行+2处理保证后面计算概率时，范围在0~1之间）
    for i in range(num_train_docs):
        if train_category[i] == 1:        # 辱骂性文档
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:                            # 正常文档
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vec = p1_num / p1_denom      # 辱骂性文档中每个词条出现的概率
    p0_vec = p0_num / p0_denom      # 非辱骂性文档中每个词条出现的概率
    return p0_vec, p1_vec, pAbusive

# 朴素贝叶斯分类器分类函数
# 参数：input_vec - 需要进行分类的文档向量（已转为词袋模型向量）
#       p0_vec - 正常文档中，每个词条出现的概率向量（在训练函数中得出的结果）
#       p1_vec - 辱骂性文档中，每个词条出现的概率向量（在训练函数中得出的结果）
#       p_abusive - 辱骂性文档出现的概率（在训练函数中得出的结果）
def classify(input_vec, p0_vec, p1_vec, p_abusive):
    # 使用对数加法实现乘法，避免太接近于0的浮点数相乘所造成的误差
    p1 = sum(input_vec * p1_vec) + log(p_abusive)
    p0 = sum(input_vec * p0_vec) + log(1.0 - p_abusive)
    # 比较两个文档类型的概率，取概率高者作为文档分类
    return 1 if p1 > p0 else 0