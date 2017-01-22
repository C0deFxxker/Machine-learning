from tree import *

def create_test_data():
    data_set = [
        [True,True],    # 不浮出水面可生存，有脚蹼（鱼类）
        [True,True],    # 不浮出水面可生存，有脚蹼（鱼类）
        [True,False],   # 不浮出水面可生存，无脚蹼（不是鱼类）
        [False,True],   # 不浮出水面无法生存，有脚蹼（不是鱼类）
        [False,True],   # 不浮出水面无法生存，有脚蹼（不是鱼类）
    ]
    result_set = ['yes','yes','no','no','no']
    labels = ['不浮出水面可生存', '有脚蹼']
    return data_set, result_set, labels

data_set, result_set, labels = create_test_data()
print("根据“不浮出水面是否可生存”以及“是否有脚蹼”两个特征判断是否鱼类的特征树：\n%s" % create_tree(data_set, result_set, labels))