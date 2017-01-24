import bayes
from numpy import *

def load_data_set():
    posting_list = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    class_vec = [0,1,0,1,0,1]
    return posting_list, class_vec

posting_list, class_vec = load_data_set()
vocab_list = bayes.create_vocab_list(posting_list)

# 构建词条0-1向量
train_mat = []
for posting_doc in posting_list:
    train_mat.append( bayes.bag_of_words_2_vec(vocab_list, posting_doc) )

p0,p1,pAb = bayes.trainNB0(train_mat, class_vec)

test_entry = ['love', 'my', 'dalmation']
this_doc = array(bayes.bag_of_words_2_vec(vocab_list, test_entry))
print( test_entry, '是辱骂性文档' if bayes.classify(this_doc, p0, p1, pAb) else '不是辱骂性文档' )

test_entry = ['stupid', 'garbage']
this_doc = array(bayes.bag_of_words_2_vec(vocab_list, test_entry))
print( test_entry, '是辱骂性文档' if bayes.classify(this_doc, p0, p1, pAb) else '不是辱骂性文档' )
