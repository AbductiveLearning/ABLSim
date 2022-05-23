import random
import numpy as np

def shuffle_str(s):
    str_list = list(s)
    random.shuffle(str_list)
    return ''.join(str_list)

def gen_ascii2num(sign2num):
    ascii2num = dict()
    for key, values in sign2num.items():
        ascii2num[ord(key)] = values
    return ascii2num

def strs2idxs(str_list, sign2num):
    ascii2num_func = np.vectorize(lambda x: gen_ascii2num(sign2num)[x])
    result = []
    for j in range(len(str_list)):
        result.append(ascii2num_func(np.fromstring(str_list[j], dtype = np.uint8)))
    return result

def count_wrong_eq(checker, predict_labels_str_list, vocab, mode, level_matched):
    cnt = 0
    for predict_labels_str in predict_labels_str_list:
        if checker(predict_labels_str, vocab, mode, level_matched) == False:
            cnt += 1
    return cnt

def filter_large_size(X, res_list, Z_num, predict_labels_str_list, thres=100):
    X_new, res_list_new, Z_num_new, predict_labels_str_list_new = [], [], [], []
    for x, r, z, p in zip(X, res_list, Z_num, predict_labels_str_list):
        if len(r[0])<thres:
            X_new.append(x)
            res_list_new.append(r)
            Z_num_new.append(z)
            predict_labels_str_list_new.append(p)
    return X_new, res_list_new, Z_num_new, predict_labels_str_list_new