import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import itertools
from tqdm import tqdm
import time
# from zoopt import ValueType, Dimension2, Objective, Parameter, Opt
import cupy as cp 

def sum2score(out_class_sum_list, out_class_cnt_list, inner_class_sum_list, inner_class_cnt_list):
    outer_class_mean_dist = out_class_sum_list / out_class_cnt_list
    inner_class_mean_dist = inner_class_sum_list / (inner_class_cnt_list - 1)
    nan_bool = cp.isnan(inner_class_mean_dist) # The line is time-costing
    idxs = cp.nonzero(nan_bool)
    avg = cp.nanmax(inner_class_mean_dist, axis=1) # Use the max to replace
    inner_class_mean_dist[idxs[0], idxs[1]] = avg[idxs[0]]
    score = cp.mean(outer_class_mean_dist - inner_class_mean_dist, axis=1)
    return score

def calc_new_sum_cnt(new, new2, old = None):
    # new: (beamsize*eqsize)x(n)
    # new2: (beamsize*eqsize)x(deltan)
    # old: (beamsize)x(n-deltan)
    if old is None:
        return new
    if old is not None:
        n = new.shape[1]
        deltan = new2.shape[1]
        eqsize = new.shape[0]//old.shape[0]
        assert(old.shape[1]==n-deltan and new.shape[0]%old.shape[0]==0)
        #final = old.repeat(eqsize, axis=0) # (beamsize*eqsize)x(n-deltan)
        #final += new[:,0:n-deltan] # (beamsize*eqsize)x(n-deltan)
        final = new.reshape((old.shape[0], eqsize, new.shape[1]))[:,:,0:old.shape[1]] + old.reshape((old.shape[0], 1, old.shape[1]))
        final = final.reshape((-1, final.shape[2]))
        final = cp.concatenate((final,new2), axis=1) # (beamsize*eqsize)x(n)
    return final

def score_label_similarity(label_lists_org, pair_dist_org, inner_sum_old = None, inner_cnt_old = None, outer_sum_old = None, outer_cnt_old = None):
    # label_lists: (beamsize*eqsize)x(n)
    # label_lists2: (beamsize*eqsize)x(deltan)
    # pair_dist: (n)x(deltan)
    # out_class_sum_list: (beamsize*eqsize)x(n)
    n = label_lists_org.shape[1]
    if inner_sum_old is None:
        delta_n = n
    else: # Calculate incrementally
        delta_n = n - inner_sum_old.shape[1]
    # Slice pair_distance
    assert(n <= pair_dist_org.shape[0])
    pair_distance = pair_dist_org[0:n, n-delta_n:n]

    label_lists = cp.array(label_lists_org)
    label_lists2 = label_lists[:, -delta_n:]
    pair_dist = cp.array(pair_distance)
    same_matrix = cp.equal(label_lists.reshape(-1, n, 1), label_lists2.reshape(-1, 1, delta_n))
    #diff_matrix = ~same_matrix
    # shape: (beamsize*eqsize)x(n)
    inner_class_sum_list = cp.sum(cp.multiply(same_matrix, pair_dist), axis=2)
    inner_class_cnt_list = cp.sum(same_matrix, axis=2)
    #out_class_sum_list = cp.sum(cp.multiply(diff_matrix, pair_dist), axis=2)
    out_class_sum_list = cp.sum(pair_dist, axis=1) - inner_class_sum_list
    #out_class_cnt_list = cp.sum(diff_matrix, axis=2)
    out_class_cnt_list = same_matrix.shape[2]-inner_class_cnt_list
    # shape: (beamsize*eqsize)x(deltan)
    inner_class_sum_list2 = cp.sum(cp.multiply(same_matrix, pair_dist), axis=1)
    inner_class_cnt_list2 = cp.sum(same_matrix, axis=1)
    #out_class_sum_list2 = cp.sum(cp.multiply(diff_matrix, pair_dist), axis=1)
    out_class_sum_list2 = cp.sum(pair_dist, axis=0) - inner_class_sum_list2
    #out_class_cnt_list2 = cp.sum(diff_matrix, axis=1)
    out_class_cnt_list2 = same_matrix.shape[1]-inner_class_cnt_list2

    out_class_sum_list = calc_new_sum_cnt(out_class_sum_list, out_class_sum_list2, outer_sum_old)
    out_class_cnt_list = calc_new_sum_cnt(out_class_cnt_list, out_class_cnt_list2, outer_cnt_old)
    inner_class_sum_list = calc_new_sum_cnt(inner_class_sum_list, inner_class_sum_list2, inner_sum_old)
    inner_class_cnt_list = calc_new_sum_cnt(inner_class_cnt_list, inner_class_cnt_list2, inner_cnt_old)
    score = sum2score(out_class_sum_list, out_class_cnt_list, inner_class_sum_list, inner_class_cnt_list)
    return score, out_class_sum_list, out_class_cnt_list, inner_class_sum_list, inner_class_cnt_list
    '''
    score = 0
    for i in range(len(label)):
        inner_class_dist_list = []
        outer_class_dist_list = []
        for j in range(0, len(label)):
            if i == j:
                continue
            if label[i] == label[j]:
                inner_class_dist_list.append(pair_dist[i][j])
            else:
                outer_class_dist_list.append(pair_dist[i][j])
        if len(inner_class_dist_list) == 0:
            inner_class_dist_list.append(0) # The average inter-class distance is better
        if len(outer_class_dist_list) == 0:
            outer_class_dist_list.append(0) # Average intra-class distance is better
        score += np.mean(outer_class_dist_list) - np.mean(inner_class_dist_list)
    score /= len(label)
    return score
    '''

def score_label_prob(label_lists, prob_val):
    # Slice prob_val
    assert(label_lists.shape[1] <= len(prob_val))
    prob_val = prob_val[0:label_lists.shape[1]]
    label_lists, prob_val = cp.array(label_lists), cp.array(prob_val)
    probs_list = prob_val[np.arange(len(prob_val)), label_lists]
    log_probs_list = cp.log(probs_list)
    log_prods_list = cp.sum(log_probs_list, axis=1, dtype = cp.float64)
    prods_list = cp.exp(log_prods_list, dtype = cp.float64)
    score_list = prods_list #/np.sum(prods_list)
    return score_list

def list_split(list1, abduced_list):
    ret = []
    cur = 0
    for i in range(len(abduced_list)):
        ret.append(list1[cur:cur+len(abduced_list[i][0])])
        cur += len(abduced_list[i][0])
    return ret

# def build_zoopt_dim(abduced_iterables):
#     dim_list = []
#     for it in abduced_iterables:
#         dim_list.append((ValueType.DISCRETE, [0, len(it)-1], False))
#     dim = Dimension2(dim_list)
#     return dim

# def score_label_zoopt(sol):
#     idxs = sol.get_x()
#     label = np.array(abduced_iterables_global)[np.arange(len(abduced_iterables_global)), idxs]
#     label = np.concatenate((labeled_y_global, label.flatten()))
#     label = np.array([label], dtype=np.int32)
#     if similar_coef_global > 0:
#         score_similarity_org_list, _, _, _, _ = score_label_similarity(label, pair_distance_global)
#         score_similarity_org_list = score_similarity_org_list.get() # TO CPU
#         score_similarity_list = preprocessing.scale(score_similarity_org_list) #TODO
#     if similar_coef_global < 1:
#         score_prob_org_list = score_label_prob(label, prob_val_global)
#         score_prob_list = preprocessing.scale(score_prob_org_list) #TODO
#     if similar_coef_global == 0:
#         score_list = score_prob_list
#     elif similar_coef_global == 1:
#         score_list = score_similarity_list
#     else:
#         score_list = similar_coef_global * score_similarity_list + (1 - similar_coef_global) * score_prob_list
#     score = -score_list[0]
#     return -score

# def select_abduced_result_zoopt(abduced_batch_list, pair_distance, prob_val, abduced_iterables, labeled_y, ground_label = None, similar_coef = 1):
#     global abduced_iterables_global
#     global pair_distance_global
#     global prob_val_global
#     global similar_coef_global
#     global labeled_y_global
#     abduced_iterables_global, pair_distance_global, prob_val_global, similar_coef_global, labeled_y_global = abduced_iterables, pair_distance, prob_val, similar_coef, labeled_y
#     dim = build_zoopt_dim(abduced_iterables)
#     obj = Objective(score_label_zoopt, dim)
#     solution = Opt.min(obj, Parameter(budget=10000, parallel=False, server_num=2))
#     print(solution.get_x(), solution.get_value())

#     idxs, score = solution.get_x(), -solution.get_value()
#     best_label = np.array(abduced_iterables)[np.arange(len(abduced_iterables_global)), idxs]
#     best_label = np.concatenate((labeled_y, best_label.flatten()))
#     best_label = np.array([best_label], dtype=np.int32)
#     print('best   score', score, score_label_similarity(best_label, pair_distance)[0][0], score_label_prob(best_label, prob_val)[0], best_label)
#     ground_all = np.array([labeled_y + ground_label])
#     print('ground score', similar_coef*score_label_similarity(ground_all, pair_distance)[0][0]+(1-similar_coef)*score_label_prob(ground_all, prob_val)[0], score_label_similarity(ground_all, pair_distance)[0][0], score_label_prob(ground_all, prob_val)[0], ground_label)
#     input()
#     return best_label

def select_abduced_result(pair_distance, prob_val, abduced_result, labeled_y, ground_label = None, beam_width = None, similar_coef = 1, inner_sum_old = None, inner_cnt_old = None, outer_sum_old = None, outer_cnt_old = None):
    '''
    inner_sum_old: (beamsize)x(n-deltan)
    inner_cnt_old: (beamsize)x(n-deltan)
    outer_sum_old: (beamsize)x(n-deltan)
    outer_cnt_old: (beamsize)x(n-deltan)
    abduced_result:(beamsize*eqsize)x(n)
    '''
    out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = inner_sum_old, inner_cnt_old, outer_sum_old, outer_cnt_old
    # Only one abduced result
    if len(abduced_result) == 1:
        return abduced_result, abduced_result[0], inner_sum_old, inner_cnt_old, outer_sum_old, outer_cnt_old
    # Slice prob_val
    assert(abduced_result.shape[1] <= len(prob_val))
    prob_val = prob_val[0:abduced_result.shape[1]]
    '''
    global abduced_iterables_global
    global pair_distance_global
    abduced_iterables_global = abduced_iterables
    pair_distance_global = pair_distance
    dim = build_zoopt_dim(abduced_iterables)
    obj = Objective(score_label_zoopt, dim)
    solution = Opt.min(obj, Parameter(budget=1000000, parallel=True, server_num=40))
    print(solution.get_x(), solution.get_value())
    print('best   score', -solution.get_value(), "".join([abduced_iterables[i][solution.get_x()[i]] for i in range(len(abduced_iterables))]))
    print('ground score', score_label(ground_labels, pair_distance, sign2num), ground_labels)
    '''
    # Score each abduced result and select the best
    if similar_coef > 0:
        score_similarity_org_list, out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = score_label_similarity(abduced_result, pair_distance, inner_sum_old, inner_cnt_old, outer_sum_old, outer_cnt_old)
        score_similarity_list = (score_similarity_org_list-score_similarity_org_list.mean())/score_similarity_org_list.std() # scale
    if similar_coef < 1:
        score_prob_org_list = score_label_prob(abduced_result, prob_val)
        score_prob_list = (score_prob_org_list-score_prob_org_list.mean())/score_prob_org_list.std() # scale
    if similar_coef == 0:
        score_list = score_prob_list
    elif similar_coef == 1:
        score_list = score_similarity_list
    else:
        score_list = similar_coef * score_similarity_list + (1 - similar_coef) * score_prob_list
    score_list = score_list.get()  # TO CPU
    best = np.argmax(score_list)
    #print('best   score', similar_coef*score_similarity_org_list[best]+(1-similar_coef)*score_prob_org_list[best], score_similarity_org_list[best], score_prob_org_list[best], list(abduced_result[best][len(labeled_y):]))
    #ground_all = np.array([labeled_y + ground_label])
    #print('ground score', similar_coef*score_label_similarity(ground_all, pair_distance)[0][0]+(1-similar_coef)*score_label_prob(ground_all, prob_val)[0],score_label_similarity(ground_all, pair_distance)[0][0], score_label_prob(ground_all, prob_val)[0], ground_label)
    #input()
    if beam_width == None:
        return None, abduced_result[best], None, None, None, None
    # Beam search
    if len(score_list) <= beam_width:
        return abduced_result, abduced_result[best], out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt
    top_k_score_idxs = np.argpartition(-np.array(score_list), beam_width)[0:beam_width]
    if similar_coef > 0:
        return abduced_result[top_k_score_idxs], abduced_result[best], out_class_sum[top_k_score_idxs], out_class_cnt[top_k_score_idxs], inner_class_sum[top_k_score_idxs], inner_class_cnt[top_k_score_idxs]
    else:
        return abduced_result[top_k_score_idxs], abduced_result[best], None, None, None, None

def get_eqs_feature(model, X):
    images_np = np.array(list(itertools.chain.from_iterable(X)))
    predict_prob_list, predict_feature_list = model.predict(X=images_np)
    predict_prob_list, predict_feature_list = predict_prob_list.cpu().numpy(), predict_feature_list.cpu().numpy()
    ret_prob_list, ret_feature_list, cur_idx = [], [], 0
    for eq in X:
        ret_prob_list.append(predict_prob_list[cur_idx : cur_idx + len(eq)])
        ret_feature_list.append(predict_feature_list[cur_idx : cur_idx + len(eq)])
        cur_idx += len(eq)
    assert (cur_idx == len(predict_feature_list) and cur_idx == len(predict_prob_list))
    return ret_prob_list, ret_feature_list

def nn_select_batch_abduced_result(model, labeled_X, labeled_y, imgs_list, abduced_list, abduction_batch_size = 3, ground_labels_list = None, beam_width= None, similar_coef = 0.9):
    print("Getting labeled data's prob and feature")
    if labeled_X is not None:
        prob_val_labeled_list, dense_val_labeled_list = model.predict(X=labeled_X)
        prob_val_labeled_list, dense_val_labeled_list = prob_val_labeled_list.cpu().numpy(), dense_val_labeled_list.cpu().numpy()
    print("Getting eqs' prob and feature")
    prob_val_eq_list, dense_val_eq_list = get_eqs_feature(model, imgs_list)

    print("Select each batch's eqs based on score")
    best_abduced_list = []
    for i in tqdm(range(0, len(abduced_list), abduction_batch_size)): # Every batch eq
        dense_val_list = np.concatenate(dense_val_eq_list[i:i+abduction_batch_size])
        prob_val_list = np.concatenate(prob_val_eq_list[i:i+abduction_batch_size])
        if labeled_X is not None:
            dense_val_list = np.concatenate((dense_val_labeled_list, dense_val_list))
            prob_val_list = np.concatenate((prob_val_labeled_list, prob_val_list))
        # Compared distance for img pair
        pair_distance = pairwise_distances(dense_val_list, metric="cosine")
        if beam_width == None or abduction_batch_size==1:
            abduced_results = gen_abduced_list(([labeled_y],*abduced_list[i:i+abduction_batch_size]))
            ground_label = None#list(itertools.chain.from_iterable(ground_labels_list[i:i+abduction_batch_size]))
            _, best_abduced_batch, _, _, _, _ = select_abduced_result(pair_distance, prob_val_list, abduced_results, labeled_y, ground_label, beam_width, similar_coef)
        else: # Beam search
            abduced_batch_list = gen_abduced_list(([labeled_y], abduced_list[i]))
            out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = None, None, None, None
            for j in range(i + 1, min(i + abduction_batch_size, len(abduced_list))): # Beam search
                abduced_results = gen_abduced_list((abduced_batch_list, abduced_list[j]))
                ground_label = None#list(itertools.chain.from_iterable(ground_labels_list[i:j+1]))
                abduced_batch_list, best_abduced_batch, out_class_sum, out_class_cnt, inner_class_sum, inner_class_cnt = select_abduced_result(pair_distance, prob_val_list, abduced_results, labeled_y, ground_label, beam_width, similar_coef = similar_coef, inner_sum_old = inner_class_sum, inner_cnt_old = inner_class_cnt, outer_sum_old = out_class_sum, outer_cnt_old = out_class_cnt)
            #best_abduced_batch = select_abduced_result_zoopt(abduced_batch_list, pair_distance, prob_val_list, abduced_list[i:i+abduction_batch_size], labeled_y, list(itertools.chain.from_iterable(ground_labels_list[i:i+abduction_batch_size])), similar_coef)
        best_abduced_list.extend(list_split(best_abduced_batch[len(labeled_y):], abduced_list[i:i+abduction_batch_size]))
    return best_abduced_list

def gen_abduced_list(abduced_iterables):
    # Generate abduced candidates
    if len(abduced_iterables) == 2:
        a = np.array(abduced_iterables[0], dtype=np.uint8)
        b = np.array(abduced_iterables[1], dtype=np.uint8)
        left = np.repeat(a, len(b), axis=0)
        right = np.tile(b, (len(a),1))
        result = np.zeros((left.shape[0], left.shape[1]+right.shape[1]), dtype=np.uint8)
        result[:,:left.shape[1]]=left
        result[:,left.shape[1]:]=right
        #result = np.concatenate((left, right),axis=1)
        return result
    abduced_results = []
    for abduced in itertools.product(*abduced_iterables):
        abduced_results.append(list(itertools.chain.from_iterable(abduced)))
    return np.array(abduced_results, dtype=np.uint8)

def nn_test_class(cur_class1, cur_class2, hash_val_list):
    #print("Start ", cur_class1, cur_class2)
    idxs1 = np.array(np.where(y_train==cur_class1)[0])
    idxs2 = np.array(np.where(y_train==cur_class2)[0])
    if cur_class1 == cur_class2:
        result = pairwise_distances(hash_val_list[idxs1], metric="cosine")
        avg_dis = np.mean(result.flatten())
        #print("\nInner class %d\n Average distance %f" % (cur_class1, avg_dis))
    else:
        result = pairwise_distances(hash_val_list[idxs1], hash_val_list[idxs2], metric="cosine")
        avg_dis = np.mean(result.flatten())
        #print("\nBetween class %d %d\n Average distance %f" % (cur_class1, cur_class2, avg_dis))
    return avg_dis

if __name__ == "__main__":    
    sign2num = {"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "+":10, "=":11, "*":12}
    
    
    # dist_list1 = []
    # for cur_class1 in range(0, 10):
    #     dist = nn_test_class(cur_class1, cur_class1, dense_val_list)
    #     dist_list1.append(dist)
    # print("Average", np.mean(dist_list1))

    # dist_list2 = []
    # for cur_class1 in range(0, 10):
    #     for cur_class2 in range(cur_class1+1, 10):
    #         dist = nn_test_class(cur_class1, cur_class2, dense_val_list)
    #         dist_list2.append(dist)
    # print("Average", np.mean(dist_list2))
    # ratio = np.mean(dist_list2)/np.mean(dist_list1)
    # print("Ratio", ratio)


