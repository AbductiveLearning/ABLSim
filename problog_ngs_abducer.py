from multiprocessing import Pool
import bisect
from functools import partial
import numpy as np

def expr_generate(length, opts, nums, cur_list, ret):
    if len(cur_list) == length:
        ret.append("".join(cur_list))
        return
    if len(cur_list) % 2 == 0:
        cands = nums
    else:
        cands = opts

    for cand in cands:
        if len(cur_list) > 0 and cur_list[-1] == '/' and cand == 0:
            continue
        cur_list.append(str(cand))
        expr_generate(length, opts, nums, cur_list, ret)
        cur_list.pop()

class Abducer:
    def __init__(self, min_len = 1, max_len = 7, mode = "HWF"):
        self.mode = mode
        if mode == "HWF":
        	opts = ["+", "-", "*", "/"]
        	nums = list(range(1, 10))
        elif mode == "2ADD":
        	opts = ["+"]
        	nums = list(range(0, 10))
        else:
            assert mode in ["HWF", "2ADD"]

        self.KB = {}
        for l in range(min_len, max_len + 1, 2):
            expr_list = []
            expr_generate(l, opts, nums, [], expr_list)
            ans_list = map(eval, expr_list)
            expr_dict = {}
            for ans, expr in zip(ans_list, expr_list):
                 expr_dict.setdefault(ans, [])
                 expr_dict[ans].append(expr)
            ans_list = [k for k, _ in expr_dict.items()]
            expr_list = [v for _, v in expr_dict.items()]
            equation_list = sorted(list(zip(ans_list, expr_list)))
            ans_list = [e[0] for e in equation_list]
            expr_list = [e[1] for e in equation_list]
            self.KB[l] = (ans_list, expr_list)

    def _get_ans_set(self, expr, ans):
        l = len(expr)
        sub_KB = self.KB[l]
        ans_list = sub_KB[0]
        expr_list = sub_KB[1]

        idx = bisect.bisect_left(sub_KB[0], ans)
        begin = max(0, idx - 1)
        end = min(idx + 2, len(sub_KB[0]))

        min_err = 999999
        best_set = []
        for idx in range(begin, end):
            err = abs(ans_list[idx] - ans)
            if err < min_err:
                best_set = expr_list[idx]
                min_err = err
        return best_set

    def _abduce(self, data, max_address_num, require_more_address):
         sequence, ans = data
         ans_set = self._get_ans_set(sequence, ans)
         hamming_dist_list = np.array([sum(c1 != c2 for c1, c2 in zip(sequence, s)) for s in ans_set])
         address_num = np.min(hamming_dist_list)
         idxs = np.where(hamming_dist_list<=address_num+require_more_address)[0]

         strip = 1
         if self.mode == "2ADD":
            strip = 2
         return [ans_set[idx][::strip] for idx in idxs], address_num

    def abduce(self, sequences, anss, max_address_num, require_more_address = 0):
         if self.mode == "2ADD":
             sequences = ["+".join(s for s in sequence) for sequence in sequences]
         partial__abduce = partial(self._abduce, max_address_num = max_address_num, require_more_address = require_more_address)
         #pool = Pool(processes = 30)
         #ret = pool.map(partial__abduce, zip(sequences, anss))
         #pool.close()
         #pool.join()
         ret = list(map(partial__abduce, zip(sequences, anss)))
         return ret

if __name__ == "__main__":
    # abducor = Abducer(mode = "HWF")
    # ans = abducor(["2*3", "2+3"], [5, 6], 1)
    # print(ans)
    abducor = Abducer(mode = "2ADD")
    ans = abducor.abduce(["23", "23", "87"], [5, 6, 15], 1, 2)
    print(ans)

