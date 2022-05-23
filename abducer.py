import itertools
from kb import ProdKB
import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle as pk
import os

class Abducer:
    def __init__(self, checker, vocab):
        self.checker = checker
        self.default_vocab = vocab
        self.ans_set_dict = dict()
        self.filename = "abducer.pk"
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.ans_set_dict = pk.load(f)

    def _try_abduce(self, tmp_sequence, possible_fault_pos_list, mode, vocab, level_matched):
        possible_fix_mathod_list = itertools.product(vocab, repeat = len(possible_fault_pos_list))
        ret = []
        for possible_fix_method in possible_fix_mathod_list:
            for pos, value in zip(possible_fault_pos_list, possible_fix_method):
                tmp_sequence[pos] = value
                if self.checker(tmp_sequence, vocab, mode, level_matched):
                    ret.append("".join(tmp_sequence))
        return ret

    def _abduce(self, sequence, max_address_num, mode, vocab, require_more_address, level_matched):
        if self.checker(sequence, vocab, mode, level_matched) == True:
            return [sequence], 0
        max_address_num = min(max_address_num, len(sequence))
        best_address_num = -1
        ret = []
        for address_num in range(1, max_address_num + 1):
            possible_faults_list = itertools.combinations(range(len(sequence)), address_num)
            for possible_faults in possible_faults_list:
                tmp_sequence = [c for c in sequence]
                res = self._try_abduce(tmp_sequence, possible_faults, mode, vocab, level_matched)
                if len(res) == 0:
                    continue
                ret.extend(res)
            if (len(ret) > 0):
                if (best_address_num == -1):
                    best_address_num = address_num
                require_more_address -= 1
            if (require_more_address == -1):
                return list(set(ret)), best_address_num
        return ret, best_address_num

    def _abduce_quick(self, sequence, max_address_num, mode, vocab, require_more_address, level_matched):
        ans_set, _ = self._return_ans_set(len(sequence), mode, vocab, level_matched)
        hamming_dist_list = np.array([sum(c1 != c2 for c1, c2 in zip(sequence, s)) for s in ans_set])
        address_num = np.min(hamming_dist_list)
        idxs = np.where(hamming_dist_list<=address_num+require_more_address)[0]
        return [ans_set[idx] for idx in idxs], address_num

    def abduce(self, sequences, max_address_num, mode, vocab = None, require_more_address = 0, level_matched = None):
        if vocab is None:
            vocab = self.default_vocab
        
        ret = []
        # Pre calculate for thread write safe
        self._return_ans_set(5, mode, vocab, level_matched)
        self._return_ans_set(6, mode, vocab, level_matched)
        self._return_ans_set(7, mode, vocab, level_matched)
        if not os.path.exists(self.filename):
           with open(self.filename, "wb") as f:
               pk.dump(self.ans_set_dict, f)
        if require_more_address == 99: # Return answer set
            for sequence in sequences:
                ret.append(self._return_ans_set(len(sequence), mode, vocab, level_matched))
        else:
            partial__abduce_quick = partial(self._abduce_quick, max_address_num = max_address_num, mode = mode, vocab = vocab, require_more_address = require_more_address, level_matched = level_matched)
            pool = Pool(processes = 30)
            ret = pool.map(partial__abduce_quick, sequences)
            pool.close()
            pool.join()
        return ret

    def _return_ans_set(self, length, mode, vocab, level_matched = None):
        key = (length, mode, vocab, level_matched)
        if self.ans_set_dict.get(key) is not None:
            return self.ans_set_dict[key]
        ret = []
        possible_seq_list = itertools.product(vocab, repeat = length)
        for possible_seq in possible_seq_list:
            if self.checker(possible_seq, vocab, mode, level_matched):
                ret.append("".join(possible_seq))
        self.ans_set_dict[key] = ret, length
        return ret, length

if __name__ == "__main__":
    checker = ProdKB()
    vocab = "0123456789*="
    abducer = Abducer(checker, vocab)

    sequences = ["1+1=2", "111=0", "2*2=3", "3*3=9", "4*5=26", "3*3=9"]

    vocab = "0123456789+="
    res = abducer.abduce(sequences, 3, mode = "sum", vocab = vocab, require_more_address = 1, level_matched = None)
    print(res)

    res = abducer.abduce(sequences, 3, mode = "prod", vocab = "01*=")
    print(res)

    res = abducer._return_ans_set(5, mode = "sum", vocab = vocab, level_matched = None)
    print(res)
    res = abducer._return_ans_set(6, mode = "sum", vocab = vocab, level_matched = None)
    print(res)
    res = abducer._return_ans_set(7, mode = "sum", vocab = vocab, level_matched = None)
    print(res)
    res = abducer._return_ans_set(8, mode = "sum", vocab = vocab, level_matched = None)
    print(res)