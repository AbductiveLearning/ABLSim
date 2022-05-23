class ProdKB:
    def __init__(self, zero_check = False):
        self.zero_check = zero_check
        pass

    def _find_sign(self, sequence, sign):
        info = None
        for idx, char in enumerate(sequence):
            if char == sign:
                if info is None:
                    info = idx
                else:
                    return False, 2
        if info is None:
            return False, 0
        return True, info

    def _convert(self, sequence):
        if sequence is None or len(sequence) == 0:
            return None
        if len(sequence) > 1 and sequence[0] == '0':
            return None
        for c in sequence:
            if c.isdigit() == False:
                return None
        num = int("".join(sequence))
        return num

    def _know(self, sequence, vocab):
        for c in sequence:
            if c not in vocab:
                return False
        return True

    def _check_level_matched(self, sequence, level_matched):
        for c in sequence:
            if c == str(level_matched - 1):
                return True
        return False

    def __call__(self, sequence, vocab, mode = "sum", level_matched = None):
        if self._know(sequence, vocab) == False:
            return False
        if (level_matched is not None) and self._check_level_matched(sequence, level_matched) == False:
            return False

        if mode == "sum":
            opt = lambda x, y: x + y
            opt_sign = "+"
        elif mode == "prod":
            opt = lambda x, y: x * y
            opt_sign = "*"

        opt_sign_info = self._find_sign(sequence, opt_sign)
        if opt_sign_info[0] == False:
            return False

        equal_sign_info = self._find_sign(sequence, "=")
        if equal_sign_info[0] == False:
            return False

        opt_sign_idx = opt_sign_info[1]
        equal_sign_idx = equal_sign_info[1]
        
        last = len(sequence) - 1
        if opt_sign_idx == 0 or opt_sign_idx == last or\
                equal_sign_idx == 0 or equal_sign_idx == last or\
                opt_sign_idx > equal_sign_idx:
            return False
        
        number1 = sequence[:opt_sign_idx]
        number2 = sequence[opt_sign_idx + 1: equal_sign_idx]
        result = sequence[equal_sign_idx + 1:]

        number1 = self._convert(number1)
        number2 = self._convert(number2)
        result = self._convert(result)
        if number1 is None or number2 is None or result is None:
            return False
        if self.zero_check:
            if number1 * number2 * result == 0:
                return False
        if opt(number1, number2) != result:
            return False
        return True

if __name__ == "__main__":
    kb = ProdKB()
    test_data = [
        ("1*1=1", True),
        ("1*1*1=1", False),
        ("2*2=4", True),
        ("c*a=b", False),
        ("111=111", False),
        ("111*111", False),
        ("0*5=20", False),
        ("0*0=00", False)
    ]
    for data, flag in test_data:
        res = kb(data, vocab="0123456789*=", mode="prod", level_matched=4)
        #flag)
        if (flag != res):
            print(flag, res, data)
            print("################")


    test_data = [
        ("1+1=2", True),
        ("1+1+1=1", False),
        ("2+2=4", True),
        ("c+a=b", False),
        ("111=111", False),
        ("111+111", False),
        ("0+5=20", False),
        ("0+0=0", True)
    ]
    for data, flag in test_data:
        res = kb(data, vocab="0123456789+=", mode="sum")
        #flag)
        if (flag != res):
            print(flag, res, data)
            print("################")

