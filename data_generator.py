import os
import collections
import random
import itertools
import cv2 as cv
import numpy as np

class ProdDataGenerator:
    def __init__(self, data_dir, mapping, mode = "prod", shape = (28, 28, 1), require_level_matched = True):
        self.level = 1
        self.image_pool = collections.defaultdict(list)
        self.default_shape = shape
        self.vocub = mapping.values()

        image_suffix = ["jpg", "png", "jpeg"]
        for root, dirs, files in os.walk(data_dir):
            sign = os.path.split(root)[-1]
            if sign not in mapping:
                continue
            label = mapping[sign]
            filepaths = [os.path.join(root, filename) for filename in files if filename.split(".")[-1].lower() in image_suffix]
            self.image_pool[label] = filepaths          

        self.set_mode(mode)
        self.require_level_matched = require_level_matched

    def set_mode(self, mode):
        assert mode in ["prod", "sum"]
        self.mode = mode
        if mode == "prod":
            self._opt = lambda a, b : a * b
            self.opt_sign = "*"
        elif mode == "sum":
            self._opt = lambda a, b : a + b
            self.opt_sign = "+"

    def _generate_nums(self, length, level):
        nums = []
        str_digits = "".join([str(c) for c in range(0, level)])
        nums = itertools.product(str_digits, repeat=length)
        nums = [int("".join(s)) for s in nums]
        return nums

    def _get_num_len(self, num):
        return len(str(num))

    def has_new_letter(self, num):
        new_c = str(self.level - 1)
        for c in str(num):
            if c == new_c:
                return True
        return False
    
    def _get_all_possible(self, length, level, include_zero):
        nums = self._generate_nums(length - 3, self.level)
        if include_zero == False:
            nums.remove(0)
        nums = sorted(nums)
        nums_set = set(nums)
        all_possible_list = []
        for n1 in nums:
            for n2 in nums:
                res = self._opt(n1, n2)
                if res not in nums_set:
                    continue
                if self.require_level_matched:
                    if self.has_new_letter(n1) == False \
                            and self.has_new_letter(n2) == False \
                            and self.has_new_letter(res) == False:
                        continue
                l1 = self._get_num_len(n1)
                l2 = self._get_num_len(n2)
                lr = self._get_num_len(res)

                if l1 + l2 + lr + 2 == length:
                    all_possible_list.append(f"{n1}{self.opt_sign}{n2}={res}")
                if l1 + l2 + lr + 2 > length:
                    break
        return all_possible_list

    def _get_all_impossible(self, length, level, include_zero):
        nums = self._generate_nums(length - 3, self.level)
        if include_zero == False:
            nums.remove(0)
        nums = sorted(nums)
        nums_set = set(nums)
        ret_list = []
        for n1 in nums:
            for n2 in nums:
                res = self._opt(n1, n2)
                flag = random.randint(0, 2)
                if flag == 0:
                    n1 = random.randint(n1 // 2, n1 * 2)
                elif flag == 1:
                    n2 = random.randint(n2 // 2, n2 * 2)
                elif flag == 2:
                    res = random.randint(res // 2, res * 2)

                if self._opt(n1, n2) == res:
                    continue

                l1 = self._get_num_len(n1)
                l2 = self._get_num_len(n2)
                lr = self._get_num_len(res)
                if l1 + l2 + lr + 2 > length:
                    break

                if self.require_level_matched:
                    if self.has_new_letter(n1) == False \
                            and self.has_new_letter(n2) == False \
                            and self.has_new_letter(res) == False:
                        continue

                if res not in nums_set or n1 not in nums_set or n2 not in nums_set:
                    continue
                if l1 + l2 + lr + 2 == length:
                    ret_list.append(f"{n1}{self.opt_sign}{n2}={res}")

        return ret_list

    def _imagelization(self, equations):
        ret = []
        for equation in equations:
            images = []
            for c in equation:
                pool = self.image_pool[c]
                image_idx = random.randint(0, len(pool) - 1)
                image_path = pool[image_idx]
                if self.default_shape[2] == 1:
                    image_resized = cv.resize(cv.imread(image_path, cv.IMREAD_GRAYSCALE), (self.default_shape[0], self.default_shape[1]))
                else:
                    image_resized = cv.resize(cv.imread(image_path, cv.IMREAD_COLOR), (self.default_shape[0], self.default_shape[1]))
                images.append(np.array(image_resized).reshape(self.default_shape))
            ret.append(images)
        return ret

    def get_batch_data(self, is_valid = True, length = 5, include_zero = False, batch_size = 1):
        equations = []
        if is_valid:
            equations_list = self._get_all_possible(length, self.level, include_zero)
            equations_num = len(equations_list)
            if equations_num > 0:
                equations = [random.choice(equations_list) for _ in range(batch_size)]
        else:
            equations_list = self._get_all_impossible(length, self.level, include_zero)
            equations_num = len(equations_list)
            if equations_num > 0:
                equations = [random.choice(equations_list) for _ in range(batch_size)]
         
        if len(equations) == 0:
            return [], []
        
        return self._imagelization(equations), equations

    def evolution(self):
        if self.level < 10:
            self.level += 1

if __name__ == "__main__":
    mapping = dict()
    for i in range(10):
        mapping[str(i)] = str(i)
    mapping["12"] = "*"
    mapping["10"] = "+"
    mapping["11"] = "="
    data_generator = ProdDataGenerator("mnist_images/training", mapping, mode = "sum", require_level_matched = 1)
    data_generator.evolution()
    data_generator.evolution()
    data_generator.evolution()
    data_generator.evolution()
    data_generator.evolution()
    data_generator.evolution()
    data_generator.evolution()
    data_generator.evolution()
    data_generator.evolution()

    X, Y = data_generator.get_batch_data(length = 5, is_valid = True, include_zero = True, batch_size = 1)
    for idx, (images, labels) in enumerate(zip(X, Y)):
        img = np.concatenate(images, axis=1)
        #print(img.shape)
        labels = labels.replace("*", "x")
        cv.imwrite(f"tmp/{labels}_{idx}.png", img)
        print(labels)
        #print(len(images), images[0].shape)
        #np.concat()
    #print(batch_data[1])

