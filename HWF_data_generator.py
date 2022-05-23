import json
import os
import random
import numpy as np
import cv2 as cv
import pickle as pk

import torchvision
trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True)

def load_expr(raw_dara_path, json_name):
    expr_path = os.path.join(raw_dara_path, json_name)
    return json.load(open(expr_path))

class HWFDataGenerator:
    def __init__(self, raw_dara_path, mode = "train", shape = (45, 45), is_color = False, basic_dataset = None):
        self.raw_data_path = raw_dara_path
        self.expr_train = load_expr(raw_dara_path, "expr_train.json")
        self.expr_test = load_expr(raw_dara_path, "expr_test.json")
        self.image_dir = os.path.join(raw_dara_path, "Handwritten_Math_Symbols")
        self.mode = mode
        self.default_shape = shape
        self.is_color = is_color
        if is_color:
            self.color = cv.IMREAD_COLOR
        else:
            self.color = cv.IMREAD_GRAYSCALE
        self.basic_dataset = basic_dataset

        self._image_set_init()

        self.__set_mode(mode)
        self.numbers = [str(c) for c in range(10)]

        self.images_pool = {}
        if self.basic_dataset is not None:
            for image, label in self.basic_dataset:
                label = str(label)
                self.images_pool.setdefault(label, [])
                self.images_pool[label].append(image)

    def _image_set_init(self):
        train_image_list = []
        test_image_list = []
        for expr in self.expr_train:
            train_image_list.extend(expr["img_paths"])

        for expr in self.expr_test:
            test_image_list.extend(expr["img_paths"])

        self.train_image_set = set(train_image_list)
        self.test_image_set = set(test_image_list)
        train_image_set = self.train_image_set
        test_image_set = self.test_image_set

        print()
        print("============== Report ==============")
        print("train images number:", len(train_image_list), "\t\ttrain unique images number:", len(train_image_set))
        print("test images number:", len(test_image_list), "\t\ttest unique images number:", len(test_image_set))
        print("all unique images number:", len(train_image_set.union(test_image_set)), "\ttrain unique plus test unique:", len(train_image_set) + len(test_image_set))
        print("====================================\n")

    def _create_dataset(self, image_set):
        X = []
        Y = []
        for image_path in image_set:
            image = self.read_image_npy(image_path)
            X.append(image)
            sign = image_path.split('/')[0]
            if sign == 'div':
                sign = '/'
            if sign == 'times':
                sign = '*'
            Y.append(sign)
        return X, Y

    def get_raw_dataset(self, mode = None):
        if mode == None:
            mode = self.mode
        
        if mode == "train":
            X, Y = self._create_dataset(self.train_image_set)
        elif mode == "test":
            X, Y = self._create_dataset(self.test_image_set)
        else:
            print("ERROR!")
        return X, Y

    def __set_mode(self, mode):
        if mode == "train":
            self.expr_pool = self.expr_train
        else:
            self.expr_pool = self.expr_test

    def _read_image_npy(self, img_path):
        sign = img_path.split('/')[-2]
        if self.basic_dataset:
            if sign in self.numbers:
                k = random.randint(0, len(self.images_pool[sign]) - 1)
                return np.array(self.images_pool[sign][k])
            
        img_path = os.path.join(self.image_dir, img_path)
        img = cv.imread(img_path, self.color)
        return img

    def read_image_npy(self, img_path):
        img = self._read_image_npy(img_path)
        if self.is_color:
            if len(img.shape) == 2 and img.shape[-1] == 1:
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            if len(img.shape) > 2 and img.shape[-1] == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img = cv.resize(img, (self.default_shape[0], self.default_shape[1]))
        return np.array(img)

    def read_images(self, img_paths):
        return [self.read_image_npy(img_path) for img_path in img_paths]

    def __call__(self, number):
        data_size = len(self.expr_pool)

        idxs = []
        while number > 0:
            tmp_idxs = list(range(data_size))
            random.shuffle(tmp_idxs)
            if number > data_size:
                idxs += tmp_idxs
            else:
                idxs += tmp_idxs[:number]
            number -= min(number, data_size)

        exprs = [self.expr_pool[idx] for idx in idxs]
        X = []
        Y = []
        Z = []
        for expr in exprs:
            expr_str = expr["expr"]
            result = expr["res"]
            img_paths = expr["img_paths"]
            imgs = self.read_images(img_paths)
            X.append(imgs)
            Y.append(result)
            Z.append(expr_str)
        return X, Y, Z
     
    def __len__(self):
        return len(self.expr_pool)


if __name__ == "__main__":
    generator = HWFDataGenerator("data", basic_dataset = trainset, is_color = False)
    X, Y, Z = generator(10)
    print(Y, Z)
    for idx, (images, labels, z) in enumerate(zip(X, Y, Z)):
        img = np.concatenate(images, axis=1)
        cv.imwrite(f"tmp/{labels}_{z}_{idx}.png", img)
        print(labels)
        if idx > 10:
            break
    
    X, Y = generator.get_raw_dataset(mode = "test")
    print(len(X), len(Y))
