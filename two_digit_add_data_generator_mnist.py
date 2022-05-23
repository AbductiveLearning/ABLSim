import torchvision
import numpy as np
import os
import random
import cv2 as cv

trainset = torchvision.datasets.MNIST(root='data', train=True, download=True)
testset = torchvision.datasets.MNIST(root='data', train=False, download=True)

pix = np.array(trainset[0][1])
print(pix)
print(len(trainset[0]))

def load_expr(raw_dara_path, data_file):
    data_path = os.path.join(raw_dara_path, data_file)
    with open(data_path) as fin:
        raw_data = [d.strip() for d in fin]
    data = []
    for d in raw_data:
        t = d.split(')')[:3]
        a = t[0].split('(')[-1]
        b = t[1].split('(')[-1]
        c = t[2][1:]
        data.append(int(x) for x in [a, b, c])
    return data

class TwoDigitAddDataGenerateMNIST:
    def __init__(self, raw_dara_path, mode = "train", shape = (28, 28)):
        self.raw_data_path = raw_dara_path
        self.expr_train = load_expr(raw_dara_path, "train_data.txt")
        self.expr_test = load_expr(raw_dara_path, "test_data.txt")
        self.mode = mode
        self.default_shape = shape
        self.__set_mode(mode)

    def __set_mode(self, mode):
        if mode == "train":
            self.dataset = trainset
            self.expr_pool = self.expr_train
        else:
            self.dataset = testset
            self.expr_pool = self.expr_test

    def read_image_npy(self, img_path):
        img_path = os.path.join(self.image_dir, img_path)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
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
            a, b, c = expr

            a_info = self.dataset[a]
            b_info = self.dataset[b]
            img_a = np.array(a_info[0])
            img_b = np.array(b_info[0])
            
            add1 = a_info[1]
            add2 = b_info[1]

            if add1 + add2 != c:
                print(add1, add2, c)

            X.append([img_a, img_b])
            Y.append(c)
            Z.append("%d%d" % (add1, add2))

        return X, Y, Z
     
    def __len__(self):
        return len(self.expr_pool)


if __name__ == "__main__":
    generator = TwoDigitAddDataGenerateMNIST("deepprolog")
    # X, Y, Z = generator(len(generator))
    # print(len(X), len(Y), len(Z))
    
    X, Y, Z = generator(10)
    print(Y, Z)
    for idx, (images, labels, z) in enumerate(zip(X, Y, Z)):
        img = np.concatenate(images, axis=1)
        cv.imwrite(f"tmp/{labels}_{z}_{idx}.png", img)
        print(labels)
        if idx > 10:
            break
