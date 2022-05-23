import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import random
import itertools
import time
import sklearn
import argparse
from two_digit_add_data_generator_mnist import TwoDigitAddDataGenerateMNIST
from two_digit_add_data_generator_cifar import TwoDigitAddDataGenerateCIFAR
from HWF_data_generator import HWFDataGenerator
from similarity_calculator import nn_select_batch_abduced_result
from problog_ngs_abducer import Abducer
from logger import Logger
from utils import strs2idxs, filter_large_size
from NN_model_pytorch import ResNet50, LeNet, SymbolNet
from NN_model_pytorch import TorchDataset, train_transform, test_transform, Lenet_transform, SymbolNet_transform
from NN_model_pytorch import get_eqs_predict
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
import torch

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['HWF','2ADD'], default="2ADD", type=str, help="dataset (HWF or 2ADD")
    parser.add_argument('--images', choices=['handwritten','CIFAR'], default="handwritten", type=str, help="images (handwritten or CIFAR")
    parser.add_argument('--run', default=1, type=int, help='the run time for log file name, modify it every time' )
    parser.add_argument('--epochs', default=20, type=int, help='epochs for equation examples')
    # Model
    parser.add_argument('--num_pretrain_exs', default=0, type=int, help='number of examples for supervised pretraining')
    parser.add_argument('--num_pretrain_epoch', default=20, type=int, help='epoch for supervised pretraining')
    parser.add_argument('--pre_train_model_path', default=None, type=str, help="self-supervised weights of CIFAR-10 for resnet")#'results/128_0.5_200_512_500_model.pth'
    parser.add_argument('--nn_batch_size', default=256, type=int, help='batch size of nn training')
    parser.add_argument('--nn_fit_epoch', default=7, type=int, help='train epoch for nn')
    # Data
    parser.add_argument('--min_eq_length', default=5, type=int, help='shortest equation length')
    parser.add_argument('--batch_each_length', default=4, type=int, help='batches for each length equation')
    parser.add_argument('--eq_each_batch', default=2048, type=int, help="the number of equation per batch")
    # Abduction
    parser.add_argument('--abduction_batch_size', default=128, type=int, help="number of equations used for each abduction")
    parser.add_argument('--similar_coef', default=0.96, type=float, help="ratio of similarity scores, 1 means only similarity, 0 means only confidence, <0 means randomly selection")
    parser.add_argument('--beam_width', default=600, type=int, help="beam with for beam search")

    args = parser.parse_args()
    return args

def get_mapping():
    num2sign = dict()
    for i in range(10):
        num2sign[str(i)] = str(i)
    num2sign["10"] = "+"
    num2sign["11"] = "-"
    num2sign["12"] = "*"
    num2sign["13"] = "/"
    return num2sign

def train_model(args):
    dataset = args.dataset
    images_type = args.images
    if images_type == "CIFAR":
        input_shape = (32, 32, 3)
    else:
        if dataset == "HWF":
            input_shape = (45, 45, 1)
        else:
            input_shape = (28, 28, 1)
    run = args.run # The run time for log file name, modify it every time
    epochs = args.epochs # epochs for equation examples
    num_pretrain_exs = args.num_pretrain_exs # number of examples for supervised pretraining
    num_pretrain_epoch = args.num_pretrain_epoch # epoch for supervised pretraining
    pre_train_model_path = args.pre_train_model_path #None # 
    nn_batch_size = args.nn_batch_size # batch size of nn training
    nn_fit_epoch = args.nn_fit_epoch # train epoch for nn
    batch_each_length = args.batch_each_length # batches for each length equation
    eq_each_batch = args.eq_each_batch # the number of equation per batch
    abduction_batch_size = args.abduction_batch_size # number of equations used for each abduction
    abduction_batch_size = 128 #1
    similar_coef = args.similar_coef #The ratio of similarity scores, 1 means only similarity, 0 means only confidence, <0 means randomly selection
    beam_width = args.beam_width # beam with for beam search

    freeze_feature = False

    # Drawer
    logger = Logger(dataset, images_type, pre_train_model_path, num_pretrain_exs, batch_each_length, eq_each_batch, abduction_batch_size, beam_width, similar_coef)
    # Mapping
    num2sign = get_mapping()
    sign2num = dict(zip(num2sign.values(), [int(k) for k in num2sign.keys()]))

    # Data and abducer
    if dataset == '2ADD':
        if input_shape[0] == 28:
            generator = TwoDigitAddDataGenerateMNIST("deepprolog")
            train_data = MNIST(root='data', train=True, transform=Lenet_transform, download=True)
            test_data = MNIST(root='data', train=False, transform=Lenet_transform, download=True)
        elif input_shape[0] == 32:
            generator = TwoDigitAddDataGenerateCIFAR("deepprolog")
            train_data = CIFAR10(root='data', train=True, transform=train_transform, download=True)
            test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
        else:
            assert False
        abducer = Abducer(mode = "2ADD")
        num_classes = 10
        require_more_address = 1
    elif dataset == 'HWF':
        if input_shape[0] == 45:
            generator = HWFDataGenerator("data")
            X, Y = generator.get_raw_dataset(mode = "train")
            Y = np.array(strs2idxs(Y, sign2num))
            test_data = TorchDataset(X, Y, SymbolNet_transform)
        elif input_shape[0] == 32:
            generator = HWFDataGenerator("data", shape = (32, 32), basic_dataset = CIFAR10(root='data', train=True, download=True), is_color = True)
            X, Y = generator.get_raw_dataset(mode = "train")
            Y = np.array(strs2idxs(Y, sign2num))
            test_data = TorchDataset(X, Y, SymbolNet_transform)
        else:
            assert False
        abducer = Abducer(mode = "HWF")
        num_classes = 14
        require_more_address = 1

    # CNN
    if input_shape[0]==28:
        model = LeNet(num_class=num_classes, loss_criterion=torch.nn.CrossEntropyLoss(), batch_size=nn_batch_size).cuda()
    elif input_shape[0]==32:
        model = ResNet50(num_class=num_classes, pretrained_path=pre_train_model_path, loss_criterion=torch.nn.CrossEntropyLoss(), batch_size=nn_batch_size, freeze_feature=freeze_feature).cuda()
    else:
        model = SymbolNet(num_class=num_classes, loss_criterion=torch.nn.CrossEntropyLoss(), batch_size=nn_batch_size).cuda()


    labeled_X, labeled_y = None, []
    if num_pretrain_exs > 0:
        train_idxs = np.random.choice(len(train_data), num_pretrain_exs, replace=False)
        train_data = torch.utils.data.Subset(train_data, train_idxs)
        if input_shape[0] == 28:
            train_data_untrans = torch.utils.data.Subset(MNIST(root='data', train=True, transform=None), train_idxs)
        else:
            train_data_untrans = torch.utils.data.Subset(CIFAR10(root='data', train=True, transform=None), train_idxs)
        train_loader = DataLoader(train_data, batch_size=nn_batch_size, shuffle=False, num_workers=16, pin_memory=True)
        labeled_X = np.array([np.array(c[0]) for c in train_data_untrans])
        labeled_y = [c[1] for c in train_data_untrans]
        model.train_val(num_pretrain_epoch, True, data_loader=train_loader)


    X_all, Y_all, Z_all = generator(len(generator))
    Z_all_num = strs2idxs(Z_all, sign2num)
    print("Got data!")

    test_loader = DataLoader(test_data, batch_size=nn_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    image_test_loss, image_test_acc = model.train_val(1, False, data_loader=test_loader)
    logger.add_data(image_test_acc = image_test_acc)
    logger.to_pk(run)
    
    time_start=time.time()    
    for cur_epoch in range(1, epochs+1):
        # random.seed(0)
        # random.shuffle(X_all)
        # random.seed(0)
        # random.shuffle(Y_all)
        # random.seed(0)
        # random.shuffle(Z_all_num)
        X_all, Y_all, Z_all_num = sklearn.utils.shuffle(X_all, Y_all, Z_all_num)
        epoch_abduce_correct_sign_cnt, epoch_sign_total = 0, 0
        
        for begin in range(0, len(X_all), eq_each_batch):
            X, Y, Z_num = X_all[begin : begin + eq_each_batch], Y_all[begin : begin + eq_each_batch], Z_all_num[begin : begin + eq_each_batch]
            # Predict, abduce and train
            print("\nGetting all predicted label")
            predict_labels_str_list = get_eqs_predict(model, X, num2sign)
            print("Getting abduced label")

            res_list = abducer.abduce(predict_labels_str_list, Y, max_address_num = 100, require_more_address = require_more_address) #[(['0*1=0', '1*0=0', '1*1=1'], 1), (['2*2=4'], 1), ('3*3=9', 0), (['4*5=20'], 1)] #shuffle_str(alphabet)
            
            if dataset == 'HWF': #Filter
                if cur_epoch == 1 and begin <= eq_each_batch:
                    thres = 20
                else:
                    thres = 70
                X, res_list, Z_num, predict_labels_str_list = filter_large_size(X, res_list, Z_num, predict_labels_str_list, thres=thres)

            print("Generated abduction candidate list!")
            # Convert str to idx
            select_result = [strs2idxs(res[0], sign2num) for res in res_list]

            print("Selecting the best result according to similarity")
            if similar_coef >= 0:
                final_result = nn_select_batch_abduced_result(model, labeled_X, labeled_y, X, select_result, abduction_batch_size, Z_num, beam_width, similar_coef) 
            else:
                final_result = [res[random.randint(0,len(res)-1)] for res in select_result]
            
            batch_abduce_correct_sign_cnt, batch_sign_total = 0, 0
            for i in range(len(X)):
                cur_abduce_correct_cnt = sum(c1 == c2 for c1, c2 in zip(final_result[i], Z_num[i]))
                batch_abduce_correct_sign_cnt += cur_abduce_correct_cnt
                batch_sign_total += len(Z_num[i])
                if i % (eq_each_batch//5) == 0:
                    print("\nGround labels: ", Z_num[i])
                    print("Predict labels: ", predict_labels_str_list[i])
                    print("Abduced labels: ", select_result[i])
                    print("Final best label: ", final_result[i])
            epoch_abduce_correct_sign_cnt += batch_abduce_correct_sign_cnt
            epoch_sign_total += batch_sign_total

            labels_list = final_result#get_abduced_num(select_result, sign2num) 
            data_list = X
            if labeled_X is not None: # Use supervised data
                labels_list.append(labeled_y)
                data_list.append(labeled_X)
            images_np = np.array(list(itertools.chain.from_iterable(data_list)))
            labels_list = np.concatenate(labels_list, axis = 0)
            model.train_val(nn_fit_epoch, True, X=images_np, y=labels_list)

            print("Current epoch %d  Eq batch %d / %d  Abduce correct batch %.3f (%d / %d) Abduce correct epoch %.3f (%d / %d) "%(cur_epoch, \
            begin + eq_each_batch, len(X_all), batch_abduce_correct_sign_cnt/batch_sign_total, batch_abduce_correct_sign_cnt, batch_sign_total, epoch_abduce_correct_sign_cnt/epoch_sign_total, epoch_abduce_correct_sign_cnt, epoch_sign_total))

            image_test_loss, image_test_acc = model.train_val(1, False, data_loader=test_loader)
            logger.add_data(image_test_acc = image_test_acc, abduce_correct_rate = epoch_abduce_correct_sign_cnt/epoch_sign_total)
            logger.to_pk(run)
            time_end=time.time()
            print('------ time cost -------\n', time_end-time_start,'s')


if __name__ == "__main__":
    # Parameters
    args = arg_init()
    print(args)

    train_model(args)


    