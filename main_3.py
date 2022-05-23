# coding=utf-8
import numpy as np
import os
import random
import itertools
import time
import argparse
from similarity_calculator import nn_select_batch_abduced_result
from logger import Logger
from abducer import Abducer
from kb import ProdKB
from data_generator import ProdDataGenerator
from utils import strs2idxs

os.environ['CUDA_VISIBLE_DEVICES']='0'
from NN_model_pytorch import ResNet50, LeNet
from NN_model_pytorch import train_transform, test_transform, Lenet_transform
from NN_model_pytorch import get_eqs_predict
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
import torch

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', choices=['handwritten','CIFAR'], default="CIFAR", type=str, help="images (handwritten or CIFAR")
    parser.add_argument('--run', default=1, type=int, help='the run time for log file name, modify it every time' )
    parser.add_argument('--epochs', default=20, type=int, help='epochs for equation examples')
    # Model
    parser.add_argument('--num_pretrain_exs', default=0, type=int, help='number of examples for supervised pretraining')
    parser.add_argument('--num_pretrain_epoch', default=20, type=int, help='epoch for supervised pretraining')
    parser.add_argument('--pre_train_model_path', default='results/128_0.5_200_512_500_model.pth', type=str, help="self-supervised weights of CIFAR-10 for resnet")
    parser.add_argument('--nn_batch_size', default=256, type=int, help='batch size of nn training')
    parser.add_argument('--nn_fit_epoch', default=5, type=int, help='train epoch for nn')
    # Data
    parser.add_argument('--min_eq_length', default=5, type=int, help='shortest equation length')
    parser.add_argument('--batch_each_length', default=4, type=int, help='batches for each length equation')
    parser.add_argument('--eq_each_batch', default=2048, type=int, help="the number of equation per batch")
    # Abduction
    parser.add_argument('--abduction_batch_size', default=128, type=int, help="number of equations used for each abduction")
    parser.add_argument('--similar_coef', default=0.99, type=float, help="ratio of similarity scores, 1 means only similarity, 0 means only confidence, <0 means randomly selection")
    parser.add_argument('--beam_width', default=700, type=int, help="beam with for beam search")

    args = parser.parse_args()
    return args

def get_mapping():
    num2sign = dict()
    for i in range(10):
        num2sign[str(i)] = str(i)
    num2sign["10"] = "+"
    num2sign["11"] = "="
    num2sign["12"] = "*"
    return num2sign

def train_model(args):
    images_type = args.images
    run = args.run # The run time for log file name, modify it every time
    epochs = args.epochs # epochs for equation examples
    num_pretrain_exs = args.num_pretrain_exs # number of examples for supervised pretraining
    num_pretrain_epoch = args.num_pretrain_epoch # epoch for supervised pretraining
    pre_train_model_path = args.pre_train_model_path #None # self-supervised weights of CIFAR-10 for resnet
    nn_batch_size = args.nn_batch_size # batch size of nn training
    nn_fit_epoch = args.nn_fit_epoch # train epoch for nn
    min_eq_length = args.min_eq_length # Shortest equation length
    batch_each_length = args.batch_each_length # batches for each length equation
    eq_each_batch = args.eq_each_batch # the number of equation per batch
    abduction_batch_size = args.abduction_batch_size # number of equations used for each abduction
    similar_coef = args.similar_coef #The ratio of similarity scores, 1 means only similarity, 0 means only confidence, <0 means randomly selection
    beam_width = args.beam_width # beam with for beam search
    if images_type == "CIFAR":
        input_shape = (32, 32, 3)
    else:
        input_shape = (28, 28, 1)
    
    mode = "sum"
    require_level_matched = False # If each task contains the old equation
    max_eq_length = 6 # Longest equation length
    num_classes = 13
    freeze_feature = False # freeze the feature layers in resnet, only available when using resnet

    # Drawer
    logger = Logger("DEC", images_type, pre_train_model_path, num_pretrain_exs, batch_each_length, eq_each_batch, abduction_batch_size, beam_width, similar_coef)

    # CNN
    if input_shape[0] == 28:
        model = LeNet(num_class=num_classes, loss_criterion=torch.nn.CrossEntropyLoss(), batch_size=nn_batch_size).cuda()
        train_data = MNIST(root='data', train=True, transform=Lenet_transform, download=True)
        test_data = MNIST(root='data', train=False, transform=Lenet_transform, download=True)
    else:
        model = ResNet50(num_class=num_classes, pretrained_path=pre_train_model_path, loss_criterion=torch.nn.CrossEntropyLoss(), batch_size=nn_batch_size, freeze_feature=freeze_feature).cuda()
        train_data = CIFAR10(root='data', train=True, transform=train_transform, download=True)
        test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
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

    # Mapping
    num2sign = get_mapping()
    sign2num = dict(zip(num2sign.values(), [int(k) for k in num2sign.keys()]))

    # Data and abducer
    sum_alphabet = "0+=" # default alphabet 
    sum_data_generator = ProdDataGenerator("data/mnist_images/training" if input_shape[0] == 28 else "data/cifar_images/training", num2sign, mode = "sum", shape = input_shape, require_level_matched = require_level_matched)
    checker = ProdKB(zero_check = False)
    sum_abducer = Abducer(checker, sum_alphabet)

    level = 10
    for i in range(2, level+1):        
        sum_data_generator.evolution()
        sum_alphabet += str(i-1)
    print("Current alphabet: ", sum_alphabet)
    
    X_all, Y_all = [], []
    for cur_length in range(min_eq_length, max_eq_length + 1):
        if mode == "sum":
            X_sum, Y_sum = sum_data_generator.get_batch_data(length = cur_length, is_valid = True, include_zero = True, batch_size = batch_each_length * eq_each_batch)
            X_all.extend(X_sum)
            Y_all.extend(Y_sum)
            alphabet, abducer = sum_alphabet, sum_abducer
    print("Got batch data!")

    test_loader = DataLoader(test_data, batch_size=nn_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    image_test_loss, image_test_acc = model.train_val(1, False, data_loader=test_loader)

    train_images_np = np.array(list(itertools.chain.from_iterable(X_all)))
    train_images_labels = np.concatenate(strs2idxs(Y_all, sign2num), axis = 0)
    predict_embeddings = model.predict(X=train_images_np, is_train=False)[1].cpu().numpy()
    logger.add_data(image_test_acc = image_test_acc, embeddings = predict_embeddings.copy(), embeddings_y = train_images_labels)
    logger.to_pk(run)
    
    for cur_epoch in range(1, epochs+1):
        random.seed(0)
        random.shuffle(X_all)
        random.seed(0)
        random.shuffle(Y_all)
        epoch_abduce_correct_sign_cnt, epoch_sign_total = 0, 0
        require_more_address = 99 # if (cur_epoch <= 3) else (1 if (cur_epoch <= 10) else 0)  # 1 if (cur_epoch <= 2) else 0
        for begin in range(0, len(X_all), eq_each_batch):
            X, Y = X_all[begin : begin + eq_each_batch], Y_all[begin : begin + eq_each_batch]
            Y_num = strs2idxs(Y, sign2num)
            # Predict, abduce and train
            print("\nGetting all predicted label")
            predict_labels_str_list = get_eqs_predict(model, X, num2sign)
            print("Getting abduced label")
            res_list = abducer.abduce(predict_labels_str_list, max_address_num = 100, mode = mode, vocab = alphabet, require_more_address = require_more_address, level_matched = (level if require_level_matched else None)) #[(['0*1=0', '1*0=0', '1*1=1'], 1), (['2*2=4'], 1), ('3*3=9', 0), (['4*5=20'], 1)] #shuffle_str(alphabet)
            print("Generated abduction candidate list!")
            # Convert str to idx
            select_result = [strs2idxs(res[0], sign2num) for res in res_list]

            print("Selecting the best result according to similarity")
            if similar_coef >= 0:
                final_result = nn_select_batch_abduced_result(model, labeled_X, labeled_y, X, select_result, abduction_batch_size, Y_num, beam_width, similar_coef) 
            else:
                final_result = [res[random.randint(0,len(res)-1)] for res in select_result]
            
            batch_abduce_correct_sign_cnt, batch_sign_total = 0, 0
            for i in range(len(X)):
                cur_abduce_correct_cnt = sum(c1 == c2 for c1, c2 in zip(final_result[i], Y_num[i]))
                batch_abduce_correct_sign_cnt += cur_abduce_correct_cnt
                batch_sign_total += len(Y_num[i])
                if i % (eq_each_batch//5) == 0:
                    print("\nGround labels: ", Y_num[i])
                    print("Predict labels: ", predict_labels_str_list[i])
                    print("Abduced labels: ", select_result[i], alphabet)
                    print("Final best label: ", final_result[i])
            epoch_abduce_correct_sign_cnt += batch_abduce_correct_sign_cnt
            epoch_sign_total += batch_sign_total

            labels_list = final_result
            data_list = X
            if labeled_X is not None: # Use supervised data
                labels_list.append(labeled_y)
                data_list.append(labeled_X)
            images_np = np.array(list(itertools.chain.from_iterable(data_list)))
            labels_list = np.concatenate(labels_list, axis = 0)
            model.train_val(nn_fit_epoch, True, X=images_np, y=labels_list)

            print("Current epoch %d  Eq batch %d / %d  Abduce correct batch %.3f (%d / %d) Abduce correct epoch %.3f (%d / %d) "%(cur_epoch, \
            begin, len(X_all), batch_abduce_correct_sign_cnt/batch_sign_total, batch_abduce_correct_sign_cnt, batch_sign_total, epoch_abduce_correct_sign_cnt/epoch_sign_total, epoch_abduce_correct_sign_cnt, epoch_sign_total))
            image_test_loss, image_test_acc = model.train_val(1, False, data_loader=test_loader)
            logger.add_data(image_test_acc = image_test_acc, abduce_correct_rate = epoch_abduce_correct_sign_cnt/epoch_sign_total)
            logger.to_pk(run)
        predict_embeddings = model.predict(X=train_images_np, is_train=False)[1].cpu().numpy()
        logger.add_data(embeddings = predict_embeddings.copy(), embeddings_y = train_images_labels)
        logger.to_pk(run)


if __name__ == "__main__":
    # Parameters
    args = arg_init()
    print(args)

    train_model(args)