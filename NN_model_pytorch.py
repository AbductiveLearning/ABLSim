import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet50
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm
import numpy as np
from PIL import Image
import itertools
import time

train_transform = transforms.Compose([
transforms.RandomResizedCrop(32),
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
transforms.RandomGrayscale(p=0.2),
transforms.ToTensor(),
transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

Lenet_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
SymbolNet_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1,))])

class ResNet50(nn.Module):
    def __init__(self, num_class, pretrained_path, loss_criterion, batch_size=512, freeze_feature=True, num_workers=16):
        super(ResNet50, self).__init__()
        # encoder
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        # classifier
        self.fc = nn.Linear(2048, 10, bias=True)
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.fc = nn.Linear(2048, num_class, bias=True) # Initialize new weight
        self.loss_criterion = loss_criterion
        self.batch_size = batch_size
        self.optimizer = None
        self.set_freeze_feature(freeze_feature)
        self.num_workers = num_workers
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out, F.normalize(feature, dim=-1)

    def set_freeze_feature(self, freeze):
        for param in self.f.parameters():
            param.requires_grad = not freeze
        parameters = self.fc.parameters() if freeze else self.parameters()
        self.optimizer = optim.Adam(parameters, lr=1e-3, weight_decay=1e-6)

    # train or test for several epochs
    def train_val(self, epochs, is_train, data_loader=None, X=None, y=None):
        if data_loader is None:
            dataset = TorchDataset(X, y, train_transform if is_train else test_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        if is_train:
            self.train() # for BN
        else:
            self.eval()
            epochs = 1

        #for name, parameters in self.named_parameters():
        #    if name == "f.1.weight" or name == "f.6.2.bn3.bias" or name == "f.6.0.bn1.weight":
        #        print(name, ':', parameters.size(), parameters)
        for epoch in range(1, epochs+1):
            total_loss, total_correct_1, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
            with (torch.enable_grad() if is_train else torch.no_grad()):
                for data, target in data_bar:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    out, feature = self(data)
                    loss = self.loss_criterion(out, target)

                    if is_train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    
                    total_num += data.size(0)
                    total_loss += loss.item() * data.size(0)
                    prediction = torch.argsort(out, dim=-1, descending=True)
                    total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                    data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                            .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num, total_correct_1 / total_num * 100))

        return total_loss / total_num, total_correct_1 / total_num * 100

    def predict(self, data_loader=None, X=None, is_train=False):
        if data_loader is None:
            y = np.zeros(X.shape[0], dtype='long')
            dataset = TorchDataset(X, y, train_transform if is_train else test_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.eval()
        results_out, results_feature = [], []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda(non_blocking=True)
                out, feature = self(data)
                results_out.append(out), results_feature.append(feature)
        return torch.softmax(torch.cat(results_out, axis=0), axis=1), torch.cat(results_feature, axis=0)
    
    def predict_classes(self, data_loader=None, X=None, is_train=False):
        return self.predict(data_loader, X, is_train)[0].argmax(axis=1)



class LeNet(nn.Module):
    def __init__(self, num_class, loss_criterion, pretrained_path=None, batch_size=512, num_workers=16):
        super(LeNet, self).__init__()
        # encoder
        self.f1 = [
            nn.Conv2d(1,  6, 5),
            nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        ]
        self.f2 = [
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84)
        ]
        self.f3 = [
            nn.ReLU(),
            nn.Linear(84, num_class)
        ]
        self.f1 = nn.Sequential(*self.f1)
        self.f2 = nn.Sequential(*self.f2)
        self.f3 = nn.Sequential(*self.f3)
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.loss_criterion = loss_criterion
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.num_workers = num_workers
    def forward(self, x):
        x = self.f1(x)
        x = x.view(-1, 16 * 4 * 4)
        feature = self.f2(x)
        out = self.f3(feature)
        return out, F.normalize(feature, dim=-1)

    # train or test for several epochs
    def train_val(self, epochs, is_train, data_loader=None, X=None, y=None):
        if data_loader is None:
            dataset = TorchDataset(X, y, Lenet_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        if is_train:
            self.train() # for BN
        else:
            self.eval()
            epochs = 1

        for epoch in range(1, epochs+1):
            total_loss, total_correct_1, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
            with (torch.enable_grad() if is_train else torch.no_grad()):
                for data, target in data_bar:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    out, feature = self(data)
                    loss = self.loss_criterion(out, target)

                    if is_train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    total_num += data.size(0)
                    total_loss += loss.item() * data.size(0)
                    prediction = torch.argsort(out, dim=-1, descending=True)
                    total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                    data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                            .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num, total_correct_1 / total_num * 100))

        return total_loss / total_num, total_correct_1 / total_num * 100

    def predict(self, data_loader=None, X=None, is_train=False):
        if data_loader is None:
            y = np.zeros(X.shape[0], dtype='long')
            dataset = TorchDataset(X, y, Lenet_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.eval()
        results_out, results_feature = [], []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda(non_blocking=True)
                out, feature = self(data)
                results_out.append(out), results_feature.append(feature)
        return torch.softmax(torch.cat(results_out, axis=0), axis=1), torch.cat(results_feature, axis=0)
    
    def predict_classes(self, data_loader=None, X=None, is_train=False):
        return self.predict(data_loader, X, is_train)[0].argmax(axis=1)


class SymbolNet(nn.Module):
    def __init__(self, num_class, loss_criterion, pretrained_path=None, batch_size=512, num_workers=16):
        super(SymbolNet, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(30976, 128)
        self.fc2 = nn.Linear(128, num_class)

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.loss_criterion = loss_criterion
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.num_workers = num_workers
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        feature = self.fc1(x)
        x = F.relu(feature)
        x = self.dropout2(x)
        out = self.fc2(x)

        return out, F.normalize(feature, dim=-1)

    # train or test for several epochs
    def train_val(self, epochs, is_train, data_loader=None, X=None, y=None):
        if data_loader is None:
            dataset = TorchDataset(X, y, SymbolNet_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        if is_train:
            self.train() # for BN
        else:
            self.eval()
            epochs = 1

        for epoch in range(1, epochs+1):
            total_loss, total_correct_1, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
            with (torch.enable_grad() if is_train else torch.no_grad()):
                for data, target in data_bar:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    out, feature = self(data)
                    loss = self.loss_criterion(out, target)

                    if is_train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    total_num += data.size(0)
                    total_loss += loss.item() * data.size(0)
                    prediction = torch.argsort(out, dim=-1, descending=True)
                    total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

                    data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                            .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num, total_correct_1 / total_num * 100))

        return total_loss / total_num, total_correct_1 / total_num * 100

    def predict(self, data_loader=None, X=None, is_train=False):
        if data_loader is None:
            y = np.zeros(X.shape[0], dtype='long')
            dataset = TorchDataset(X, y, SymbolNet_transform)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.eval()
        results_out, results_feature = [], []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda(non_blocking=True)
                out, feature = self(data)
                results_out.append(out), results_feature.append(feature)
        return torch.softmax(torch.cat(results_out, axis=0), axis=1), torch.cat(results_feature, axis=0)
    
    def predict_classes(self, data_loader=None, X=None, is_train=False):
        return self.predict(data_loader, X, is_train)[0].argmax(axis=1)


def get_eqs_predict(model, X, num2sign):
    images_np = np.array(list(itertools.chain.from_iterable(X)))
    predict_labels_list = model.predict_classes(X=images_np, is_train=False).cpu().numpy()
    predict_labels_str_list, cur_idx = [], 0
    for eq in X:
        predict_eq_labels_list = predict_labels_list[cur_idx : cur_idx + len(eq)]
        predict_eq_labels_list = [num2sign[str(label)] for label in predict_eq_labels_list]
        predict_labels_str = "".join(predict_eq_labels_list)
        predict_labels_str_list.append(predict_labels_str)
        cur_idx += len(eq)
    assert (cur_idx == len(predict_labels_list))
    return predict_labels_str_list


class TorchDataset(Dataset):
    def __init__(self, images_np, label, transform=None):
        self.data = images_np
        self.label = torch.LongTensor(label.flatten())
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    model_path = 'results/128_0.5_200_512_500_model.pth'
    batch_size, epochs = 128, 50
    
    '''
    import pickle
    fr = open("cifar10.pk", "rb")
    cifar10 = pickle.load(fr)
    (X_train, y_train), (X_test, y_test) = cifar10
    
    train_data = TorchDataset(X_train, y_train, train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = TorchDataset(X_test, y_test, test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    train_data = CIFAR10(root='data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    model = ResNet50(num_class=13, pretrained_path=model_path, loss_criterion=nn.CrossEntropyLoss(), batch_size=batch_size, freeze_feature=False).cuda()
    '''


    train_data = MNIST(root='data', train=True, download=True,transform=Lenet_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_data = MNIST(root='data', train=False, download=True,transform=Lenet_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = LeNet(num_class=10, loss_criterion=nn.CrossEntropyLoss(), batch_size=batch_size).cuda()
    #model = SymbolNet(num_class=10, loss_criterion=nn.CrossEntropyLoss(), batch_size=batch_size).cuda()
    
    
    '''
    train_data = TorchDataset(X_train, np.zeros(X_train.shape[0]), train_transform)
    print(train_data[0])
    y = model(train_data)
    print(y)
    '''

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1 = model.train_val(4, True, data_loader=train_loader)
        test_loss, test_acc_1 = model.train_val(1, False, data_loader=test_loader)
        # train_loss, train_acc_1 = model.train_val(3, True, X=X_train, y=y_train)
        # test_loss, test_acc_1 = model.train_val(1, False, X=X_test, y=y_test)
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/linear_model.pth')
        #y_pred, feature = model.predict(X=X_test)
