import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import numpy as np
from sklearn.manifold import TSNE
import os

def get_data_from_file(pk_list, name, used_iter_num = 30):
    data = []
    for pk_file in pk_list:
        with open(pk_file, 'rb') as f:
            one_result = pickle.load(f)[name]
            print(len(one_result))
            data.append(one_result[:used_iter_num])
    data = np.array(data)
    return data

def get_mean(data):
    mean = np.mean(data, axis=0)
    return mean

def get_std(data):
    std = np.std(data, axis=0)
    return std

def plot(y_mean_list, y_std_list, label_list, xlabel = "Epoch", ylabel = "Test CNN Accuracy", save_name = None):
    plt.cla()
    linewidth = 1.0
    markersize = 4.0
    std_alpha = 0.15
    color_list = ['#DB3340', '#1A6396', '#59DD97', 'gold', 'black']
    x = list(range(0, len(y_mean_list[0])))

    for i in range(len(y_mean_list)):
        color = color_list[i%len(color_list)]
        plt.plot(x, y_mean_list[i], linewidth=linewidth, markersize=markersize, color=color, label=label_list[i])
        plt.fill_between(x, y_mean_list[i]-y_std_list[i], y_mean_list[i]+y_std_list[i], color=color, alpha=std_alpha)

    plt.xlabel(xlabel)#Abscissa name
    plt.ylabel(ylabel)#Ordinate name
    #plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #plt.grid()
    plt.legend(loc = "lower right")
    if save_name is not None:
        plt.savefig("results/%s.pdf"%(save_name))
    plt.show()

def plot_tsne(X, y = None, save_name = None):
    X_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(X)
    vis_x = X_embedded[:, 0]
    vis_y = X_embedded[:, 1]
    plt.cla()
    plt.scatter(vis_x, vis_y, c=y, s=1, cmap=plt.cm.get_cmap("jet", 10))
    if save_name is not None:
        plt.savefig("results/%s.pdf"%(save_name))
    plt.show()

class Logger:
    def __init__(self, dataset, images_type, model_path, num_pretrain_exs, batch_each_length,eq_each_batch,abduction_batch_size,beam_width,similar_coef):
        self.image_test_acc_list = []
        self.equation_train_acc_list = []
        self.abduce_correct_rate_list = []
        self.train_embeddings_list = []
        self.train_embeddings_y_list = []

        self.dataset = dataset
        self.images_type = images_type
        self.load_ssl_model = model_path is not None #load self supervised model
        self.num_pretrain_exs = num_pretrain_exs
        self.batch_each_length = batch_each_length
        self.eq_each_batch = eq_each_batch
        self.abduction_batch_size = abduction_batch_size
        self.beam_width = beam_width
        self.similar_coef = similar_coef
        self.folder = "./pickle"
    
    def add_data(self, image_test_acc = None, equation_train_acc = None, abduce_correct_rate = None, embeddings = None, embeddings_y = None):
        if image_test_acc is not None:
            self.image_test_acc_list.append(image_test_acc)
        if equation_train_acc is not None:
            self.equation_train_acc_list.append(equation_train_acc)
        if abduce_correct_rate is not None:
            self.abduce_correct_rate_list.append(abduce_correct_rate)
        if embeddings is not None:
            self.train_embeddings_list.append(embeddings)
        if embeddings_y is not None:
            self.train_embeddings_y_list.append(embeddings_y)

    def to_pk(self, run = 0):
        if os.path.exists(self.folder)==False:
            os.makedirs(self.folder)
        params = "%s-%s-%s-%d-%d-%d-%d-%s-%.2f-run%d"%(self.dataset, self.images_type, self.load_ssl_model, self.num_pretrain_exs, self.batch_each_length, self.eq_each_batch, self.abduction_batch_size, self.beam_width, self.similar_coef, run)
        filename = '%s/result-%s.pickle'%(self.folder,params)
        results = {
            "image_test_acc_list": self.image_test_acc_list,
            "equation_train_acc_list": self.equation_train_acc_list,
            "abduce_correct_rate_list": self.abduce_correct_rate_list,
            "train_embeddings_list": self.train_embeddings_list,
            "train_embeddings_y_list": self.train_embeddings_y_list
        }
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print("Dump the results to ", filename)

if __name__ == "__main__":
    pass