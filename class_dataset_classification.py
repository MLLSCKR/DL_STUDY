import os
import csv
from re import sub
from isort import file
from matplotlib import image
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from IPython.core.display import HTML
from sklearn.metrics import accuracy_score
from sympy import arg

from mathutil import *

# dataset class 선언
class Classifiaction_Dataset(object):
    
    def __init__(self, name, mode, resolution = [100, 100], input_shape = [-1]):
        self.name = name
        self.mode = mode
        
        file_path = "./flowers"
        
        self.target_names = self.list_dir(file_path)

        images = []
        idxs = []

        for dx, dname in enumerate(self.target_names):
            subpath = file_path + '/' + dname
            filenames = self.list_dir(subpath)

            for fname in filenames:
                if fname[-4:] != '.jpg':
                    # .jpg가 아닌 file들의 경우 무시
                    continue
                
                imagepath = os.path.join(subpath, fname)
                pixels = self.load_image_pixels(imagepath, resolution, input_shape)
                images.append(pixels)
                idxs.append(dx)
        
        self.image_shape = resolution + [3]

        self.xs = np.asarray(images, np.float32)
        self.ys = np.eye(len(self.target_names))[np.array(idxs).astype(int)]

        self.shuffle_dataset(0.6, 0.2, 0.2)


    def train_count(self):
        return len(self.tr_X)
    
    def shuffle_dataset(self, tr, te, val):
        self.dataset_indices = np.arange(self.xs.shape[0])
        np.random.shuffle(self.dataset_indices)
        
        dataset_len = self.xs.shape[0]

        tr_cnt = int(dataset_len * tr)
        te_cnt = int(dataset_len * te)
        val_cnt = int(dataset_len * val)
        
        self.tr_X = self.xs[self.dataset_indices[:tr_cnt]]
        self.tr_Y = self.ys[self.dataset_indices[:tr_cnt]]
        self.te_X = self.xs[self.dataset_indices[tr_cnt + 1:tr_cnt + te_cnt]]
        self.te_Y = self.ys[self.dataset_indices[tr_cnt + 1:tr_cnt + te_cnt]]
        self.val_X = self.xs[self.dataset_indices[tr_cnt + te_cnt + 1:tr_cnt + te_cnt + val_cnt]]
        self.val_Y = self.ys[self.dataset_indices[tr_cnt + te_cnt + 1:tr_cnt + te_cnt + val_cnt]]
    
    def shuffle_train_dataset(self, size):
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)
        
    def get_train_data(self, batch_size, iteration_num):
        # batch size, iteration_num을 이용하여 현재 학습에 이용할 batch group data를 return한다.
        return self.tr_X[batch_size * iteration_num + 1: batch_size * (iteration_num + 1)], \
            self.tr_Y[batch_size * iteration_num + 1: batch_size * (iteration_num + 1)]
            
    def get_validate_data(self):
        self.val_indices = np.arange(len(self.val_X))
        np.random.shuffle(self.val_indices)
        
        va_X = self.val_X
        va_Y = self.val_Y
        
        return va_X, va_Y
    
    def get_test_data(self):
        te_X = self.te_X
        te_Y = self.te_Y
        
        return te_X, te_Y
    
    def train_prt_result(self, current_epoch, costs, accs, acc, tm1, tm2):
        print('Epoch {} : cost = {:5.3f}, accuracy = {:5.3f}/{:5.3f} {}/{} secs'.format(current_epoch, np.mean(costs), np.mean(accs), acc, tm1, tm2))
    
    def test_prt_result(self, name, acc, tm1):
        print('Model {} test report : accuracy = {:5.3f}, {}secs'.format(name, acc, tm1))
    
    def forward_postproc(self, output, y):
        # select(multiple classification) dataset's postproc
        # LOSS : cross entropy
        
        entropy = softmax_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [output, y, entropy]

        # aux will be used for backpropagation
        return loss, aux
    
    def backprop_postproc(self, G_loss, aux):
        output, y, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy
        
        return G_output
        
    def eval_accuracy(self, x, y, output):
        estimate = np.argmax(output, axis = 1)
        answer = np.argmax(y, axis = 1)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)
        
        return accuracy
    
    def get_visualize_data(self, cnt):
        idx = np.random.choice(self.indices, cnt)
        X = self.xs[idx]
        Y = self.ys[idx]
        
        return X, Y
    
    def visualize(self, xs, estimates, answers):
        self.draw_images_horz(xs, self.image_shape)
        self.show_select_results(estimates, answers, self.target_names)

    def list_dir(self, path):
        filenames = os.listdir(path)
        filenames.sort()
        return filenames

    def load_image_pixels(self, imagepath, resolution, input_shape):
        img = Image.open(imagepath)
        resized = img.resize(resolution)
        
        return np.array(resized).reshape(input_shape)

    def draw_images_horz(self, xs, image_shape = None):
        show_cnt = len(xs)
        fig, axes = plt.subplots(1, show_cnt, figsize = (5, 5))
        for n in range(show_cnt):
            img = xs[n]

            if image_shape:
                x3d = img.reshape(image_shape)
                img = Image.fromarray(np.uint8(x3d))
            axes[n].imshow(img)
            axes[n].axis('off')

        plt.draw()
        plt.show()

    def show_select_results(self, est, ans, target_names, max_cnt = 0):
        for n in range(len(est)):
            pstr = vector_to_str(100*est[n], '%2.0f', max_cnt)
            estr = target_names[np.argmax(est[n])]
            astr = target_names[np.argmax(ans[n])]
            rstr = 'O'

            if estr != astr:
                rstr = 'X'
            
            print('estimated probability distribution {} -> estimate : {} : answer {} -> {}'.format(pstr, estr, astr, rstr))
