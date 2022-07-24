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
class Adam_Dataset(object):
    
    def __init__(self, name, mode, resolution = [100, 100], input_shape = [-1]):
        self.name = name
        self.mode = mode
        
        file_path = "./Office-31"
        
        domain_names = self.list_dir(file_path)

        images = []
        didxs, oidxs = [], []

        for dx, dname in enumerate(domain_names):
            domainpath = os.path.join(file_path, dname)
            object_names = self.list_dir(domainpath)

            for ox, oname in enumerate(object_names):
                objectpath = os.path.join(domainpath, oname)
                filenames = self.list_dir(objectpath)
                for fname in filenames:
                    if fname[-4:] != '.jpg':
                        continue
                    imagepath = os.path.join(objectpath, fname)
                    pixels = self.load_image_pixels(imagepath, resolution, input_shape)
                    images.append(pixels)
                    didxs.append(dx)
                    oidxs.append(ox)
        
        self.image_shape = resolution + [3]

        self.xs = np.asarray(images, np.float32)
        
        ys0 = self.onehot(didxs, len(domain_names))
        ys1 = self.onehot(oidxs, len(object_names))
        self.ys = np.hstack([ys0, ys1])

        self.shuffle_dataset(0.6, 0.2, 0.2)
        self.target_names = [domain_names, object_names]
        self.cnts = [len(domain_names)]

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
        acc_pair = np.mean(accs, axis = 0)
        
        print('Epoch {} : cost = {:5.3f}, accuracy = {:5.3f} + {:5.3f}/{:5.3f} + {:5.3f} ({}/{} secs)'.format(current_epoch, np.mean(costs), acc_pair[0], acc_pair[1], acc[0], acc[1], tm1, tm2))
    
    def test_prt_result(self, name, acc, tm1):
        print('Model {} test report : accuracy = {:5.3f} + {:5.3f}, ({}secs)\n'.format(name, acc[0], acc[1], tm1))
    
    def forward_postproc(self, output, y):
                
        output, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)

        # loss0, aux0
        entropy0 = softmax_cross_entropy_with_logits(ys[0], output[0])
        loss0 = np.mean(entropy0)
        aux0 = [output[0], ys[0], entropy0]

        # loss1, aux1
        entropy1 = softmax_cross_entropy_with_logits(ys[1], output[1])
        loss1 = np.mean(entropy1)
        aux1 = [output[1], ys[1], entropy1]

        # aux will be used for backpropagation
        return loss0 + loss1, [aux0, aux1]

    def backprop_postproc(self, G_loss, aux):
        aux0, aux1 = aux

        # G_output0
        output0, y0, entropy0 = aux0

        g_loss_entropy0 = 1.0 / np.prod(entropy0.shape)
        g_entropy_output0 = softmax_cross_entropy_with_logits_derv(y0, output0)

        G_entropy0 = g_loss_entropy0 * G_loss
        G_output0 = g_entropy_output0 * G_entropy0

        # G_output1
        output1, y1, entropy1 = aux1

        g_loss_entropy1 = 1.0 / np.prod(entropy1.shape)
        g_entropy_output1 = softmax_cross_entropy_with_logits_derv(y1, output1)

        G_entropy1 = g_loss_entropy1 * G_loss
        G_output1 = g_entropy_output1 * G_entropy1

        return np.hstack([G_output0, G_output1])
        
    def base_eval_accuracy(self, x, y, output):
        estimate = np.argmax(output, axis = 1)
        answer = np.argmax(y, axis = 1)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)
        
        return accuracy

    def eval_accuracy(self, x, y, output):
        outputs, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)

        acc0 = self.base_eval_accuracy(x, ys[0], outputs[0])
        acc1 = self.base_eval_accuracy(x, ys[1], outputs[1])

        return [acc0, acc1]
        
    def get_visualize_data(self, cnt):
        idx = np.random.choice(self.indices, cnt)
        X = self.xs[idx]
        Y = self.ys[idx]
        
        return X, Y
    
    def visualize(self, xs, estimates, answers):
        self.draw_images_horz(xs, self.image_shape)
        ests, anss = np.hsplit(estimates, self.cnts), np.hsplit(answers, self.cnts)

        captions = ['domain', 'product']

        for m in range(2):
            print('[ {} estimated result]'.format(captions[m]))
            self.show_select_results(ests[m], anss[m], self.target_names[m], 8)

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

    def onehot(self, xs, cnt):
        return np.eye(cnt)[np.array(xs).astype(int)]