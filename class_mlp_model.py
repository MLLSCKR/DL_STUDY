"""
Date 220616.
    MLP 직접 구현해 보기 with using class. (by SCL)
        VER01(220616. only for regression)
Date 220703.
    BATCH NORMALIZATION 논문구현하기. (by SCL)
        VER02(220703. B.N)
"""

import os
import time
import numpy as np
from mathutil import *

# mlp class 선언
class mlp_model(object):
    
    ###
    # for ver02. add batch_nor parameter
    ###
    def __init__(self, name, hconfigs, learning_type, dataset, batch_nor = False):
        self.name = name
        self.learning_type = learning_type
        
        self.batch_nor = batch_nor

        self.dataset = dataset
        # dataset은 달리 선언한 class이다.
        # 위 class(dataset)은 tr_X, tr_Y, te_X, te_Y, val_X, val_Y 라는 class attribute를 지닌다.
        
        # use layers_para_generate function to produce weight matrixs and bias vectors
        #   to call layers_para_generate function, input tr_X, tr_Y[traning data input, output], hconfigs[list type]
        # 
        #       result of layers_param_generate function(self.pm)
        #       self.pm is dictionary type
        #           {'key[layer #]' : [weight_matrix, bias_matrix], ...}
        #     
        #     ver02. add batch_nor parameter to seperate functionality when user want to use batch normalization
        self.layers_param_generate(hconfigs, self.dataset.xs.shape[-1],\
                                   self.dataset.ys.shape[-1], self.batch_nor)
    
    # ver02. add batch normalization
    #     add batch_nor parameter at __Str__

    def __str__(self):
        print('model {} brief information. {}, training data : {}, test data :\
              {}, validation data : {}, batch normarlization : {}'.format(self.name, self.learning_type, \
            self.dataset.tr_X.shape[0], self.dataset.te_X.shape[0], \
            self.dataset.val_X.shape[0], self.batch_nor))
    

    # ver02. add batch normalization
    def __exec__(self, epoch_count = 10, batch_size = 10, learning_rate = 0.001, report = 0, cnt = 3):
        # for model training, use self.dataset.tr_X, self.dataset.tr_Y, self.pm
        """
        Deep learning 학습 절차
            training
                function call 순서
                    1. forward_neuralnet
                    2. forward_postproc
                    3. backprop_postproc
                    4. backprop_neuralnet
            학습에 필요한 각 함수의 세부적인 내용은 어떠한 학습인지에 따라 변하기에, class dataset(object)에서 별도로 정의하여 사용한다.
        """
        # ver02. add batch_nor
        self.mlp_train(epoch_count, batch_size, learning_rate, report)
        
        # for model testing, use self.dataset.te_X, self.dataset.tr_Y, self.pm
        self.mlp_test()
        
        # for model visualization(show results), use self.dataset.te_X, 
        # self.dataset.te_Y, self.pm
        self.visualization(cnt)

    # ver02. add batch_nor parameter to seperate layers param generate function's operations
    def layers_param_generate(self, hconfigs, input_shape, output_shape, batch_nor):
        self.hconfigs = hconfigs
        self.pm_hiddens = []
        
        for i in range(len(hconfigs) + 1):
            if i == 0:
                pre_cnt = input_shape
                aft_cnt = hconfigs[i]
                gamma = 1
                beta = 0
            elif i == len(hconfigs):
                pre_cnt = hconfigs[i - 1]
                aft_cnt = output_shape
                gamma = 0
                beta = 0
            else:
                pre_cnt = hconfigs[i - 1]
                aft_cnt = hconfigs[i]
                gamma = 1
                beta = 0
            
            weight = np.random.normal(0, 0.030, [pre_cnt, aft_cnt])
            bias = np.zeros(aft_cnt)

            if batch_nor == False:
                self.pm_hiddens.append({'w' : weight, 'b' : bias})  
            else:
                # when batch_nor == True
                # it saves random values[weight, bias, gamma, beta] at self.pm_hiddens
                self.pm_hiddens.append({'w' : weight, 'b' : bias, 'g' : gamma, 'beta' : beta})

    def mlp_train(self, epoch_count = 10, batch_size = 10, \
                  learning_rate = 0.001, report = 0):
        self.learning_rate = learning_rate
        
        # random shuffle 후, trainin, test, validation set을 해당 비중으로 조정
        # self.tr_X, tr_Y, val_X, val_Y, te_X, te_Y에 저장됨
        self.dataset.shuffle_dataset(0.6, 0.2, 0.2)
        
        batch_count = int(self.dataset.tr_X.shape[0] / batch_size)
        
        print("model {} traiing is started".format(self.name))
        
        time_start, time_temp  = time.time(), time.time()
        
        for i in range(epoch_count):
            costs = []
            accs = []
            
            # only treat batch_size * batch_count number of data
            #   to avoid not divided error
            #
            #   dataset.shuffle_dataset function
            #       작동 원리 요약
            #       self.indices = np.arange(batch_size * batch_count)
            #       np.random.shuffle(self.indices)
            #           무작위로 섞인 index들의 정보가 self.dataset.indices에 저장되어 있음
            self.dataset.shuffle_train_dataset(batch_size * batch_count)
            
            # training
            for j in range(batch_count):
                trX, trY = self.dataset.get_train_data(batch_size, j)
                
                # forward propagation
                # ver02. add batch_normalization
                output, aux_nn = self.forward_neuralnet(trX)
                # output = X * W
                # aux_nn = X
                loss, aux_pp = self.forward_postproc(output, trY)
                # loss = cost function
                # aux_pp = 
                
                accuracy = self.eval_accuracy(trX, trY, output)
                
                # backward propagation
                G_loss=  1.0
                G_output = self.backprop_postproc(G_loss, aux_pp)
                self.backprop_neuralnet(G_output, aux_nn)
                
                costs.append(loss)
                accs.append(accuracy)
            
            # validation
            if report > 0 and (i + 1) % report == 0:
                vaX, vaY = self.dataset.get_validate_data()
                acc = self.eval_accuracy(vaX, vaY)
                time_mid = time.time()
                
                self.dataset.train_prt_result(i + 1, costs, accs, acc, time_mid - time_temp, time_mid - time_start)
                
                time_temp = time_mid
                
        time_end = time.time()
        print('Model {} train ended in {} secs'.format(self.name, time_end - time_start))
        
    def mlp_test(self):
       
        print("model {} test is started".format(self.name))
        
        start_time = time.time()
        
        teX, teY = self.dataset.get_test_data()
        
        # forward propagation
        output, aux_nn = self.forward_neuralnet(teX)
        accuracy = self.eval_accuracy(teX, teY, output)
        
        end_time = time.time()
        
        self.dataset.test_prt_result(self.name, accuracy, end_time - start_time)
        
    
    def eval_accuracy(self, x, y, output = None):
        if output is None:
            output, _ = self.forward_neuralnet(x)
            
        accuracy = self.dataset.eval_accuracy(x, y, output)
            
        return accuracy

    def forward_neuralnet(self, x):
        aux_nn = []
        temp_x = x

        # batch normalization
        if self.batch_nor == False:
            for n, pm in enumerate(self.pm_hiddens):
                
                temp_y = np.matmul(temp_x, pm['w']) + pm['b']
                
                if n != (len(self.pm_hiddens) - 1):
                    output = relu(temp_y)
                    aux_nn.append([temp_x, output])
                    temp_x = output
                else:
                    output = temp_y
                    aux_nn.append([temp_x, output])
        else:
            for n, pm in enumerate(self.pm_hiddens):
                # print(temp_x.shape, pm['w'].shape, pm['b'].shape)
                temp_x2 = np.matmul(temp_x, pm['w']) + pm['b']

                temp_batch_mean = batch_mean(temp_x2)
                temp_batch_std = batch_std(temp_x2)

                epsilon = np.ones(shape = (temp_batch_std.shape[0], 1)) * 1e-5

                temp_y = pm['g'] * np.divide((np.subtract(temp_x2, temp_batch_mean)), (temp_batch_std + epsilon)) + pm['beta']

                if n != (len(self.pm_hiddens) - 1):
                    # output은 Relu 비선형 함수의 결과이자, 그 다음 layer에 들어가는 input이다.
                    output = relu(temp_y)
                    aux_nn.append([temp_x, temp_x2, temp_y, output, temp_batch_mean, temp_batch_std])
                    temp_x = output
                else:
                    output = temp_x2
                    aux_nn.append([temp_x, temp_x2, temp_y, output, temp_batch_mean, temp_batch_std])
                    temp_x = output

        return output, aux_nn

    # by using output(calculated from forward_neuralnet), estimate result
    def forward_postproc(self, output, trY):
        loss, aux_pp = self.dataset.forward_postproc(output, trY)
        
        return loss, aux_pp
    
    # backprop postproc -> dL / dY, backprop postproc -> dL / dY * dY / dW
    def backprop_postproc(self, G_loss, aux):
        # aux : diff(output - y)
        G_output = self.dataset.backprop_postproc(G_loss, aux)
        
        return G_output
    
    
    def backprop_neuralnet(self, G_output, aux_nn):
        # aux_nn = [X_1st, X_2nd, ...]
        first = 1
        
        if self.batch_nor == False:
            for n in reversed(range(len(self.pm_hiddens))):
                if first == 1:
                    G_y = G_output
                    x, y = aux_nn[n]
                    
                    first = 0
                else:
                    x, y = aux_nn[n]
                    G_y = derv_relu(y) * G_y
                
                g_y_weight = x.transpose()
                g_y_input = self.pm_hiddens[n]['w'].transpose()
                
                G_weight = np.matmul(g_y_weight, G_y)
                G_bias = np.sum(G_y, axis = 0)
                G_input = np.matmul(G_y, g_y_input)
                
                # updating weight
                self.pm_hiddens[n]['w'] -= self.learning_rate * G_weight
                
                # updating bias
                self.pm_hiddens[n]['b'] -= self.learning_rate * G_bias
                
                G_y = G_input
        else:
            for n in reversed(range(len(self.pm_hiddens))):
                if first == 1:
                    # at first(마지막)의 경우 relu function, batch nor 이 적용되지 않기에 분리한다.
                    G_y = G_output
                    x1, x2, y_batch, y, mean, std = aux_nn[n]
                    
                    first = 0
                    g_y_weight = x1.transpose()
                    g_y_input = self.pm_hiddens[n]['w'].transpose()
                    
                    G_weight = np.matmul(g_y_weight, G_y)
                    G_bias = np.sum(G_y, axis = 0)
                    G_input = np.matmul(G_y, g_y_input)
                    
                    # updating weight
                    self.pm_hiddens[n]['w'] -= self.learning_rate * G_weight
                    
                    # updating bias
                    self.pm_hiddens[n]['b'] -= self.learning_rate * G_bias
                    
                    G_y = G_input
                else:
                    x1, x2, y_batch, y, mean, std = aux_nn[n]
                    G_y = derv_relu(y) * G_y
                    # G_y는 deltaL / deltaY 이다.
                    
                    G_y = self.batch_nor_backprop(G_y, x1, x2, y_batch, mean, std, n)
                    
                    g_y_weight = x1.transpose()
                    g_y_input = self.pm_hiddens[n]['w'].transpose()
                    
                    G_weight = np.matmul(g_y_weight, G_y)
                    G_bias = np.sum(G_y, axis = 0)
                    G_input = np.matmul(G_y, g_y_input)
                    
                    # updating weight
                    self.pm_hiddens[n]['w'] -= self.learning_rate * G_weight
                    
                    # updating bias
                    self.pm_hiddens[n]['b'] -= self.learning_rate * G_bias
                    
                    G_y = G_input

    
    def visualization(self, cnt):
        print('Model {} visualization'.format(self.name))
        
        deX, deY = self.dataset.get_visualize_data(cnt)
        est = self.get_estimate(deX)
        self.dataset.visualize(deX, deY, est)
        
    def get_estimate(self, X):
        output, _ = self.forward_neuralnet(X)
        
        return output

    def batch_nor_backprop(self, G_output, x1, x2, y_batch, mean, std, n):
        epsilon = np.ones(shape = (std.shape[0], 1)) * 1e-5
        
        # x1 : fully nn input
        # x2 : fully nn output, batch norm input
        # y_batch : batch norm output

        G_y_batch = self.pm_hiddens[n]['g'] * G_output
        G_std = np.sum(np.multiply(G_y_batch, np.multiply(np.add(x2, mean * -1), -1/2 * np.power(np.add(np.square(std), epsilon), -1.5))), axis = 0)
        G_mean = np.sum(np.multiply(G_y_batch, -1 * np.power(np.add(np.square(std), epsilon), -0.5)), axis = 0) + G_std * (-2) * (np.sum(np.add(x2, -1 * mean), axis = 0)) / mean.shape[0]
        G_x2 = np.multiply(G_y_batch, np.power(np.add(np.square(std), epsilon), -0.5)) + G_std * 2 * (np.subtract(x2, mean)) / mean.shape[0] + G_mean / mean.shape[0]
        
        G_gamma = np.sum(np.multiply(G_output, y_batch), axis = 0)
        G_beta = np.sum(G_output, axis = 0)

        # updating gamma
        self.pm_hiddens[n]['g'] -= self.learning_rate * G_gamma

        # updating beta            
        self.pm_hiddens[n]['beta'] -= self.learning_rate * G_beta

        G_y = G_x2

        return G_y