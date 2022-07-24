from class_mlp_model import *
import mathutil

class AdamModel(mlp_model):
    def __init__(self, name, hconfigs, learning_type, dataset, batch_nor = False ):
        self.use_adam = False
        super(AdamModel, self).__init__(name, hconfigs, learning_type, dataset, batch_nor = False)

    def update_param(self, pm, key, delta):
        if self.use_adam:
            delta = self.eval_adam_delta(pm, key, delta)

        pm[key] -= self.learning_rate * delta
    
    def eval_adam_delta(self, pm, key, delta):
        ro_1 = 0.9
        ro_2 = 0.999
        epsilon = 1.0e-8

        skey, tkey, step = 's' + key, 't' + key, 'n' + key
        if skey not in pm:
            pm[skey] = np.zeros(pm[key].shape)
            pm[tkey] = np.zeros(pm[key].shape)
            pm[step] = 0
        
        s = pm[skey] = ro_1 * pm[skey] + (1 - ro_1) * delta
        t = pm[tkey] = ro_2 * pm[tkey] + (1 - ro_2) * (delta * delta)

        pm[step] += 1
        s = s/(1 - np.power(ro_1, pm[step]))
        t = t/(1 - np.power(ro_2, pm[step]))

        return s / (np.sqrt(t) + epsilon)

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
                
                # Adam Algorithm!!!
                # updating weight의 방식을 수정함!!
                self.update_param(self.pm_hiddens[n], 'w', G_weight)
                
                # Adam Algorithm!!!
                # updating bias의 방식을 수정함!!
                self.update_param(self.pm_hiddens[n], 'b', G_bias)
                
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
                    
                    # Adam Algorithm!!!
                    # updating weight의 방식을 수정함!!
                    self.update_param(self.pm_hiddens[n], 'w', G_weight)
                    
                    # Adam Algorithm!!!
                    # updating bias의 방식을 수정함!!
                    self.update_param(self.pm_hiddens[n], 'b', G_bias)
                    
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
                    
                    # Adam Algorithm!!!
                    # updating weight의 방식을 수정함!!
                    self.update_param(self.pm_hiddens[n], 'w', G_weight)
                    
                    # Adam Algorithm!!!
                    # updating bias의 방식을 수정함!!
                    self.update_param(self.pm_hiddens[n], 'b', G_bias)
                    
                    G_y = G_input