# mathutil

import pandas as pd
import numpy as np
import os

def relu(x):
    return np.maximum(x, 0)

def derv_relu(x):
    return np.sign(x)

def batch_mean(x):
    return np.mean(x, axis = 1).reshape(x.shape[0], 1)

def batch_std(x):
    return np.std(x, axis = 1).reshape(x.shape[0], 1)

def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))

def sigmoid_derv(y):
    return y * (1-y)

def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)

def tanh(x):
    return 2 * sigmoid(2*x) - 1

def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)

def softmax(x):
    max_elem = np.max(x, axis = 1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis = 1)
    probs = (exp.transpose() / sum_exp).transpose()
    
    return probs

def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs + 1.0e-10), axis = 1)

def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels

def vector_to_str(x, fmt = '%.2f', max_cnt = 0):
    if max_cnt == 0 or len(x) <= max_cnt:
        return '[' + ','.join([fmt]*len(x)) % tuple(x) + ']'

    v = x[0:max_cnt]

    return '[' + ','.join([fmt]*len(v)) % tuple(v) + ',...]'