# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:13:37 2019

@author: hwdon
"""

import numpy as np
import math
def calculate_fan_in_and_fan_out(tensor):
    if len(tensor.shape) < 2:
        raise ValueError("tensor with fewer than 2 dimensions")
    if len(tensor.shape) ==2:
        fan_in,fan_out = tensor.shape
    else: #F,C,kH,kW
        num_input_fmaps = tensor.shape[1]  #size(1)  F,C,H,W
        num_output_fmaps = tensor.shape[0]  #size(0)
        receptive_field_size = tensor[0][0].size
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size   

    return fan_in, fan_out



def xavier_uniform(tensor, gain=1.):    
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std  
    tensor[:] = np.random.uniform(-bound,bound,(tensor.shape))

def xavier_normal(tensor, gain=1.):
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    tensor[:] = np.random.normal(0,std,(tensor.shape))

# copy from Pytorch
def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================
    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function
    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

        
def kaiming_uniform(tensor,a=0,mode = 'fan_in', nonlinearity='leaky_relu'):
    fan_in,fan_out = calculate_fan_in_and_fan_out(tensor)
    if mode=='fan_in':       fan = fan_in
    else: fan = fan_out
    
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std      
    tensor[:] = np.random.uniform(-bound,bound,(tensor.shape))

def kaiming_normal(tensor,a=0,mode = 'fan_in', nonlinearity='leaky_relu'):
    fan_in,fan_out = calculate_fan_in_and_fan_out(tensor)
    if mode=='fan_in':     fan = fan_in
    else: fan = fan_out
    
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    
    tensor[:] = np.random.normal(0,std,(tensor.shape))


def kaiming(tensor,method_params=None):
    method_type,a,mode,nonlinearity='uniform',0,'fan_in','leaky_relu'
    if method_params:
        method_type = method_params.get('type', "uniform")
        a =  method_params.get('a', 0)
        mode = method_params.get('mode','fan_in' )
        nonlinearity = method_params.get('nonlinearity', 'leaky_relu')
    if method_params=="uniform":
        kaiming_uniform(tensor,a,mode,nonlinearity)
    else:
        kaiming_normal(tensor,a,mode,nonlinearity)
        
