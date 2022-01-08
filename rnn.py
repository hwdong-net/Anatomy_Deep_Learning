# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:18:34 2019

@author: dhw
"""

#参考：https://pytorch.org/docs/stable/nn.html#lstmcell
import numpy as np
import math 
class RNNCellBase(object):
    __constants__ = ['input_size', 'hidden_size']
    def __init__(self, input_size, hidden_size,bias, num_chunks):
        super(RNNCellBase, self).__init__()        
        self.input_size, self.hidden_size = input_size, hidden_size
        self.bias = bias
        self.W_ih= np.empty((input_size, num_chunks*hidden_size))   # input to hidden
        self.W_hh = np.empty((hidden_size, num_chunks*hidden_size))  # hidden to hidden
        if bias:
            self.b_ih = np.zeros((1,num_chunks*hidden_size))
            self.b_hh = np.zeros((1,num_chunks*hidden_size))
        else:
            self.b_ih = None
            self.b_hh = None

        #Wf = np.empty(hidden_size, output_size)        # hidden to output        
        #bf = np.zeros((1,output_size)) # output bias
        
        self.params = [self.W_ih,self.W_hh,self.b_ih,self.b_hh]
        self.grads = [np.zeros_like(param)for param in self.params]
        self.param_grads = self.params.copy()
        self.param_grads.extend(self.grads)
        
        self.reset_parameters()
      
    def parameters(self,no_grad = True):
        if no_grad:   return self.params;  
        return self.param_grads;            
            
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.params:
            w = param
            w[:] = np.random.uniform(-stdv, stdv,(w.shape))
            
    def check_forward_input(self, input):
        if input.shape[1] != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.shape[1], self.input_size))

    def check_forward_hidden(self, input, h, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.shape[0] != h.shape[0]:
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.shape[0], hidden_label, h.shape[0]))

        if h.shape[1] != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, h.shape[1], self.hidden_size))
    def init_hidden(batch_size):
        return np.zeros(input.shape[0], self.hidden_size, dtype=input.dtype)          
            
def relu(x):
    return x * (x > 0)

def rnn_tanh_cell(x, h,W_ih, W_hh,b_ih, b_hh):
    #h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    if b_ih is None:        
        return np.tanh(np.dot(x,W_ih) +  np.dot(h,W_hh))
    else:
        return np.tanh(np.dot(x,W_ih) + b_ih  +  np.dot(h,W_hh) + b_hh)
   
    
def rnn_relu_cell(x, h,W_ih,W_hh,b_ih, b_hh):
    #h' = \relu(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    if b_ih is None:
        return relu(np.dot(x,W_ih) +  np.dot(h,W_hh) )
    else:
        return relu(np.dot(x,W_ih) + b_ih  +  np.dot(h,W_hh) + b_hh)       
    
class RNNCell(RNNCellBase):
    """        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})"""
    
    __constants__ = ['input_size', 'hidden_size',  'nonlinearity']

    def __init__(self, input_size, hidden_size,bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__(input_size, hidden_size,bias,num_chunks=1)
        self.nonlinearity = nonlinearity
        
    def forward(self, input, h=None): 
        self.check_forward_input(input)
        if h is None:
            h = np.zeros(input.shape[0], self.hidden_size, dtype=input.dtype)
        self.check_forward_hidden(input, h, '')
        if self.nonlinearity == "tanh":
            ret = rnn_tanh_cell(
                input, h,
                self.W_ih, self.W_hh,
                self.b_ih, self.b_hh,
            )
        elif self.nonlinearity == "relu":
            ret = rnn_relu_cell(
                input, h,
                self.W_ih, self.W_hh,
                self.b_ih, self.b_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret
    def __call__(self, input, h=None): 
        return self.forward(input,h)
    
    def backward(self,dh,H,X,H_pre):
        if self.nonlinearity == "tanh":
            dZh = (1 - H * H) * dh # backprop through tanh nonlinearity
        else:
            dZh = H*(1-H)* dh        
        db_hh = np.sum(dZh, axis=0, keepdims=True) 
        db_ih = np.sum(dZh, axis=0, keepdims=True) 
        dW_ih = np.dot(X.T,dZh)
        dW_hh = np.dot(H_pre.T,dZh)
        dh_pre = np.dot(dZh,self.W_hh.T)
        dx =  np.dot(dZh,self.W_ih.T)
        grads = (dW_ih,dW_hh,db_ih,db_hh)
        for a, b in zip(self.grads,grads):
            a+=b
        
        return dx,dh_pre,grads



def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
def lstm_cell(x, hc,w_ih, w_hh,b_ih, b_hh): 
    h,c = hc[0],hc[1]
    hidden_size = w_ih.shape[1]//4
    ifgo_Z = np.dot(x,w_ih) + b_ih  +  np.dot(h,w_hh) + b_hh
    i = sigmoid(ifgo_Z[:,:hidden_size])
    f = sigmoid(ifgo_Z[:,hidden_size:2*hidden_size])
    g = np.tanh(ifgo_Z[:,2*hidden_size:3*hidden_size])
    o = sigmoid(ifgo_Z[:,3*hidden_size:])   
    c_ = f*c+i*g
    h_ = o*np.tanh(c_)
    return (h_,c_),np.column_stack((i,f,g,o))

def lstm_cell_back(dhc,ifgo,x,hc_pre,w_ih, w_hh,b_ih, b_hh):
    hidden_size = w_ih.shape[1]//4
    if isinstance(dhc, tuple):
        dh_,dc_next = dhc
    else:
        dh_ = dhc
        dc_next = np.zeros_like(dh_)
    h_pre,c = hc_pre
    i,f,g,o = ifgo[:,:hidden_size],ifgo[:,hidden_size:2*hidden_size]\
              , ifgo[:,2*hidden_size:3*hidden_size],ifgo[:,3*hidden_size:]
    c_ = f*c+i*g
    dc_ = dc_next+dh_*o*(1-np.square(np.tanh(c_)))
    do = dh_*np.tanh(c_)
    di = dc_*g
    dg = dc_*i
    df = dc_*c
    
    diz = i*(1-i)*di
    dfz = f*(1-f)*df
    dgz = (1-np.square(g))*dg
    doz = o*(1-o)*do
    
    dZ = np.column_stack((diz,dfz,dgz,doz))
    
    dW_ih = np.dot(x.T,dZ)
    dW_hh = np.dot(h_pre.T,dZ)
    db_hh = np.sum(dZ, axis=0, keepdims=True) 
    db_ih = np.sum(dZ, axis=0, keepdims=True) 
    dx =  np.dot(dZ,w_ih.T)
    dh_pre = np.dot(dZ,w_hh.T)
    #return dx,dh_pre,(dW_ih,dW_hh,db_ih,db_hh)
    dc = dc_*f
    return dx,(dh_pre,dc),(dW_ih,dW_hh,db_ih,db_hh)

    
class LSTMCell(RNNCellBase):
    """   \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
        
        Inputs: input, (h_0, c_0)
        If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        
        Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch
        - **c_1** of shape `(batch, hidden_size)`: tensor containing the next cell state
          for each element in the batch
          
    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`
        
        """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__(input_size, hidden_size,bias, num_chunks=4)
     
    def init_hidden(batch_size):
        zeros= np.zeros(input.shape[0], self.hidden_size, dtype=input.dtype)  
        return (zeros, zeros)#np.array([zeros, zeros])
                
    def forward(self, input, h=None): 
        self.check_forward_input(input)
        if h is None:
            h = init_hidden(input.shape[0])
            #zeros= np.zeros(input.shape[0], self.hidden_size, dtype=input.dtype)  
            #h = (zeros, zeros)#np.array([zeros, zeros])
        self.check_forward_hidden(input, h[0], '[0]')
        self.check_forward_hidden(input, h[1], '[1]')
        return lstm_cell(
                input, h,
                self.W_ih, self.W_hh,
                self.b_ih, self.b_hh,
            )
    def __call__(self, input, h=None): 
        return self.forward(input,h)
    
    def backward(self, dhc,ifgo,input,hc_pre):
        if hc_pre is None:
            hc_pre = init_hidden(input.shape[0])
        dx,dh_pre,grads = lstm_cell_back(
                            dhc,ifgo,
                            input, hc_pre,
                            self.W_ih, self.W_hh,
                            self.b_ih, self.b_hh)
            
        #grads = (dW_ih,dW_hh,db_ih,db_hh)
        for a, b in zip(self.grads,grads):
            a+=b 
        return dx,dh_pre,grads


def gru_cell(x, h,w_ih, w_hh,b_ih, b_hh):
    Z_ih,Z_hh = np.dot(x,w_ih) + b_ih, np.dot(h,w_hh) + b_hh
    hidden_size = w_ih.shape[1]//3
    r = sigmoid(Z_ih[:,:hidden_size]+Z_hh[:,:hidden_size])
    u = sigmoid(Z_ih[:,hidden_size:2*hidden_size]+Z_hh[:,hidden_size:2*hidden_size]) 
    n = np.tanh(Z_ih[:,2*hidden_size:]+r*Z_hh[:,2*hidden_size:]) 
    h_next= u*h+(1-u)*n 
    run = np.column_stack((r,u,n))
    #return h_next,(r,u,n)  
    return h_next,run 

def gru_cell_back(dh,run,x,h_pre,w_ih, w_hh,b_ih, b_hh):
    hidden_size = w_ih.shape[1]//3
    #r,u,n = run
    r,u,n = run[:,:hidden_size],run[:,hidden_size:2*hidden_size]\
              , run[:,2*hidden_size:]
              
    #  H =  U H_pre+(1-U)H_tildas
    dn = dh*(1-u)
    dh_pre = dh*u
    du = h_pre*dh -n*dh   
  
    #n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) 
    dnz = (1-np.square(n))*dn 
        
    Z_hn = np.dot(h_pre,w_hh[:,2*hidden_size:])+b_hh[:,2*hidden_size:]
    dr = dnz*Z_hn
    dZ_ih_n = dnz
    dZ_hh_n = dnz*r
                
    duz = u*(1-u)*du
    dZ_ih_u = duz
    dZ_hh_u = duz
     
    drz = r*(1-r)*dr
    dZ_ih_r = drz
    dZ_hh_r = drz    
    
    dZ_ih = np.column_stack((dZ_ih_r,dZ_ih_u,dZ_ih_n))
    dZ_hh = np.column_stack((dZ_hh_r,dZ_hh_u,dZ_hh_n))            
    
    dW_ih = np.dot(x.T,dZ_ih)
    dW_hh = np.dot(h_pre.T,dZ_hh)
    db_ih = np.sum(dZ_ih, axis=0, keepdims=True) 
    db_hh = np.sum(dZ_hh, axis=0, keepdims=True)             
  
    dh_pre+=np.dot(dZ_hh,w_hh.T)
    dx =  np.dot(dZ_ih,w_ih.T)
    return dx,dh_pre,(dW_ih,dW_hh,db_ih,db_hh)

class GRUCell(RNNCellBase):
    """  \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}
        
        Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.
          
        Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch
          
        Attributes:
            weight_ih: the learnable input-hidden weights, of shape
                `(3*hidden_size, input_size)`
            weight_hh: the learnable hidden-hidden weights, of shape
                `(3*hidden_size, hidden_size)`
            bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
            bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`
        
        """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__(input_size, hidden_size,bias, num_chunks=3)
        
    def forward(self, input, h=None): 
        self.check_forward_input(input)
        if h is None:
            h= np.zeros(input.shape[0], self.hidden_size, dtype=input.dtype) 
        self.check_forward_hidden(input, h, '')
        return gru_cell(
                input, h,
                self.W_ih, self.W_hh,
                self.b_ih, self.b_hh,
            )  
    def __call__(self, input, h=None): 
        return self.forward(input,h)
    
    def backward(self, dh,run,input,h_pre):       
        if  h_pre is None:
            h_pre = np.zeros(input.shape[0], self.hidden_size, dtype=input.dtype)      
        dx,dh_pre,grads = gru_cell_back(
                            dh,run,
                            input, h_pre,
                            self.W_ih, self.W_hh,
                            self.b_ih, self.b_hh )
        #grads = (dW_ih,dW_hh,db_ih,db_hh)
        for a, b in zip(self.grads,grads):
            a+=b 
        return dx,dh_pre,grads
         
    
   
 #=========================RNNBase============================================
from Layers import * 
class RNNBase(Layer):
    def __init__(self,mode,input_size, hidden_size, n_layers,bias = True):
        super(RNNBase, self).__init__()
        
        self.mode = mode
        if mode == 'RNN_TANH':
            self.cells = [RNNCell(input_size, hidden_size,bias,nonlinearity="tanh")]
            self.cells += [RNNCell(hidden_size, hidden_size,bias,nonlinearity="tanh") for i in range(n_layers-1)]
        elif mode == 'RNN_RELU':
            self.cells = [RNNCell(input_size, hidden_size,bias,nonlinearity="relu")]
            self.cells += [RNNCell(hidden_size, hidden_size,bias,nonlinearity="relu") for i in range(n_layers-1)]
        elif mode == 'LSTM':
            self.cells = [LSTMCell(input_size, hidden_size,bias)]
            self.cells += [LSTMCell(hidden_size, hidden_size,bias) for i in range(n_layers-1)]
        elif mode == 'GRU':
            self.cells = [GRUCell(input_size, hidden_size,bias)]
            self.cells += [GRUCell(hidden_size, hidden_size,bias) for i in range(n_layers-1)]
        
        self.input_size, self.hidden_size = input_size,hidden_size
        self.n_layers = n_layers
        self.flatten_parameters()
        self._params = None
     
    def flatten_parameters(self):
        self.params = []
        self.grads = []
        for i in range(self.n_layers):
            rnn = self.cells[i]
            for j,p in enumerate(rnn.params):
                self.params.append(p)
                self.grads.append(rnn.grads[j])
                
    def forward(self, x,h=None):
        seq_len,batch_size = x.shape[0], x.shape[1]
        n_layers = self.n_layers
        mode = self.mode
        
        hs = [[] for i in range(n_layers)]
        zs = [[] for i in range(n_layers)]
        
        if h is None:
            h = self.init_hidden(batch_size)
        if False:
            if mode == 'LSTM':#isinstance(h, tuple):
                self.h = (h[0].copy(),h[1].copy())       
            else:
                self.h = h.copy()     
        else:
            self.h = h
       
        for i in range(n_layers):
            cell = self.cells[i]
            if i!=0:
                x = hs[i-1]  # out h of pre layer
                if mode == 'LSTM':
                    x = np.array([h for h,c in x])
                    
            hi = h[i]
            if mode == 'LSTM':
                hi = (h[0][i],h[1][i])
            for t in range(seq_len):
                hi =  cell(x[t],hi) 
                if isinstance(hi, tuple):
                    hi,z = hi[0],hi[1]
                    zs[i].append(z) 
             
                hs[i].append(hi)                
              #  if mode == 'LSTM' or mode == 'GRU':
              #      zs[i].append(z)                 
                
        self.hs = np.array(hs)  #(layer_size,seq_size,batch_size,hidden_size)
        if len(zs[0])>0:
            self.zs = np.array(zs)
        else:self.zs = None
        
        output = hs[-1] # containing the output features (`h_t`) from the last layer of the RNN,
        if mode == 'LSTM':
            output = [h for h,c in output]
        hn = self.hs[:,-1,:,:]  # containing the hidden state for `t = seq_len`
        return np.array(output),hn
    
    def __call__(self, x,h=None):
        return self.forward(x,h)
    
    def init_hidden(self, batch_size):
        zeros = np.zeros((self.n_layers, batch_size, self.hidden_size))
        if self.mode=='LSTM':
            self.h = (zeros,zeros)
        else:
            self.h = zeros
        return self.h
    
    def backward(self,dhs,input):#,hs):      
        if self.hs is None:
            self.hs,_ = self.forward(input)
        hs = self.hs
        zs = self.zs if self.zs is not None else hs        
        seq_len,batch_size = input.shape[0], input.shape[1]
       
        dinput = [None for i in range(seq_len)]
     
        if len(dhs.shape)==2:  # dh at last time(batch,hidden)
            dhs_ = [np.zeros_like(dhs) for i in range(seq_len)]
            dhs_[-1] = dhs
            dhs = np.array(dhs_)
        elif dhs.shape[0]!=seq_len:            
            raise RuntimeError(
                "dhs has inconsistent seq_len: got {}, expected {}".format(
                    dhs.shape[0], seq_len))
        else:
            #print('dhs has  consistent seq_len')
            pass
        #dhs.shape[0]==hs[0].shape[0]: #(seq,batch,hidden)

         #----dhidden--------    
        dhidden = [None for i in range(self.n_layers)]
        
        for layer in reversed(range(self.n_layers)):
            layer_hs = hs[layer]
            layer_zs = zs[layer]
            cell = self.cells[layer]
            if layer==0:
                layer_input = input
            else:
                if self.mode =='LSTM':
                    layer_input  = self.hs[layer-1]
                    layer_input = [h for h,c in layer_input]
                else:
                    layer_input = self.hs[layer-1]

            #print('layer_input.shape',layer_input.shape)
            #print('zs.shape',zs.shape)
            #print('layer_zs.shape',layer_zs.shape)

            h_0 = self.h[layer]                 
            dh = np.zeros_like(dhs[0]) #来自后一时刻的梯度                
            if self.mode =='LSTM':
                h_0 = (self.h[0][layer],self.h[1][layer])
                dc = np.zeros_like(dhs[0])
            for t in reversed(range(seq_len)):
                dh += dhs[t]          #后一时刻的梯度+当前时刻的梯度
                h_pre = h_0 if t==0 else layer_hs[t-1]
                if self.mode=='LSTM':
                    dhc = (dh,dc)
                    dx,dhc,_ = cell.backward(dhc,layer_zs[t],layer_input[t],h_pre)  
                    dh,dc = dhc
                else:
                    dx,dh,_ = cell.backward(dh,layer_zs[t],layer_input[t],h_pre)  
                if layer>0:
                    dhs[t] = dx
                else :
                    dinput[t] = dx
                #----dhidden--------    
                if t==0:
                    if self.mode=='LSTM':
                        dhidden[layer] = dhc
                    else:
                        dhidden[layer] = dh
            
                    
        return np.array(dinput),np.array(dhidden)

    def parameters(self):
        if self._params is None:
            self._params = []           
            for  i, _ in enumerate(self.params):  
                self._params.append([self.params[i],self.grads[i]])  
        return self._params

class RNN(RNNBase):
    def __init__(self,*args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'
        super(RNN, self).__init__(mode, *args, **kwargs)

class LSTM(RNNBase):
    def __init__(self,*args, **kwargs):        
        super(LSTM, self).__init__('LSTM', *args, **kwargs)
        
class GRU(RNNBase):
    def __init__(self,*args, **kwargs):        
        super(GRU, self).__init__('GRU', *args, **kwargs)
    