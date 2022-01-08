import numpy as np
#from im2row import *
from init_weights import *

class Layer:
    def __init__(self):
        self.params = None    
    def forward(self, x):       
        raise NotImplementedError
    def backward(self, x, grad):        
        raise NotImplementedError
    def reg_grad(self,reg):
        pass
    def reg_loss(self,reg):
        return 0.  
    
#----------加权和计算------------    
class Dense(Layer): 
    # Z = XW+b
    def __init__(self, input_dim, out_dim,init_method = None):  
        super().__init__()
      
        self.W = np.empty((input_dim, out_dim))
        self.b = np.zeros((1, out_dim))  
        self.reset_parameters(init_method)
             
            
        self.params = [self.W,self.b]
        self.grads = [np.zeros_like(self.W),np.zeros_like(self.b)]
  
    def reset_parameters(self,init_method = None):
        if init_method is not None:
            method_name,value = init_method
        else:
            method_name,value = 'kaiming_uniform',math.sqrt(5)
        
        if method_name == "kaiming_uniform":           
            kaiming_uniform(self.W,value)
            fan_in = self.W.shape[0]
            bound = 1 / math.sqrt(fan_in)
            self.b = np.random.uniform(-bound,bound,(self.b.shape))
        elif method_name == "kaiming_normal":           
            kaiming_normal(self.W,value)
            fan_in = self.W.shape[0]
            std = 1 / math.sqrt(fan_in)
            self.b = np.random.normal(0,std,(self.b.shape))
        elif method_name == "xavier_uniform":
            xavier_uniform(self.W,value)
        elif method_name == "xavier_normal":
            xavier_normal(self.W,value)
        else:
            self.W[:] = np.random.randn(self.W.shape) * value  #0.01 * np.random.randn
        
                    
    def forward(self, x): 
        self.x = x        
        x1 = x.reshape(x.shape[0],np.prod(x.shape[1:]))  #将多通道的x摊平      
        Z = np.matmul(x1, self.W) + self.b        
        return Z
    def __call__(self,x):
        return self.forward(x)
    
    def backward(self, dZ):
        # 反向传播      
        x = self.x
        x1 = x.reshape(x.shape[0],np.prod(x.shape[1:]))  #将多通道的x摊平
        dW = np.dot(x1.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)          
        dx = np.dot(dZ, np.transpose(self.W)) 
        dx = dx.reshape(x.shape)    #反摊平为多通道的x的形状   
        
        #self.grads = [dW, db]
        self.grads[0] += dW
        self.grads[1] += db
       
        return dx
    
    #--------添加正则项的梯度-----
    def reg_grad(self,reg):
        self.grads[0]+= 2*reg * self.W
        
    def reg_loss(self,reg):
        return  reg*np.sum(self.W**2)
    
    def reg_loss_grad(self,reg):
        self.grads[0]+= 2*reg * self.W
        return  reg*np.sum(self.W**2)


class Relu(Layer):
    def __init__(self):
        super().__init__()
       
    def forward(self, x):
        self.x = x  
        return np.maximum(0, x)

    def __call__(self,x):
        return self.forward(x)

    def backward(self, grad_output):
        # 如果x>0，导数为1,否则0
        x = self.x
        relu_grad = x > 0
        return grad_output * relu_grad 
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        self.x = x  
        return 1.0/(1.0 + np.exp(-x))    

    def __call__(self,x):
        return self.forward(x) 

    def backward(self, grad_output): 
        x = self.x  
        a  = 1.0/(1.0 + np.exp(-x))         
        return grad_output * a*(1-a) 
    
class Tanh(Layer):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        self.x = x  
        self.a = np.tanh(x)  
        return self.a    

    def __call__(self,x):
        return self.forward(x)

    def backward(self, grad_output):           
        d = (1-np.square(self.a))           
        return grad_output * d
    
class Leaky_relu(Layer):
    def __init__(self,leaky_slope):
        super().__init__()
        self.leaky_slope = leaky_slope        
    def forward(self, x):
        self.x = x  
        return np.maximum(self.leaky_slope*x,x)     

    def __call__(self,x):
        return self.forward(x)       

    def backward(self, grad_output): 
        x = self.x    
        d=np.zeros_like(x)
        d[x<=0]=self.leaky_slope
        d[x>0]=1       
        return grad_output * d

