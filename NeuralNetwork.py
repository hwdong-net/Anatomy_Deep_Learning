from Layers import *
#from Train import *
import util

class NeuralNetwork:  
    def __init__(self):
        self._layers = []
        self._params = []
 
    def add_layer(self, layer):      
        self._layers.append(layer)
        if layer.params: 
           # for  i in range(len(layer.params)): 
            for  i, _ in enumerate(layer.params):                         
                self._params.append([layer.params[i],layer.grads[i]])            
    
    def forward(self, X): 
        for layer in self._layers:
            X = layer.forward(X) 
        return X   

    def __call__(self, X):
        return self.forward(X)
    
    def predict(self, X):
        p = self.forward(X)       
        if p.ndim == 1:     #单样本
            return np.argmax(ff)  
        return np.argmax(p, axis=1)   # 多样本  
   
    def backward(self,loss_grad,reg = 0.):
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i] 
            loss_grad = layer.backward(loss_grad)
            layer.reg_grad(reg) 
        return loss_grad
    
    def reg_loss(self,reg):
        reg_loss = 0
        for i in range(len(self._layers)):
            reg_loss+=self._layers[i].reg_loss(reg)
        return reg_loss
    
    def parameters(self): 
        return self._params
    
    def zero_grad(self):
        for i,_ in enumerate(self._params):           
            #self.params[i][1].fill(0.) 
            self.params[i][1][:] = 0 
            
    def get_parameters(self):
        return self._params   
    
    def save_parameters(self,filename):
        params = {}
        for i in range(len(self._layers)):
            if self._layers[i].params:
                params[i] = self._layers[i].params
        np.save(filename, params)               
        
    def load_parameters(self,filename):
        params = np.load(filename,allow_pickle = True)
        count = 0
        for i in range(len(self._layers)):
            if self._layers[i].params:
                layer_params = params.item().get(i)
                self._layers[i].params = layer_params                
                for j in range(len(layer_params)):                   
                    self._params[count][0] = layer_params[j]   
                    count+=1  
                    
                    

#-----------------y优化器-------------------------
class Optimizer():
    def __init__(self):
        self.params = None
        
    def reset(self):
        pass

    def zero_grad(self):  
        if self.params is None:    return
        for i,_ in enumerate(self.params):
             #self.params[i][1]*= 0.
            self.params[i][1][:] = 0.  
     
    def step(self): 
        pass
    
    def parameters(self):
        return self.params 
    
class SGD(Optimizer):
    def __init__(self,model_params,learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.params,self.lr,self.momentum = model_params,learning_rate,momentum
        self.vs = []
        for p,grad in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)

    def reset(self):
        for p in self.vs:
            p[:] = 0.

        
    def zero_grad(self):   
        for i,_ in enumerate(self.params):
             self.params[i][1]*= 0.
            #self.params[i][1][:] = 0.          
            #self.params[i][1].fill(0) 
                
    def step(self): 
        for i,_ in enumerate(self.params):     
            p,grad = self.params[i] # p_grad           
            self.vs[i] = self.momentum*self.vs[i]+self.lr* grad             
            self.params[i][0] -= self.vs[i]

    def scale_learning_rate(self,scale):
        self.lr *= scale

    
    def debug_params(self):
         if DEBUG:
            for p,grad in self.params:  
                print("p",p)
                print("grad",grad)
            print()

class Adam(Optimizer):
    def __init__(self,model_params,learning_rate=0.01, beta_1 = 0.9,beta_2 = 0.999,epsilon =1e-8):
        super().__init__()
        self.params,self.lr = model_params,learning_rate
        self.beta_1,self.beta_2,self.epsilon = beta_1,beta_2,epsilon
        self.ms = []
        self.vs = []
        self.t = 0
        for p,grad in self.params:
            m = np.zeros_like(p)
            v = np.zeros_like(p)
            self.ms.append(m)
            self.vs.append(v)

    def reset(self):
        self.t = 0
        for i in range(len(self.vs)):
            self.ms[i][:] = 0.
            self.vs[i][:] = 0.
            
    def zero_grad(self):        
        #for p,grad in params:      
        for i,_ in enumerate(self.params):
            #self.params[i][1]*= 0.
            #self.params[i][1][:] = 0.          
            self.params[i][1].fill(0) 
                
    def step(self):   
        #for  i in range(len(self.params)): 
        beta_1,beta_2,lr = self.beta_1,self.beta_2,self.lr
        self.t+=1
        t = self.t
        for i,_ in enumerate(self.params):     
            p,grad = self.params[i]       
            
            self.ms[i] = beta_1*self.ms[i]+(1-beta_1)*grad
            self.vs[i] = beta_2*self.vs[i]+(1-beta_2)*grad**2        
            
            
            m_1 = self.ms[i]/(1-np.power(beta_1, t))
            v_1 = self.vs[i]/(1-np.power(beta_2, t))  
            self.params[i][0]-= lr*m_1/(np.sqrt(v_1)+self.epsilon)
      
    def scale_learning_rate(self,scale):
        self.lr *= scale
        
def train_nn(nn,X,y,optimizer,loss_fn,epochs=100,batch_size = 50,reg = 1e-3,print_n=10):
    iter = 0
    losses = [] 
    for epoch in range(epochs):
        for X_batch,y_bacth in util.data_iterator(X,y,batch_size):     
            optimizer.zero_grad()      
            if True:
                f = nn(X_batch) # nn.forward(X_batch)      
                loss,loss_grad = loss_fn(f, y_bacth)       
                nn.backward(loss_grad,reg)               
                loss += nn.reg_loss(reg)
           # loss = nn.backpropagation(X_batch,y_bacth,loss_fn,reg)
            optimizer.step()

            losses.append(loss)
            
            if iter%print_n==0:
                print(iter,"iter:",loss)
            iter +=1 
          

    return losses       
        