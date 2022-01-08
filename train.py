import numpy as np

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
    

    def  regularization(self,reg):
        reg_loss =0
        for p,grad in self.params:
             grad[:] = grad+ 2*reg * p
             reg_loss+= np.sum(p**2)
        return reg*reg_loss

    def  loss_reg(self,reg):
        reg_loss =0
        for p,grad in self.params:            
             reg_loss+= np.sum(p**2)
        return reg*reg_loss
        
    def parameters(self):
        return self.params  
    
class SGD(Optimizer):
    def __init__(self,model_params,learning_rate=0.01, momentum=0.9,decay_every=None,decay=0.9):
        super().__init__()
        self.params,self.lr,self.momentum = model_params,learning_rate,momentum
        self.vs = []
        for p,grad in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)
            
        self.iter = 1
        self.decay_every,self.decay = decay_every,decay

    def reset(self):
        self.iter = 1
        for p in self.vs:
            p[:] = 0.

                
    def step(self): 
        for i,_ in enumerate(self.params):     
            p,grad = self.params[i] # p_grad           
            self.vs[i] = self.momentum*self.vs[i]+self.lr* grad             
            self.params[i][0] -= self.vs[i]
            
        self.iter +=1
        if self.decay_every is not None and self.iter%self.decay_every==0:
            self.lr *= self.decay
            

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
        for i in range(len(self.vs)):
            self.ms[i] = 0.
            self.vs[i] = 0.

                
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

        
#https://zhuanlan.zhihu.com/p/29920135
#https://github.com/benbo/adagrad/blob/master/adagrad.py
class AdaGrad(Optimizer):
    def __init__(self,model_params,learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.params,self.lr,self.momentum = model_params,learning_rate,momentum
        self.vs = []
        self.delta = 1e-7
        for p,grad in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)
        
        
    def step(self): 
        for i,_ in enumerate(self.params):     
            p,grad = self.params[i] # p_grad  
            self.vs[i] += grad**2
            self.params[i][0] -= self.lr* grad /(self.delta + np.sqrt(self.vs[i]))
            #self.vs[i] = self.momentum*self.vs[i]+self.lr* grad             
            #self.params[i][0] -= self.vs[i]

    def scale_learning_rate(self,scale):
        self.lr *= scale
        
    def debug_params(self):
         if DEBUG:
            for p,grad in self.params:  
                print("p",p)
                print("grad",grad)
            print()


class RMSprop(Optimizer):
    def __init__(self,model_params,learning_rate=0.01, beta=0.9,epsilon=1e-8):
        super().__init__()
        self.params,self.lr,self.beta,self.epsilon= model_params,learning_rate,beta,epsilon
        self.vs = []        
        for p,grad in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)
        

                
    def step(self): 
        beta = self.beta
        for i,_ in enumerate(self.params):     
            p,grad = self.params[i] 
            v = self.vs[i]
            v[:] = beta*v+(1-beta)*grad**2         
            p[:] = p - (self.lr* grad /(np.sqrt(v)+ self.epsilon))            
            #self.vs[i] = beta*self.vs[i]+(1-beta)*grad**2
            #self.params[i][0] -= self.lr* grad /(np.sqrt(self.vs[i])+ self.epsilon)
          
    def scale_learning_rate(self,scale):
        self.lr *= scale
        
  
    def debug_params(self):
         if DEBUG:
            for p,grad in self.params:  
                print("p",p)
                print("grad",grad)
            print()


def data_iterator(X,y,batch_size,shuffle=False):
    m = len(X)  
    indices = list(range(m))
    if shuffle:                 # shuffle是True表示打乱次序
        np.random.shuffle(indices)
    for i in range(0, m - batch_size + 1, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, m)])      
        yield X.take(batch_indices,axis=0), y.take(batch_indices,axis=0)


import datetime

def train_nn(nn,X,y,optimizer,loss_fn,epochs=100,batch_size = 50,reg = 1e-3,print_n=10,save= False):
    iter = 0
    losses = [] 
    today = datetime.datetime.now().strftime("%Y-%m-%d-%p-%I-%M") 
    
    for epoch in range(epochs):
        for X_batch,y_bacth in data_iterator(X,y,batch_size):     
            optimizer.zero_grad()      
          
            f = nn(X_batch) # nn.forward(X_batch)      
            loss,loss_grad = loss_fn(f, y_bacth)       
            nn.backward(loss_grad,reg)               
            loss += nn.reg_loss(reg)
           
            optimizer.step()

            losses.append(loss)            
            if iter%print_n==0:
                print('[%5d, %d] loss: %.3f' %(iter + 1, epoch + 1, loss))                
                #print(iter,"iter:",loss)
                if save:
                    nn.save_parameters(today+'-train_nn.npy')
            iter +=1          

    return losses

def grad_clipping(grads,alpha):
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i]*=ratio 