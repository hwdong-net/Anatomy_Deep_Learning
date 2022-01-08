import numpy as np

def row2im(dx_row,oH,oW,kH,kW,S):
    nRow,K2C = dx_row.shape[0],dx_row.shape[1]
    C = K2C//(kH*kW)
    N = nRow//(oH*oW)     #样本个数    
    oSize = oH*oW  
    H = (oH - 1) * S + kH
    W = (oW - 1) * S + kW
    dx = np.zeros([N,C,H,W])
    for i in range(oSize):
        row = dx_row[i::oSize,:]  #N个行向量       
        h_start = (i // oW) * S
        w_start = (i % oW) * S     
        dx[:,:,h_start:h_start+kH,w_start:w_start+kW] += row.reshape((N,C,kH,kW))  #np.reshape(row,(C,kH,kW))
    return dx

def row2im(dx_row,oH,oW,kH,kW,S):
    nRow,K2C = dx_row.shape[0],dx_row.shape[1]
    C = K2C//(kH*kW)
    N = nRow//(oH*oW)     #样本个数    
    oSize = oH*oW  
    #dx_row = dx_row.reshape(N,-1,dx_row.shape[1]) # N oS C*kH*kW
    
    H = (oH - 1) * S + kH
    W = (oW - 1) * S + kW
    dx = np.zeros([N,C,H,W])
    for h in range(oH):
        hS = h * S
        hS_kH = hS + kH
        h_start = h*oW
        for w in range(oW):
            wS = w*S          
            row =dx_row[h_start+w::oSize,:]                    
            dx[:,:,hS:hS_kH,wS:wS+kW] += row.reshape(N,C,kH,kW)          
    return dx

def conv_forward(X, K, S=1, P=0):
    N,C, H, W  = X.shape
    F,C, kH,kW = K.shape
    if P==0:
        X_pad = X
    else:
        X_pad = np.pad(X, ((0, 0), (0, 0),(P, P), (P, P)), 'constant')
   
    X_row = im2row(X_pad, kH,kW, S)
    
    K_col = K.reshape(K.shape[0],-1).transpose()    
    Z_row = np.dot(X_row, K_col)
    
    oH = (X_pad.shape[2] - kH) // S + 1
    oW = (X_pad.shape[3] - kW) // S + 1
    
    Z = Z_row.reshape(N,oH,oW,-1)
    Z = Z.transpose(0,3,1,2)
    return Z

def conv_backward(dZ,K,oH,oW,kH,kW,S=1,P=0):
    #将dZ摊平为和Z_row形状一样的矩阵
    F = dZ.shape[1]  # 将(N,F,oH,oW)转化为(N,oH,oW,F)
    dZ_row = dZ.transpose(0,2,3,1).reshape(-1,F)
    
    #计算损失函数关于卷积核参数的梯度   
    dK_col = np.dot(X_row.T,dZ_row) #X_row.T@dZ_row
    dK_col = dK_col.transpose(1,0)
    dK = dK_col.reshape(K.shape)
    db = np.sum(dZ,axis=(0,2,3))
    db = db.reshape(-1,F)
    
    K_col = K.reshape(K.shape[0],-1).transpose()  
    dX_row = np.dot(dZ_row,K_col.T)
    
    dX_pad = row2im(dX_row,oH,oW,kH,kW,S)
    if P == 0:
        return dX_pad,dK,db
    return dX_pad[:, :, P:-P, P:-P],dK,db
  
      
def get_im2row_indices(x_shape, kH, kW, S=1,P=0):  
  N, C, H, W = x_shape
  assert (H + 2 * P - kH) % S == 0
  assert (W + 2 * P - kH) % S == 0
  oH = (H + 2 * P - kH) // S + 1
  oW = (W + 2 * P - kW) // S + 1

  i0 = np.repeat(np.arange(kH), kW)
  i0 = np.tile(i0, C)
  i1 = S * np.repeat(np.arange(oH), oW)
  j0 = np.tile(np.arange(kW), kH * C)
  j1 = S * np.tile(np.arange(oW), oH)
  #i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  #j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  i = i0.reshape(1,-1) + i1.reshape(-1,1)
  j = j0.reshape(1,-1) + j1.reshape(-1,1)

  k = np.repeat(np.arange(C), kH * kW).reshape(1,-1)
  
  return (k, i, j)


def im2row_indices(x, kH, kW, S=1,P=0):  
  x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
  k, i, j = get_im2row_indices(x.shape, kH, kW, S,P)    
  rows = x_padded[:, k, i, j] 
  C = x.shape[1]
  rows = rows.reshape(-1,kH * kW * C) 
  return rows

def row2im_indices(rows, x_shape, kH, kW, S=1,P=0):
  N, C, H, W = x_shape
  H_pad, W_pad = H + 2 * P, W + 2 * P
  x_pad = np.zeros((N, C,H_pad, W_pad), dtype=rows.dtype)
  k, i, j = get_im2row_indices(x_shape, kH, kW, S,P)
  
  rows_reshaped = rows.reshape(N,-1,C * kH * kW)
 
  np.add.at(x_pad, (slice(None), k, i, j), rows_reshaped)
  if P == 0:
    return x_pad
  return x_pad[:, :, P:-P, P:-P]
