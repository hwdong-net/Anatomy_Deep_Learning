import pickle, gzip, urllib.request, json
import numpy as np
import os.path

def read_mnist():
    if not os.path.isfile("mnist.pkl.gz"):
        # Load the dataset
        urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
        
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

def draw_mnists(plt,X,indices):
    for i,index in enumerate(indices):
        plt.subplot(1, 10, i+1)
        plt.imshow(X[index].reshape(28,28),  cmap='Greys')
        plt.axis('off')
