import numpy as np

def sampling_simplex(num_points, dim):
    ''' Return uniformly random vector in the n-simplex '''
    x = np.random.exponential(scale=1.0, size=(num_points, dim))
    return x / np.sum(x,axis=1)[:,None]
