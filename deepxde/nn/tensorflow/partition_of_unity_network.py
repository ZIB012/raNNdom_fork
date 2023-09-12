from .nn import NN
from .random_fnn import partition_random_FNN
from .. import activations
from .. import initializers
from .. import regularizers
from ...backend import tf
import numpy as np

'''
Construction of the partition of unity (PoU) functions with smoothness properties.
'''

def func_dx(x, a=0):    # auxiliary function defines right side of the function
    c1 = - 0.16171875
    res = (c1 + 1/5*(x-a)**5 - (x-a)**4 + 47/24*(x-a)**3 - 15/8*(x-a)**2 + 225/256*(x-a))/0.001041666666667
    return res

def func_sx(x, a=0):   # auxiliary function defines left side of the function
    c2 =  0.1627604166666665
    res = (c2 + 1/5*(x-a)**5 + (x-a)**4 + 47/24*(x-a)**3 + 15/8*(x-a)**2 + 225/256*(x-a))/0.001041666666666
    return res

# definition of the basic PoU function defined over [-1, 1]
def pou(x, a=-1, b=1):
    return tf.clip_by_value(1-func_dx(x, b-1), 0.0, 1.0) + tf.clip_by_value(func_sx(x, a+1), 0.0, 1.0) - 1

# definition of basic Pou function of the rightmost partition
def pou_dx(x, b=1, one=1.5):
    return tf.clip_by_value(func_sx(x, b-1+one), 0.0, 1.0)

# definition of basic Pou function of the leftmost partition
def pou_sx(x, a=-1, one=1.5):
    return tf.clip_by_value(1-func_dx(x, a+1-one), 0.0, 1.0)

# definition of the i_th PoU function defined over [a, b]
def indicator(lim_sx, lim_dx, a, b, npart, i):
    if i == 0:
        return lambda x: pou_sx(x, a, np.abs(b) + (2 + lim_sx))
    elif i == npart-1:
        return lambda x: pou_dx(x, b, np.abs(a) + (2 - lim_dx))
    else:
        return lambda x: pou(x, a, b)

'''Function to call in order to create all the PoU functions needed to construct
    the partition_random_FNN. 
    Args:
        geom: variable that contains all the geometry informations of the problem.
            In particular the domain's extrema
        
        npart: number of partitions to construct
        
    It returns a list containing all the PoU functions
'''
def pou_indicators(geom, npart):
    lim_dx = geom.r
    lim_sx = geom.l
    arr = np.linspace(lim_sx, lim_dx, npart+1)
    res = []
    for i in range(npart):
        res.append(indicator(lim_sx, lim_dx, arr[i], arr[i+1], npart, i))
    return res