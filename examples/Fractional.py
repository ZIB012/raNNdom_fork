import sys
sys.path.append('../../raNNdom')
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
from scipy.special import gamma

alpha = 1.8


# Backend tensorflow.compat.v1
def fpde(x, y, int_mat):
    """du/dt + (D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)"""
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        int_mat = tf.SparseTensor(*int_mat)
        lhs = -tf.sparse_tensor_dense_matmul(int_mat, y)
    else:
        lhs = -tf.matmul(int_mat, y)
    dy_t = tf.gradients(y, x)[0][:, 1:2]
    x, t = x[:, :-1], x[:, -1:]
    rhs = -dy_t - tf.exp(-t) * (
        x ** 3 * (1 - x) ** 3
        + gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
        - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
        + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
        - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
    )
    return lhs - rhs[: tf.size(lhs)]


def func(x):
    x, t = x[:, :-1], x[:, -1:]
    return np.exp(-t) * x ** 3 * (1 - x) ** 3


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

# Static auxiliary points
data = dde.data.TimeFPDE(
    geomtime,
    fpde,
    alpha,
    [bc, ic],
    [52],
    meshtype="static",
    num_domain=400,
    solution=func,
)

# Dynamic auxiliary points
'''data = dde.data.TimeFPDE(
     geomtime,
     fpde,
     alpha,
     [bc, ic],
     [100],
     num_domain=20,
     num_boundary=1,
     num_initial=1,
     solution=func,
     num_test=50,
 )'''


M = 50
layer_size = [1] + [M]  +  [1]

activation = ["random_tanh", 'linear']

initializer = "Glorot uniform"
R = 4
net = dde.nn.random_FNN(layer_size, activation, initializer, Rm=R, b=R/10)
#net = dde.nn.FNN(layer_size, activation, initializer)

net.apply_output_transform(
    lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * y
    + x[:, 0:1] ** 3 * (1 - x[:, 0:1]) ** 3
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000)
print(losshistory.loss_test[-1])
model.compile("L-BFGS", lr=1e-3)
losshistory, train_state = model.train(iterations=20000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)