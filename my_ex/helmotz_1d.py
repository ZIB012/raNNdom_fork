import deepxde as dde
import numpy as np

# General parameters
n = 1
precision_train = 10
precision_test = 30
hard_constraint = True
weights = 1000  # if hard_constraint == False
iterations = 15000
parameters = [1e-3, 3, 150, "sin"]

lamb = 200

# Define sine function
if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf

    sin = tf.sin
    cos = tf.cos

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)

    #f =  -(13*np.pi**2 + lamb)*tf.sin(3*np.pi*x + 3*np.pi/20)*tf.cos(2*np.pi*x+np.pi/10) - 2*lamb
    f = -(2*np.pi**2 + lamb)*tf.sin(np.pi*x)*tf.cos(np.pi*x)

    return dy_xx - lamb*y - f

def func(x):
    #return np.sin(3*np.pi*x+3*np.pi/20)*np.cos(2*np.pi*x+np.pi/10) + 2
    return np.sin(np.pi*x)*np.cos(np.pi*x)

def transform(x, y):
    res = x* (1 - x)
    return res * y


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Interval(0, 8)
wave_len = 1 / n

hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

if hard_constraint == True:
    bc = []
else:
    bc = dde.icbc.DirichletBC(geom, lambda x: func(x), boundary)


data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=nx_train ** 2,
    num_boundary=4 * nx_train,
    solution=func,
    num_test=nx_test ** 2,
)

net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
)

if hard_constraint == True:
    net.apply_output_transform(transform)

model = dde.Model(data, net)

if hard_constraint == True:
    model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
else:
    loss_weights = [1, weights]
    model.compile(
        "adam",
        lr=learning_rate,
        metrics=["l2 relative error"],
        loss_weights=loss_weights,
    )


losshistory, train_state = model.train(iterations=iterations)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# RANDOM FEATURE METHOD
