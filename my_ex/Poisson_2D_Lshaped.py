import sys
sys.path.append('../../raNNdom')
import deepxde as dde
from deepxde.backend import tf
import numpy as np

def pde(x, y):
    # Most backends
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend jax
    # dy_xx, _ = dde.grad.hessian(y, x, i=0, j=0, component=0)
    # dy_yy, _ = dde.grad.hessian(y, x, i=1, j=1, component=0)
    return -dy_xx - dy_yy - 1

def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)

R = 10
net = dde.nn.random_FNN([2] + [200] + [1], "random_sin", "Glorot uniform",Rm=R, b=R/10)
model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=10000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)