# raNNdom

```
data = dde.data.PDE(geom, pde, bc, num_domain, num_boundary, solution=func, num_test=100)

layer_size = [1] + [M]  + [5] + [1]
activation = ["sin", 'sin', 'linear']
initializer = "Glorot uniform"

net = dde.nn.random_FNN(layer_size, activation, initializer, Rm=10, b=0.0005)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=10000)
```

In random_FNN il primo layer ha pesi e biases non trainabili e scelti casualmente:

```
freeze = False
init = tf.keras.initializers.RandomUniform(minval=-Rm, maxval=Rm)
bias = tf.keras.initializers.RandomUniform(minval=-b, maxval=b)

self.denses.append(tf.keras.layers.Dense(units,
                    activation=(activation[j] if isinstance(activation, list) else activation),
                    kernel_initializer=init,
                    kernel_regularizer=self.regularizer,
                    bias_initializer=bias, 
                    trainable=freeze,
                )
            )


```