from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ...backend import tf
import numpy as np
import matplotlib.pyplot as plt

class random_FNN(NN):
    """Random fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,           # list of numerical strings
        activation,            # list of strings
        kernel_initializer,
        Rm=1,
        b=0.0005,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate
        self.denses = []
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            fun_activation = list(map(activations.get, activation))
        else:
            fun_activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        # iteration over the list layer_sizes
        for j, units in enumerate(layer_sizes[1:-1]):

            if activation[j] == 'random_sin' or activation[j] == 'random_tanh':
                free = False
                init = tf.keras.initializers.RandomUniform(minval=-Rm, maxval=Rm)
                bias = tf.keras.initializers.RandomUniform(minval=-b, maxval=b)
            else:
                free = True
                init = initializer
                bias = "zeros"

            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=(
                        fun_activation[j]
                        if isinstance(fun_activation, list)
                        else fun_activation
                    ),
                    kernel_initializer=init,
                    kernel_regularizer=self.regularizer,
                    bias_initializer=bias, 
                    trainable=free,
                )
            )
            if self.dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.denses.append(
            tf.keras.layers.Dense(
                layer_sizes[-1],
                kernel_initializer=initializer,
                kernel_regularizer=self.regularizer,
            )
        )

    def __call__(self, inputs, training=False):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)
        for f in self.denses:
            y = f(y, training=training)
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
    


class partition_random_FNN(NN):
    """Partitioned random fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        npart,
        nn_ind,
        Rm=1,
        b=0.0005,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate
        
        self.nets = [random_FNN(layer_sizes,activation,kernel_initializer,Rm,b,regularization,dropout_rate) for i in range(npart)]

        self.denses = [self.nets[i].denses for i in range(npart)]

        self.nn_ind = nn_ind

        self.npart = npart


    def __call__(self, inputs, training=True):

        x = inputs
        res = 0
        for i in range(self.npart):
            y = inputs
            indicator = self.nn_ind[i](x)

            if self._input_transform is not None:
                y = self._input_transform(y)

            for f in self.denses[i]:
                y = f(y, training=training)

            if self._output_transform is not None:
                y = self._output_transform(inputs, y)

            y = tf.math.multiply(y, indicator)
            res += y
        return res