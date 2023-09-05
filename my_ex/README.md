# Numerical Results

Here you can find the code related to all the numerical results presented in our [report](../report.pdf) and more.

### Parameters
The main parameters we focused on:
- $M =$ it's the hidden layer dimension; so it's the number of random basis function we are considering
- $k_m,\, b_m =$ they are the random but fixed weights and biases representing the activation function of the hidden layer. They are usually sampled from a uniform distribution.
- $R_m = $ it's the coefficient that describes the domain of the uniform distributions from which $k_m$ and $b_m$ are sampled

### Helmhotz.py
$
\begin{cases}
            -\frac{d^2u(x)}{dx^2} + u(x) = f(x) & x \in \,\Omega \\
            u(-1) = c_1,\;\;\; u(1) = c_2
        \end{cases}
$

The explicit form of the solution u is assumed to be $u(x) = 0.5\,x^5 + 1.3\,x^4 - 2.7\,x^3 - 5.5\,x^2 + 2.7\,x + 2.3$

In this problem we set the hyperparameters as follows:
- $M = 100$
- number of training points $n_p = 40$
- $R_m = 10$
- weights $k_m \sim  \mathcal{U}([-R_{m},\,R_{m}])$
- biases $b_m \sim  \mathcal{U}([-R_{m},\,R_{m}])$

