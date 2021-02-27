from Perceptron import Perceptron
import numpy as np

n_inputs = 2
x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
t = np.array([-1, 1, 1, 1])
alfa = 0.1
N = 5
p1 = Perceptron(n_inputs)

#p1.hebb_iter(np.array([0.5, 0.75]), np.array([1]))
for i in range(N):
    p1.error_train_epoch(x, t, alfa)
    E = p1.error_value(x,t)
    print(E)