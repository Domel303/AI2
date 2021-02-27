from Perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('dataset_40_sonar.txt', delimiter = ',')
np.random.shuffle(data)
inputs = data[:, 0:60]
targets = data[:, 60]

n_inputs = 60
alfa = 0.005
N = 200
E = np.zeros(N)

p1 = Perceptron(n_inputs)


for i in range(N):
    p1.error_train_epoch(inputs, targets, alfa)
    E[i] = p1.error_value(inputs, targets)


plt.plot(E)
plt.ylabel('E')
plt.xlabel('Epochy')
plt.show()