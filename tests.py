from nn import Perceptron
import numpy as np

X = np.array([
    [1,4],
    [2,5],
    [3,6],
])

p = Perceptron(num_inputs=2)
print(p.predict(X))