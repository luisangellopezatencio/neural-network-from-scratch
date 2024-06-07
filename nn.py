import numpy as np

class Perceptron:
    def __init__(self, num_inputs: int, sigma: float = 0.01) -> None:
        self.w = np.random.normal(0, sigma, (num_inputs + 1 , 1)) #+1 to include bias
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_hat = np.matmul(X, self.w)
        return y_hat
    
class MLP(Perceptron):
    def __init__(self, num_inputs: int, hidden_layes: tuple, num_outputs: int, sigma: float = 0.01) -> None:
        pass
    
    def feed_forward(self, X):
        y_hat = self.p3.predict(self.p2.predict(self.p1.predict(X)))
        return y_hat
        