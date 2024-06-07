import numpy as np

class Perceptron:
    def __init__(self, num_inputs: int, num_outputs: int, sigma: float = 0.01) -> None:
        self.w = np.random.normal(0, sigma, (num_inputs + 1 , num_outputs)) #+1 to include bias
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_hat = np.matmul(X, self.w)
        return y_hat
    
    def get_weights(self):
        return self.w

    def set_weights(self, w):
        self.w = w
    
class MLP():
    def __init__(self, num_inputs: int, hidden_layers: tuple, num_outputs: int, sigma: float = 0.01) -> None:
        self.all_perceptrons = []
        inputs_layer = self.init_perceptrons(num_inputs, num_outputs, sigma)
        self.all_perceptrons.append(inputs_layer)
        for num_perceptrons in hidden_layers:
            self.all_perceptrons.append(self.init_perceptrons(num_perceptrons, num_outputs, sigma))
    
    def feed_forward(self, X):
        y_hat = self.p3.predict(self.p2.predict(self.p1.predict(X)))
        return y_hat
    
    def init_perceptrons(self, num_perceptrons:int, num_outputs:int, sigma:float=0.01) -> list[Perceptron]:
        perceptrons = []
        for _ in range(num_perceptrons):
            p = Perceptron(num_perceptrons, num_outputs, sigma)
            perceptrons.append(p)
        return perceptrons
        