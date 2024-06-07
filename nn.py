import numpy as np

class Perceptron:
    def __init__(self, num_inputs: int, num_outputs: int, sigma: float = 0.01) -> None:
        #self.w = np.random.normal(0, sigma, (num_inputs + 1 , num_outputs)) #+1 to include bias
        self.w = np.ones((num_inputs + 1, num_outputs))
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_hat = np.matmul(X, self.w)
        return y_hat
    
    def get_weights(self):
        return self.w

    def set_weights(self, w):
        self.w = w
    def __str__(self):
        return f'Perceptron: input size = {self.w.shape[0] - 1}, output size = {self.w.shape[1]} and perceptron bias'
    
class MLP():
    def __init__(self, num_inputs: int, hidden_layers: tuple, num_outputs: int, sigma: float = 0.01) -> None:
        self.all_perceptrons = []
        inputs_layer = self.init_perceptrons(num_inputs, hidden_layers[0], sigma)
        self.all_perceptrons.append(inputs_layer)
        num_perceptrons_prev = hidden_layers[0]
        for i in range(len(hidden_layers) - 1):
            self.all_perceptrons.append(self.init_perceptrons(num_perceptrons_prev, hidden_layers[i+1], sigma))
            num_perceptrons_prev = hidden_layers[i+1]
        last_hidden_layer = self.init_perceptrons(hidden_layers[-1], num_outputs, sigma)
        self.all_perceptrons.append(last_hidden_layer)
        output_layer = self.init_perceptrons(num_outputs, 1, sigma)
        self.all_perceptrons.append(output_layer)
    
    def feed_forward(self, X):
        y_hat = self.p3.predict(self.p2.predict(self.p1.predict(X)))
        return y_hat
    
    def init_perceptrons(self, num_perceptrons:int, num_outputs:int, sigma:float=0.01) -> list[Perceptron]:
        perceptrons = []
        for _ in range(num_perceptrons):
            p = Perceptron(num_perceptrons, num_outputs, sigma)
            perceptrons.append(p)
        return perceptrons
    
    def __str__(self):
        layer_strs = []
        for i, layer in enumerate(self.all_perceptrons):
            perceptron_str = str(layer[0])
            layer_str = f'Layer {i}: {len(layer)} {perceptron_str}'
            layer_strs.append(layer_str)
        return 'MLP:\n' + '\n'.join(layer_strs)