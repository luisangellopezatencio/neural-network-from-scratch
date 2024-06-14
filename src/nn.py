import numpy as np

class Linear():
    def __init__(self, num_inputs: int, num_outputs: int = 1, sigma: float = 0.01) -> None:
        self.w = np.random.randn(num_inputs + 1, num_outputs) * sigma  # Inicializar pesos aleatorios
        self.output = None  # Atributo para almacenar la salida de la capa
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        y_hat = np.matmul(X, self.w)
        self.output = y_hat  # Almacenar la salida
        return y_hat
    
    def get_weights(self):
        return self.w

    def set_weights(self, w):
        self.w = w
    
    def __str__(self):
        return f'Linear: input size = {self.w.shape[0] - 1}, output size = {self.w.shape[1]} and perceptron bias'

    def backward_pass_layer(self, input, grad_output):
        X = np.hstack((np.ones((input.shape[0], 1)), input))  # Add bias term
        grad_input = np.dot(grad_output, self.w.T)[:, 1:]  # Skip bias gradient
        grad_weights = np.dot(X.T, grad_output)
        return grad_input, grad_weights


class MLP():
    def __init__(self, num_inputs: int, hidden_layers: tuple, num_outputs: int, sigma: float = 0.01) -> None:
        self.layers = []

        # Input layer to first hidden layer
        if hidden_layers:
            self.layers.append(Linear(num_inputs, hidden_layers[0], sigma))
            # Hidden layers
            for i in range(1, len(hidden_layers)):
                self.layers.append(Linear(hidden_layers[i-1], hidden_layers[i], sigma))
            # Last hidden layer to output layer
            self.layers.append(Linear(hidden_layers[-1], num_outputs, sigma))
        else:
            # If there are no hidden layers
            self.layers.append(Linear(num_inputs, num_outputs, sigma))
    
    def reLU(self, x):
        return np.maximum(0, x)
    
    def reLU_gradient(self, x):
        return np.where(x > 0, 1, 0)
    
    def feed_forward(self, X):
        for i, layer in enumerate(self.layers):
            X = layer.predict(X)
            if i < len(self.layers) - 1:  # Apply ReLU for hidden layers
                X = self.reLU(X)
            layer.output = X  # Store output for use in backpropagation
        return X

    def __str__(self):
        layer_strs = []
        for i, layer in enumerate(self.layers):
            layer_str = str(layer)
            layer_str = f'Layer {i}: {layer_str}'
            layer_strs.append(layer_str)
        return 'MLP:\n' + '\n'.join(layer_strs)


class MLPclassifier(MLP):
    def __init__(self, num_inputs: int, hidden_layers: tuple, num_outputs: int, sigma: float = 0.01) -> None:
        super().__init__(num_inputs, hidden_layers, num_outputs, sigma)

    def predict(self, X):
        y_hat = self.feed_forward(X)
        y_hat = self.softmax(y_hat)  # Apply softmax for output layer
        return y_hat
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)  # Prevent log(0) error
        return -np.sum(y * np.log(y_hat)) / y.shape[0]  # Normalize by batch size
    
    def cross_entropy_gradient(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)  # Prevent division by zero
        return (y_hat - y) / y.shape[0]  # Normalize by batch size
    
    def backward_pass(self, X, y, y_hat, learning_rate=0.01):
        delta = self.cross_entropy_gradient(y_hat, y)
        gradients = []

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if i == len(self.layers) - 1:  # Output layer
                input_to_layer = self.layers[i-1].output if i > 0 else X
                delta, grad_weights = layer.backward_pass_layer(input_to_layer, delta)
            else:  # Hidden layers
                input_to_layer = self.layers[i-1].output if i > 0 else X
                delta = delta * self.reLU_gradient(input_to_layer)
                delta, grad_weights = layer.backward_pass_layer(input_to_layer, delta)
            
            # Actualizamos los pesos aquí usando set_weights
            new_weights = layer.get_weights() - learning_rate * grad_weights
            layer.set_weights(new_weights)

            gradients.append(grad_weights)
        
        return gradients

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            y_hat = self.predict(X_train)
            loss = self.cross_entropy_loss(y_hat, y_train)
            self.backward_pass(X_train, y_train, y_hat, learning_rate)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
