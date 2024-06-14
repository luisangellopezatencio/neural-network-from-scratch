from src.nn import Linear, MLP, MLPclassifier
import numpy as np

# Generar datos de entrenamiento simples (ejemplo de datos XOR)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # Etiquetas XOR codificadas en one-hot

# Crear una instancia de MLPclassifier
mlp = MLPclassifier(num_inputs=2, hidden_layers=(2,), num_outputs=2)

# Entrenar el modelo
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Realizar predicciones
predictions = mlp.predict(X_train)

# Imprimir resultados
print("Predicciones:")
print(predictions)

print("Pesos finales:")
for i, layer in enumerate(mlp.layers):
    print(f'Layer {i} weights:\n{layer.get_weights()}')