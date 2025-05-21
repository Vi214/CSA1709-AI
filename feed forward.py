import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class FeedForwardNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        self.learning_rate = learning_rate

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y, Y_hat):
        # Cross-entropy loss
        m = Y.shape[0]
        log_likelihood = -np.log(Y_hat[range(m), Y])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, X, Y):
        m = X.shape[0]
        # One-hot encoding
        Y_onehot = np.zeros_like(self.A2)
        Y_onehot[np.arange(m), Y] = 1

        dZ2 = self.A2 - Y_onehot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def fit(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            self.backward(X, Y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=1)

# Example usage:
if __name__ == "__main__":
    # Dummy dataset: 4 samples, 2 features, 2 classes
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([0, 1, 1, 0])  # XOR labels

    nn = FeedForwardNeuralNetwork(input_dim=2, hidden_dim=4, output_dim=2, learning_rate=0.1)
    nn.fit(X, Y, epochs=1000)
    preds = nn.predict(X)
    print("Predictions:", preds)
