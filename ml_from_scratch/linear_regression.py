# ml_from_scratch/linear_regression.py

class LinearRegression:
    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        for _ in range(epochs):
            predictions = self.predict(X)
            mse = self.mean_squared_error(predictions, y)
            gradient_w, gradient_b = self.gradient(X, predictions, y)
            self.weight -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b

    def predict(self, X):
        return [self.weight * x + self.bias for x in X]

    def mean_squared_error(self, predictions, actual):
        n = len(predictions)
        return sum((p - a) ** 2 for p, a in zip(predictions, actual)) / n

    def gradient(self, X, predictions, actual):
        n = len(X)
        gradient_w = sum(-2 * x * (a - p) for x, p, a in zip(X, predictions, actual)) / n
        gradient_b = sum(-2 * (a - p) for p, a in zip(predictions, actual)) / n
        return gradient_w, gradient_b
