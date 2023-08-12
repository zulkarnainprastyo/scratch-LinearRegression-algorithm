import random

# Initialize random values for weight and bias
weight = random.random()
bias = random.random()

# Define a function to calculate mean squared error (MSE)
def mean_squared_error(predictions, actual):
    n = len(predictions)
    return sum((p - a) ** 2 for p, a in zip(predictions, actual)) / n

# Define a function for gradient descent to update weight and bias
def gradient_descent(X, predictions, actual, learning_rate):
    n = len(X)
    gradient_w = sum(-2 * x * (a - p) for x, p, a in zip(X, predictions, actual)) / n
    gradient_b = sum(-2 * (a - p) for p, a in zip(predictions, actual)) / n
    return gradient_w, gradient_b

# Loop for a fixed number of epochs or until convergence
epochs = 1000
convergence_threshold = 0.001

# Assuming X and actual are defined
X = [1, 2, 3, 4, 5]
actual = [2, 4, 5, 4, 5]

learning_rate = 0.01

for _ in range(epochs):
    predictions = [weight * x + bias for x in X]
    mse = mean_squared_error(predictions, actual)

    if mse < convergence_threshold:
        break

    gradient_w, gradient_b = gradient_descent(X, predictions, actual, learning_rate)
    weight -= learning_rate * gradient_w
    bias -= learning_rate * gradient_b

# Fitting is complete
