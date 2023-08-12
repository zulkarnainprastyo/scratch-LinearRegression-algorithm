# Linear Regression Implementation from Scratch

# Function to initialize parameters
def initialize_params(dimensions):
    beta_0 = 0
    beta_other = [random.random() for _ in range(dimensions)]
    return beta_0, beta_other

# Function to compute gradient of parameters
def compute_gradient(x, y, beta_0, beta_other, dimension, m):
    gradient_beta_0 = 0
    gradient_beta_other = [0] * dimension

    for i in range(m):
        y_i_hat = sum(x[i][j] * beta_other[j] for j in range(dimension)) + beta_0
        derror_dy = 2 * (y[i] - y_i_hat)
        for j in range(dimension):
            gradient_beta_other[j] += derror_dy * x[i][j] / m
        gradient_beta_0 += derror_dy / m

    return gradient_beta_0, gradient_beta_other

# Function to update parameters
def update_params(beta_0, beta_other, gradient_beta_0, gradient_beta_other, learning_rate):
    beta_0 += gradient_beta_0 * learning_rate
    for i in range(len(beta_other)):
        beta_other[i] += gradient_beta_other[i] * learning_rate
    return beta_0, beta_other

# Main linear regression function
def linear_regression(x, y, iterations=100, learning_rate=0.01):
    n, m = len(x[0]), len(x)
    beta_0, beta_other = initialize_params(n)

    for _ in range(iterations):
        gradient_beta_0, gradient_beta_other = compute_gradient(x, y, beta_0, beta_other, n, m)
        beta_0, beta_other = update_params(beta_0, beta_other, gradient_beta_0, gradient_beta_other, learning_rate)

    return beta_0, beta_other

# Example usage
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 5, 4, 5]

# Train the linear regression model
final_beta_0, final_beta_other = linear_regression(X, y, iterations=1000, learning_rate=0.01)

# Display the learned parameters
print("Final Intercept (beta_0):", final_beta_0)
print("Final Coefficients (beta_other):", final_beta_other)
