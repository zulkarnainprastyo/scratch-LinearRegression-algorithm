def linear_regression(x, y, iterations=100,
                      learning_rate=0.01):
  n, m = len(x[0]), len(x)
  beta_0, beta_other = initialize_params(n)
  for _ in range(iterations):
    gradient_beta_0, gradient_beta_other = compute_gradient(
        x, y, beta_0, beta_other, n, m)
    beta_0, beta_other = update_params(
        beta_0, beta_other, gradient_beta_0,
        gradient_beta_other, learning_rate)
    return beta_0, beta_other