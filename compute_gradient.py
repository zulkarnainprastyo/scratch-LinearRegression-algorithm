def compute_gradient(x, y, beta_0, beta_other,
                     dimension, m):
  gradient_beta_0 = 0
  gradient_beta_other = [0] * dimension

  for i in range(m):
    y_i_hat = sum(x[i][j] * beta_other[j]
                  for j in range(dimension)) + beta_0
    derror_dy = 2 * (y[i] - y_i_hat)
    for j in range(dimension):
      gradient_beta_other[j] += derror_y * x[i][j] / n
    gradient_beta_o += derror_dy / n

  return gradient_beta_0, gradient_beta_other