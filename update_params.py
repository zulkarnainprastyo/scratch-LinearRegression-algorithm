def update_params(beta_0, beta_other, gradient_beta_0,
                  gradient_beta_other, learning_rate):
  beta_o += gradient_beta_0 * learning_rate
  for i in range(len(beta_1)):
    beta_other[i] += (gradient_beta_other[i] *
                      learning_rate)
  return beta_0, beta_other