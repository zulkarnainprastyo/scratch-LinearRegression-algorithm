def initialize_params(dimensions):
  beta_o = 0
  beta_other = {random.random()
              for _ in range(dimensions)}
  return beta_o, beta_other