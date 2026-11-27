// Fundamentals model (post § 3.2): a regularised multinomial-logit regression
// whose coefficients drift as a random walk from election to election (a
// dynamic linear model with a softmax link). Each race is one election's
// field of candidates (presidential) or one constituency's blocs (legislative).
//
//   shares_race ~ Dirichlet(phi * softmax(base + X * beta[, t]))
//   beta[, 1] ~ N(b1_mu, b1_sd),  beta[, t] = beta[, t-1] + tau .* z_t
//
// beta_next steps the walk once more, so predictions for the next election
// carry the uncertainty of coefficient drift (Lauderdale & Linzer 2015).
data {
  int<lower=1> T;                          // training elections, oldest first
  int<lower=1> K;                          // predictors
  int<lower=1> R;                          // races
  int<lower=1> N;                          // rows (race × option)
  array[R] int<lower=1, upper=T> race_t;
  array[R] int<lower=1, upper=N> start;
  array[R] int<lower=2> len;
  matrix[N, K] X;
  vector[N] base;                          // fixed offset per row
  vector<lower=0, upper=1>[N] y;           // shares, floored, summing to 1 per race
  vector[K] b1_mu;
  vector<lower=0>[K] b1_sd;
  vector<lower=0>[K] tau_scale;
  real phi_log_mu;
}
parameters {
  matrix[K, T] z;
  vector<lower=0>[K] tau;
  real<lower=0> phi;
}
transformed parameters {
  matrix[K, T] beta;
  beta[, 1] = b1_mu + b1_sd .* z[, 1];
  for (t in 2:T)
    beta[, t] = beta[, t - 1] + tau .* z[, t];
}
model {
  to_vector(z) ~ std_normal();
  tau ~ normal(0, tau_scale);
  phi ~ lognormal(phi_log_mu, 1);
  for (r in 1:R) {
    int s = start[r];
    int e = s + len[r] - 1;
    vector[len[r]] eta = base[s:e] + X[s:e] * col(beta, race_t[r]);
    target += dirichlet_lpdf(y[s:e] | softmax(eta) * phi);
  }
}
generated quantities {
  vector[K] beta_next;
  for (k in 1:K)
    beta_next[k] = beta[k, T] + tau[k] * normal_rng(0, 1);
}
