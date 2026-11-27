// Error meta-model (post § 4.1.1): trained on the forecasting system's own
// out-of-sample mistakes. Each error is decomposed into shared shocks at L
// levels (presidential: bloc; legislative: national bloc, region,
// département, cluster) plus an individual error:
//
//   err_n ~ student_t(nu, Σ_l shock[gid[n, l]], sd_ens_n · m0 · (1 + m1·scarcity_n + m2·disagree_n))
//   shock_g ~ student_t(nu, 0, a[level_g] + b[level_g] · sqrt(days_g))
//
// nu = number of training cycles ("we have only watched about ten elections").
data {
  int<lower=1> N;
  vector[N] err;
  vector<lower=0>[N] sd_ens;
  vector<lower=0>[N] scarcity;       // 1 / (1 + n_polls)
  vector<lower=0>[N] disagree;       // |mu_polls - mu_fund| / pooled sd
  int<lower=0> L;
  int<lower=0> M;
  array[M] int<lower=1, upper=max(L, 1)> level_of;
  vector<lower=0>[M] sqrt_h;
  array[N, L] int<lower=0, upper=M> gid;   // 0 = row not in any group of that level
  real<lower=1> nu;
}
parameters {
  vector<lower=0>[L] a;
  vector<lower=0>[L] b;
  vector[M] zs;
  real<lower=0> m0;
  real<lower=0> m1;
  real<lower=0> m2;
}
transformed parameters {
  vector[M] shock;
  for (g in 1:M)
    shock[g] = zs[g] * (a[level_of[g]] + b[level_of[g]] * sqrt_h[g]);
}
model {
  vector[N] mu = rep_vector(0, N);
  for (n in 1:N)
    for (l in 1:L)
      if (gid[n, l] > 0)
        mu[n] += shock[gid[n, l]];
  zs ~ student_t(nu, 0, 1);
  a ~ normal(0, 0.10);
  b ~ normal(0, 0.02);
  m0 ~ lognormal(0, 0.5);
  m1 ~ normal(0, 2);
  m2 ~ normal(0, 1);
  err ~ student_t(nu, mu, sd_ens .* (m0 * (1 + m1 * scarcity + m2 * disagree)));
}
