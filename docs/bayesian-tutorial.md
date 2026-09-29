# Bayesian modelling and Stan: a tutorial using this model

This tutorial is for readers who know regression, probability distributions and maximum likelihood, but haven't done Bayesian data analysis. Every idea is tied to a place in this project where it is used, so you can open the file and see it at work.

**Contents**

1. [Bayes in one page](#1-bayes-in-one-page)
2. [Shrinkage: the idea behind house effects](#2-shrinkage)
3. [Hierarchical models and partial pooling](#3-hierarchical-models-and-partial-pooling)
4. [Choosing the scale: logs, softmax and the Dirichlet](#4-choosing-the-scale-logs-softmax-and-the-dirichlet)
5. [Letting coefficients drift: random walks](#5-letting-coefficients-drift-random-walks)
6. [The fundamentals model in Stan, line by line](#6-the-fundamentals-model-in-stan-line-by-line)
7. [How Stan fits a model: MCMC and NUTS](#7-how-stan-fits-a-model-mcmc-and-nuts)
8. [The error meta-model: variance components and fat tails](#8-the-error-meta-model-variance-components-and-fat-tails)
9. [From posterior to forecast: simulation](#9-from-posterior-to-forecast-simulation)
10. [Combining models: precision weighting and stacking](#10-combining-models-precision-weighting-and-stacking)
11. [Checking a probabilistic forecast](#11-checking-a-probabilistic-forecast)
12. [Try it yourself](#12-try-it-yourself)

---

## 1. Bayes in one page

In classical (frequentist) statistics a parameter such as "Le Pen's true support" is a fixed unknown, and probability describes the data. In Bayesian statistics the parameter is also uncertain, and probability describes *what we know about it*. The rule for updating that knowledge is Bayes' theorem:

$$\underbrace{p(\theta\mid y)}_{\text{posterior}} \;\propto\; \underbrace{p(y\mid\theta)}_{\text{likelihood}}\;\times\;\underbrace{p(\theta)}_{\text{prior}}$$

- **Likelihood.** The same object you maximise in maximum-likelihood estimation: how probable the observed data are for each parameter value.
- **Prior.** What you believed before seeing these data. It can come from earlier elections, or be deliberately vague.
- **Posterior.** The combination. It is a whole *distribution*, not a point estimate, so uncertainty comes for free.

**A worked example.** Suppose you believe a candidate's support is around 20%, give or take 4 points: prior $N(20, 4^2)$. A poll of 1,000 people then shows 24%. Its sampling error is about $\sqrt{0.24\cdot0.76/1000}\approx 1.35$ points, so the likelihood is $N(24, 1.35^2)$ as a function of true support. When a normal prior meets a normal likelihood, the posterior is normal too. Its mean is a **precision-weighted average**, where precision = 1/variance:

$$\text{mean}=\frac{20/4^2+24/1.35^2}{1/4^2+1/1.35^2}\approx 23.6,\qquad \text{sd}=\Big(\tfrac{1}{4^2}+\tfrac{1}{1.35^2}\Big)^{-1/2}\approx 1.28$$

The poll dominates because it is far more precise than the prior. Two ideas from this example run through the whole model:

1. **Estimates are pulled toward prior information, in proportion to how much data you have.** This is shrinkage.
2. **Combining sources means weighting each by its precision.** This is how the ensemble works.

Nearly everything a forecast needs comes from the **posterior predictive distribution**: the distribution of *future data* (the election result), averaging over parameter uncertainty:

$$p(\tilde y\mid y)=\int p(\tilde y\mid\theta)\,p(\theta\mid y)\,d\theta$$

In practice this integral is never computed with symbols. You draw $\theta$ from the posterior, then simulate $\tilde y$ from each draw; section 9 shows how.

## 2. Shrinkage

Pollster house effects are the cleanest example of Bayesian shrinkage in the model (`forecast/polls/house_effects.py`).

**The problem.** Ifop's polls ran on average 0.05 below the consensus for one candidate. Is that a real Ifop lean, or noise from a handful of polls? With 3 polls it is probably noise. With 50 it is probably real.

**The model.** Suppose firm $p$'s true lean $\ell_p$ is drawn from $N(0,\tau^2)$, the spread of leans across firms. Each poll's residual is $r_i=\ell_p+\varepsilon_i$ with $\varepsilon_i\sim N(0,\sigma^2)$. With $n$ polls averaging $\bar r$, the posterior mean of the lean is

$$E[\ell_p\mid r]=\frac{n/\sigma^2}{n/\sigma^2+1/\tau^2}\,\bar r=\frac{n}{n+k}\,\bar r,\qquad k=\sigma^2/\tau^2 .$$

That is the post's rule: keep $n/(n+5)$ of a firm's measured lean. Here $k = 5$ says a single poll's noise variance is about five times the variance of true leans across firms. With 5 polls the lean is kept at half strength; with 50 polls, at 91%.

**Race-specific leans.** The post adds back each firm's lean in one particular race at $n/(n+20)$. The large $k=20$ encodes a strong prior belief that house effects are mostly a property of the firm, not of the firm–race pair.

**The historical prior.** A firm's leans in past cycles form the prior *mean* instead of 0. They are blended in with weight $\alpha = \text{cap}\cdot W/(W+k)$, capped at 10%, so history informs the estimate without overruling the current cycle.

```python
from forecast.polls.house_effects import shrink
shrink(5, 5), shrink(50, 5), shrink(2, 20)   # (0.5, 0.909…, 0.0909…)
```

This is **empirical Bayes**. The prior's spread ($k$) is fixed from outside, and the formula is applied in closed form rather than fitting the full model. Full Bayes would estimate $\tau$ from the data at the same time. The shrinkage behaviour is the same.

## 3. Hierarchical models and partial pooling

Shrinkage generalises to **hierarchical** (multilevel) models: parameters for groups are themselves drawn from a distribution whose parameters are estimated.

There are three ways to estimate 100 département effects on the RN vote:

| Approach | Model | Problem |
| --- | --- | --- |
| No pooling | Estimate each département separately | Noisy; small départements get extreme estimates |
| Complete pooling | One shared effect | Ignores real differences |
| **Partial pooling** | $\delta_d\sim N(0,\sigma_{dept}^2)$, with $\sigma_{dept}$ estimated | Each estimate borrows strength from the others, by an amount the data choose |

The error meta-model (section 8) is hierarchical at four levels: national bloc, region, département and cluster. Each level's $\sigma$ is estimated from how much errors actually cluster at that level in past elections. If département errors turn out to be pure noise, $\sigma_{dept}$ is estimated near zero and that level switches itself off.

## 4. Choosing the scale: logs, softmax and the Dirichlet

Vote shares are *compositional*: they are positive and sum to 100%. Three tools recur for handling that.

- **Log shares plus softmax.** Work with unconstrained numbers $\eta_j$ (log shares up to a constant) and map them back with $s_j=e^{\eta_j}/\sum_k e^{\eta_k}$. Adding the same constant to every $\eta$ changes nothing, so only differences matter. Every error in the model is added on this scale and then softmaxed, so simulated shares are always valid.
- **Why logs rather than points?** A 1-point error means something very different for a 30% candidate and a 1% candidate. The backtest showed log-scale errors are closer to homogeneous, and a size term $(s/10\%)^{-\gamma}$ handles what's left.
- **The Dirichlet distribution** is a distribution over compositions. $y\sim\text{Dirichlet}(\phi\,p)$ has mean $p$, and the concentration $\phi$ controls the spread: larger $\phi$ means results scatter less around $p$. In the fundamentals model, $\phi\approx 22$ says French first rounds scatter widely around what history alone predicts.
- **Lognormal means vs medians.** If $\log X\sim N(\mu,\sigma^2)$ then $E[X]=e^{\mu+\sigma^2/2}$, which is larger than the median $e^\mu$. The simulator subtracts $\sigma^2/2$ so that noise doesn't inflate small candidates on average. As a result, the reported *medians* sit a little below the central estimate.

## 5. Letting coefficients drift: random walks

A standard regression assumes a coefficient, for example "incumbency is worth $\beta$", is the same in 1995 and in 2022. Lauderdale and Linzer's critique, which the post adopts, is that such fundamentals shift between elections, and pretending otherwise makes forecasts overconfident.

A **random walk** lets each coefficient drift:

$$\beta_1\sim N(\mu_0,\sigma_0^2),\qquad \beta_t=\beta_{t-1}+\tau\,z_t,\quad z_t\sim N(0,1).$$

- $\tau$ is estimated. If the coefficient barely moved in the past, $\tau$ is estimated small.
- The prediction for the *next* election uses $\beta_{T+1}=\beta_T+\tau z$. So a forecast carries both the uncertainty about $\beta_T$ and the uncertainty about how far it will drift.

Because every coefficient follows its own state equation like this, the post calls the model a **dynamic linear model**.

## 6. The fundamentals model in Stan, line by line

Stan is a language for writing down a probability model. You declare the data, the unknowns and the log density; Stan then draws samples from the posterior. Here is `forecast/stan/fundamentals_dlm.stan` in full, with comments.

```stan
data {
  int<lower=1> T;                 // number of training elections
  int<lower=1> K;                 // number of predictors
  int<lower=1> R;                 // races (one election's field, or one seat)
  int<lower=1> N;                 // rows: race × candidate
  array[R] int<lower=1, upper=T> race_t;   // which election each race belongs to
  array[R] int<lower=1, upper=N> start;    // rows start[r] .. start[r]+len[r]-1
  array[R] int<lower=2> len;
  matrix[N, K] X;                 // predictors
  vector[N] base;                 // fixed offset (used by the legislative version)
  vector<lower=0, upper=1>[N] y;  // observed shares, summing to 1 within a race
  vector[K] b1_mu;                // prior mean of the first-election coefficients
  vector<lower=0>[K] b1_sd;       // … and their prior sd
  vector<lower=0>[K] tau_scale;   // prior scale of the drift per election
  real phi_log_mu;                // prior centre for log(phi)
}
```

The `data` block declares everything that comes from Python. The constraints (`<lower=0>`) are checked when the data are passed in.

```stan
parameters {
  matrix[K, T] z;          // standardised innovations
  vector<lower=0>[K] tau;  // drift sd per predictor
  real<lower=0> phi;       // Dirichlet concentration
}
transformed parameters {
  matrix[K, T] beta;
  beta[, 1] = b1_mu + b1_sd .* z[, 1];
  for (t in 2:T)
    beta[, t] = beta[, t - 1] + tau .* z[, t];
}
```

Note that `beta` isn't sampled directly. Stan samples standard-normal `z` and *builds* `beta` from it. This is the **non-centred parameterisation**. When `tau` is small the posterior of `beta` becomes a narrow, curved "funnel" that samplers struggle with; `z` stays well behaved, and sampling becomes reliable. It is the most common trick in hierarchical Stan models.

```stan
model {
  to_vector(z) ~ std_normal();          // priors
  tau ~ normal(0, tau_scale);           // half-normal, since tau > 0
  phi ~ lognormal(phi_log_mu, 1);
  for (r in 1:R) {                      // likelihood, race by race
    int s = start[r];
    int e = s + len[r] - 1;
    vector[len[r]] eta = base[s:e] + X[s:e] * col(beta, race_t[r]);
    target += dirichlet_lpdf(y[s:e] | softmax(eta) * phi);
  }
}
```

`~` adds a log density to the running total, and `target +=` does the same thing explicitly. The whole `model` block is just $\log p(\text{prior}) + \log p(\text{data}\mid\text{parameters})$.

```stan
generated quantities {
  vector[K] beta_next;
  for (k in 1:K)
    beta_next[k] = beta[k, T] + tau[k] * normal_rng(0, 1);
}
```

`generated quantities` runs once per posterior draw, so `beta_next` is a draw from the predictive distribution of next election's coefficients. Python (`forecast/fundamentals/dlm.py`) then turns each draw into shares for any lineup, and adds Dirichlet noise. The result is the full posterior predictive: coefficient uncertainty, plus drift, plus election-level noise.

The priors live in `data/reference/priors.yaml`. For example, `log_baseline: [1.0, 0.5, 0.3]` means the first-election coefficient is $N(1, 0.5^2)$, with drift scale 0.3. They are *weakly informative*: they rule out absurd values (a baseline coefficient of −20) without dictating the answer.

## 7. How Stan fits a model: MCMC and NUTS

For all but the simplest models the posterior has no closed form, and it lives in many dimensions: this model has $K\cdot T+K+1$ parameters. **Markov chain Monte Carlo** (MCMC) sidesteps the problem. It builds a random walk through parameter space whose long-run visiting frequencies equal the posterior. You never get the formula, but you get thousands of draws from it, and any quantity you want is an average over the draws.

Stan's sampler is **NUTS**, the No-U-Turn Sampler, a form of **Hamiltonian Monte Carlo**. The intuition: treat $-\log p(\theta\mid y)$ as a landscape and roll a frictionless ball over it, using the gradient (which Stan computes automatically) to make long, informed moves instead of blind random steps. NUTS picks how long to roll each time.

A run has these parts:

| Term | Meaning | This project |
| --- | --- | --- |
| Chains | Independent runs from different starting points | 4 |
| Warm-up | Early iterations used to tune step size; discarded | 1,000 (400–800 for big models) |
| Draws | Kept iterations per chain | 1,000 |
| `adapt_delta` | Target acceptance rate; higher means smaller, safer steps | 0.9–0.95 |

**Diagnostics you should look at:**

- **R-hat** compares variation within and between chains. Values ≈ 1.00 mean the chains agree; above 1.01 means they haven't converged.
- **Effective sample size (ESS).** Draws are correlated; ESS is how many independent draws they are worth. Aim for several hundred.
- **Divergences.** The sampler met curvature it couldn't follow, usually a funnel. Even a few are a warning: reparameterise, or raise `adapt_delta`.

The warnings you may see in the logs during warm-up, such as `Scale parameter is 0` or `dirichlet_lpdf: … nan`, happen when the sampler tries an extreme value early on. They are harmless unless they persist into the kept draws.

**Running a model by hand:**

```python
import numpy as np
from forecast.stan.build import compile_model

model = compile_model("fundamentals_dlm")      # compiles once into forecast/stan/build/
data = {...}                                   # see tests/forecast/test_stan_models.py for a synthetic example
fit = model.sample(data=data, chains=4, iter_warmup=500, iter_sampling=500, seed=1)
print(fit.diagnose())                          # R-hat, ESS, divergences
beta = fit.stan_variable("beta")               # numpy array: (draws, K, T)
print(beta[:, :, -1].mean(axis=0), np.percentile(beta[:, 0, -1], [5, 95]))
```

In the pipeline, `forecast/stan/build.py::cached_sample` wraps this and caches the draws by a hash of (code, data, settings), so the backtest doesn't refit identical models.

**Why not maximum likelihood?** For the hierarchical error model, the maximum of the likelihood often sets a variance component to exactly zero. That says "this level never matters", which is overconfident. MCMC averages over plausible values instead.

## 8. The error meta-model: variance components and fat tails

`forecast/stan/error_meta.stan` asks: *how wrong has this forecasting system been, and in what pattern?* Its data are the system's own out-of-sample errors from the backtest.

$$\text{err}_n=\sum_{\ell}u_{\ell,g(n)}+v_n,\qquad u_{\ell,g}\sim t_\nu\!\big(0,\;a_\ell+b_\ell\sqrt{\text{days}_g}\big),\qquad v_n\sim t_\nu\!\big(0,\;\bar\sigma_n\,m_0(1+m_1\,\text{scarcity}_n+m_2\,\text{disagree}_n)\big)$$

- $u_{\ell,g}$ is a **shared shock**, common to every row in group $g$ at level $\ell$. At the bloc level, for example, one number is shared by all right-wing candidates in one snapshot. The model estimates how big these shocks typically are ($a_\ell$), and how fast that size grows with days to the election ($b_\ell$).
- $v_n$ is the **individual error**. Its size is the ensemble's own stated uncertainty $\bar\sigma_n$ times a correction. The correction checks whether the ensemble is honest: if $m_0$ is below 1, the ensemble was too cautious.
- This is a **variance-components** model, the hierarchical idea from section 3 applied to errors.

**Why a Student-t with ν = number of training cycles?** Thousands of error rows come from only five elections, and errors within one election are correlated. A normal distribution would treat those rows as thousands of independent pieces of evidence that big misses are rare. A $t$ distribution with 5 degrees of freedom has fat tails, so a 2002-style surprise stays plausible. That is the post's "we have only watched about ten elections actually happen".

**A trap, and a bug we fixed.** Stan's `student_t(nu, mu, sigma)` takes a **scale** $\sigma$, not a standard deviation; the sd is $\sigma\sqrt{\nu/(\nu-2)}$.

- The first simulator rescaled every t draw to unit variance. With ν = 3 that shrank the scale by 42%.
- The backtest's 50% intervals covered only 36% of results.
- After the fix, `antithetic_t(..., unit_variance=False)` whenever the sd came from the meta-model, coverage went to 44% (50% intervals) and 85% (90% intervals).

When you mix distributions, always check what each "sigma" means.

## 9. From posterior to forecast: simulation

Once you have a posterior (or a set of error scales), a forecast is a **Monte Carlo simulation**. You draw every uncertain quantity and apply the election rules deterministically. Each draw is one possible election, and any probability is a frequency:

$$P(\text{Le Pen wins})\approx\frac{\#\{\text{simulations Le Pen wins}\}}{40{,}000}$$

The Monte Carlo error is $\sqrt{p(1-p)/N}$. For $p = 0.5$ and $N = 40{,}000$ that is ±0.25 points, which is why reruns can differ in the first decimal.

Details from `forecast/simulate/presidential.py` worth understanding:

- **Structure beats formulas.** Nobody could write P(Philippe reaches the runoff *and* beats Le Pen) in closed form. Simulating the steps in order (lineup → January decision → first round → top two → runoff) gets it for free, with all the correlations built in.
- **Antithetic draws.** Shocks come in ± pairs, so they average to exactly zero and never shift the central estimate (a requirement in the post). This also reduces Monte Carlo noise.
- **Two-stage shocks for decisions.** For the January centrist rule, each candidate's error is split into the part realised by 15 January (fraction $f$ of the variance) and the part realised afterwards. The rule reads the January polls, $\mu + e_1$. The first round uses $\mu+e_1+e_2$. The candidates who stay keep their January draw, so the decision and the outcome are consistently correlated.
- **Scenario mixtures.** The lineup is itself random, with probabilities set in the YAML. Pooled results such as "P(RN wins)" average over lineups, exactly like averaging over parameter uncertainty in section 1.

## 10. Combining models: precision weighting and stacking

You have two forecasts of the same quantity, from polls and from fundamentals, each with an error sd. Section 1 says to combine them by precision:

$$\hat\mu=\frac{\mu_p/\sigma_p^2+\mu_f/\sigma_f^2}{1/\sigma_p^2+1/\sigma_f^2}.$$

This is optimal *if the errors are independent and the stated sds are right*. Both assumptions can fail. So:

1. **Measure the sds out of sample.** `PrecisionModel.fit` fits each component's actual error scale on the backtest archive, rather than trusting what the component claims.
2. **Stack.** Try multipliers on the fundamentals' precision ($\kappa\in\{0, 0.1, 0.25, 0.5, 1, 2\}$) and keep the one that minimised error on *earlier* cycles.

   This is a simple version of Bayesian stacking (Yao et al., 2018), which the post cites. Stacking optimises out-of-sample predictive performance directly, instead of assuming a model is true.

   The data chose $\kappa$ = 0.25–0.5: French presidential fundamentals are weak, so they get a small voice.

## 11. Checking a probabilistic forecast

A single election can't prove a probability wrong: a 30% event happening isn't a failure. You judge a forecaster over many forecasts. `forecast/backtest/metrics.py` implements four checks.

| Metric | Question | Ideal |
| --- | --- | --- |
| **Calibration** | Of all the times we said 70–85%, how often did it happen? | Close to the stated probability. Wilson intervals show how much is noise. |
| **Interval coverage** | Did the actual share fall inside the 50% (90%) interval about 50% (90%) of the time? | 50% / 90%. Lower means overconfident. |
| **Brier score** | Mean of $(p-y)^2$ over forecasts, $y \in \{0,1\}$ | Lower is better; always saying 50% scores 0.25. |
| **Log loss** | $-\overline{y\log p+(1-y)\log(1-p)}$ | Lower is better; punishes confident misses hard. |

**Time travel** is the cardinal sin. A backtest that fits its parameters on the elections it is then scored on looks brilliant and means nothing. `forecast/backtest/harness.py` runs one chronological pass. The 2017 snapshots use only models trained on 2002–2012, and polls published by that snapshot's date. The one disclosed leak is that historical lineups are the actual ballots.

This backtest is also what the ensemble weights and the error meta-model learn from, so its honesty matters twice.

## 12. Try it yourself

1. **Shrinkage.** In a notebook, run `polling_average` on `tests/forecast/synthetic.py`'s `make_polls()`. Then read `result.house_table`: P1 should lean about +0.10 on candidate B and P2 about −0.10. Halve the number of polls and see how much the estimates shrink.
2. **Priors matter less with more data.** In `tests/forecast/test_stan_models.py`, change `b1_mu` to `[3.0, 3.0]` and rerun with `pytest -m slow`. With 120 races the posterior barely moves. Now cut the data to 10 races.
3. **Random-walk drift.** Fit `fit_fundamentals` on 1995–2012 and on 1995–2022, then compare `beta_next.std(axis=0)`. The predictive sd includes drift, so it doesn't shrink to zero with more data.
4. **Fat tails.** Compare `np.quantile(antithetic_t(rng, (100000,), 3), [0.05, 0.95])` with a normal distribution's quantiles. Then see how P(Le Pen wins) changes when you set `errors.nu_floor: 30` in `config/model.yaml`.
5. **Judgment inputs.** Change the centrist `threshold` in `scenarios_pres_2027.yaml` from 15 to 12, rerun `python -m forecast run`, and watch the "Philippe alone" share rise.

**Further reading:**

- McElreath, *Statistical Rethinking*: the gentlest full introduction.
- Gelman et al., *Bayesian Data Analysis*: the reference.
- The Stan User's Guide: the chapters on hierarchical models and reparameterisation.
- Gelman, Hullman, Wlezien & Morris (2020), "Information, incentives, and goals in election forecasts", and the FiftyPlusOne methodology post saved in `fifty_plus_one/`.
