## 1. Bayes Rule

We have data that we believe come from an underlying distribution of unknown parameters. If we find those parameters, we know everything about the process that generated this data and we can make inferences (create new data).


$P(\theta|\textbf{D}) = \frac{P(\textbf{D} |\theta) P(\theta) }{P(D)}$


$P(\theta|\textbf{D})$ is the **posterior** distribution, prob(hypothesis | data) 

$P(\textbf{D} |\theta)$ is the **likelihood** function, how probable is my data **B** for different values of the parameters

$P(\theta)$ is the marginal probability to observe the data, called the **prior**, this captures our belief about the data before observing it.

$P(\textbf{D})$ is the marginal distribution (sometimes called marginal likelihood)

#### But what is $\theta \;$?

$\theta$ is an unknown yet fixed set of parameters. In Bayesian inference we express our belief about what $\theta$ might be and instead of trying to guess $\theta$ exactly, we look for its **probability distribution**. What that means is that we are looking for the **parameters** of that distribution. For example, for a Poisson distribution our $\theta$ is only $\lambda$. In a normal distribution, our $\theta$ is often just $\mu$ and $\sigma$.



## 3. Probability distributions in `scipy` and `PyMC3`

We can invoke probability distributions from `scipy` or directly from `PyMC3`. Distributions in `PyMC3` live within the context of models, although the framework provides a way to use the distributions outside of models. For a review of most common discete and continuous distributions see separate notebook.

### `scipy`
 
- **Normal** (a.k.a. Gaussian):
$X \sim  \mathcal{N}(\mu,\,\sigma^{2})$

    A Normal distribution can be parameterized either in terms of precision $\tau$ or variance $\sigma^{2}$. The link between the two is given by $\tau = \frac{1}{\sigma^{2}}$
 - Expected value (mean) $\mu$
 - Variance $\frac{1}{\tau}$ or $\sigma^{2}$
 - Parameters: `mu: float`, `sigma: float` or `tau: float`
 - Range of values (-$\infty$, $\infty$)


## 3. Bayesian Linear Regression
Our problem is the following: we want to perform multiple linear regression to predict an outcome variable $Y$ which depends on variables $\bf{x}_1$ and $\bf{x}_2$.

We will model $Y$ as normally distributed observations with an expected value $mu$ that is a linear function of the two predictor variables, $\bf{x}_1$ and $\bf{x}_2$.

$
Y \sim  \mathcal{N}(\mu,\,\sigma^{2})
$ 

$
\mu = \beta_0 + \beta_1 \bf{x}_1 + \beta_2 x_2 
$

where $\sigma^2$ represents the measurement error (in this example, we will use $\sigma = 10$). **Note:** In the code we give the value for the standard deviation $\sigma$.

We also choose the parameters to have normal distributions with those parameters set by us.

$
\beta_i \sim  \mathcal{N}(0,\,10) \\
\sigma^2 \sim  |\mathcal{N}(0,\,10)|
$ 

We will artificially create the data to predict on. We will then see if our model predicts them correctly.