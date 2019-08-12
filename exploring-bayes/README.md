## Simulating Bayesian Inference
I created a few simulations to demonstrate Bayesian Inference
- The trade-offs between the power of a test, the significance,
  sample size and detectable effect size
- Bayesian updating to calculate outcomes as a probabilistic distribution (posterior) 
  based on a prior distribution and a likelihood function

To Demonstrate how Bayesian Updating works, I created two object oriented simulations:
 - [bayes.py](https://github.com/MaxBamberger/DataScienceProjects/blob/master/exploring-bayes/src/bayes.py). The `Bayes` class that is able to handle Bayesian updates in the discrete case. A prior is defined and at each data point, a likelihood is computed and the prior is updated to give the posterior. You can play around with priors and see how it affects the posterior outcome by using `src/dice.py`.
 - [biased_coin.py](https://github.com/MaxBamberger/DataScienceProjects/blob/master/exploring-bayes/src/biased_coin.py). We have a coin. We would like to know how biased it is. The bias is a value between 0 and 1 of the probability of flipping heads. Our prior is that all biases are equally likely.

## Bayesian Power for A/B testing
In [Bayesian Power](https://github.com/MaxBamberger/DataScienceProjects/blob/master/exploring-bayes/bayes-test.ipynb) I demonstrate how you can use Bayes to approach an A/B hypothesis test in a different way...
