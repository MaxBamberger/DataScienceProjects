## Part 0: Introduction to Bayesian Analysis

This following formula is the underpinning of all Bayesian Analysis.
It is **very important** to understand what each of the terms are. Do not
move on until you have read and understood this section.

![bayes formula](images/bayes_formula.png)

**Prior Probability**:
- A PMF / PDF representing your initial beliefs about the parameter(s).  
- The initial belief is less represented in the posterior as more data is incorporated

**Likelihood**:
- The probability of observing the data given the parameter(s)
- i.e. What is the likelihood of 3 Heads in a row given the probability of heads is 0.7?

**Posterior Probability**:
- The product of prior and likelihood (Bayesian-update)
- The posterior probability becomes the prior of the next Bayesian-update

**Normalizing Constant**:
- The probability of observing the data. 
- In Bayesian analysis, this term ensures the sum of all probabilities is 1

## Part 1: Bayesian Analysis (Discrete example)

Include your answers in `pair_answers.md`

We're going to start with a discrete example.

A box contains a 4-sided die, a 6-sided die, an 8-sided die,
a 12-sided die, and a 20-sided die. A die is selected at random, and the
rest are destroyed.

We would like to determine which die I have selected, given only information of what I roll.

You should write the solutions to these in a text or markdown file.

1. What is the prior associated with choosing any one die?

2. What is the likelihood function? You should assume that the die are all fair.

3. Say I roll an 8. After one bayesian update, what is the probability that I chose each of the dice?

4. Comment on the difference in the posteriors if I had rolled the die 50 times instead of 1.

5. Which one of these two sets of data gives you a more certain posterior and why?
`[1, 1, 1, 3, 1, 2]` or `[10, 10, 10, 10, 8, 8]`

6. Say that I modify my prior by my belief that bigger dice are more likely to be drawn from the box. This is my prior distribution:

    ```
    4-sided die: 8%
    6-sided die: 12%
    8-sided die: 16%
    12-sided die: 24%
    20-sided die: 40%
    ```

    What are my posteriors for each die after rolling the 8?

    Which die do we think is most likely? Is this different than what you got with the previous prior?

7. Say you keep the same prior and you roll the die 50 times and get values 1-8 every time. What would you expect of the posterior? How different do you think it would be if you'd used the uniform prior?


## Part 2: Object Oriented Bayes

Let's verify the answers you made in the previous question by writing some code to simulate the situation.

We will be implementing a `Bayes` class that is able to handle Bayesian updates in the discrete case.
A prior is defined and at each data point, a likelihood is computed and the
prior is updated to give the posterior.

**The posterior becomes the prior at the next data point.**

The starter code is provided for you in `bayes.py`.

1. Fill in the `normalize` method. It makes the probabilities in `self.prior` sum to 1.

   To normalize, you need to sum up all the probabilities and then divide each probability by this value.

2. Fill in the `update` method. It should take the result of one test, calculate the likelihood and use it to calculate the posterior. The posterior should now be your prior.

    The likelihood function should be used like this:

    ```python
    self.likelihood_func(data, key)
    ```

    Make sure to call the `normalize` method!

3. Fill in the `print_distribution` method in order to print out the current prior.

    It will be easier to read if you print the values in sorted order so you'll always see them in a consistent order.

## Part 3: Verifying via simulation

Now we're ready to use our code to verify our intuition about the dice problem. You'll need to run several simulations using your bayes code and verify that it matches what you said in Part 1.

We are going to consider both a uniform prior and the unbalanced prior from above:

```
4-sided die: 8%
6-sided die: 12%
8-sided die: 16%
12-sided die: 24%
20-sided die: 40%
```

Put all the code you use to answer these questions in `dice.py`.

1. Encode both priors as dictionaries.

    ***Note: When you pass these into your bayes function make sure to use `.copy()` or else you will be editing your prior and won't be able to use it in a following problem.***

2. Implement the likelihood function. It should take the data (in this case a number 1-20) and the die number (4, 6, 8, 12 or 20).

3. First, let's examine the case where we only roll the die once. Say we roll a single 8.

    * What are the posteriors if we started with the uniform prior?
    
    * What are the posteriors if we started with the unbalanced prior?
    
    * How different were these two posteriors?

4. Let's say you get all this data:

    ```
    [8,2,1,2,5,8,2,4,3,7,6,5,1,6,2,5,8,8,5,3,4,2,4,3,8,
     8,7,8,8,8,5,5,1,3,8,7,8,5,2,5,1,4,1,2,1,3,1,3,1,5]
    ```

    * What are the posteriors if we started with the uniform prior?
    
    * What are the posteriors if we started with the unbalanced prior?
    
    * How different are these two posteriors?    

5. With the uniform prior, which of these two sets of data leads to a more certain posterior? `[1, 1, 1, 3, 1, 2]` or `[10, 10, 10, 10, 8, 8]`


## Part 4: A Biased Coin (a continuous example)

We have a coin. We would like to know how biased it is. The bias is a value between 0 and 1 of the probability of flipping heads. Our prior is that all biases are equally likely.

Even though this is a continuous example, we are going to simulate it discretely so that we can use the same code from Part 2. We are going to get the values of the pmf for 100 values from 0 to 1 and then look at the graphs.

Note that when we plot this graph, the x-axis won't have the correct labels since our bayes code is for the discrete case. However, the shapes will still be correct.

Put your code in `biased_coin.py`.

1. Create the prior dictionary that has all the keys in `0, 0.01, 0.02, ..., 0.99`. The values should all be the same, as an equal probability of each of these keys occurring. Technically, each of these corresponds to a range in the pmf, not a specific value.

    You can use `np.linspace` to get the array of all the keys.

2. The likelihood function is a Bernoulli. Write the `likelihood` function. It should take the data of either `'H'` or `'T'` and the value for `p` and return a value between 0 and 1.

    For example, if the data is `H` and the value is 0.3, it should return 0.3.

    If the data is `T` and the value is 0.3, it should return 0.7.

3. Make a graph with 8 subplots that has the posterior for each of the following scenarios. Make sure to give each graph a title!

    * You get the data: `H`.

    * You get the data: `T`.
    
    * You get the data: `H, H`.

    * You get the data: `T, H`.
    
    * You get the data: `H, H, H`.

    * You get the data: `T, H, T`.
    
    * You get the data: `H, H, H, H`.

    * You get the data: `T, H, T, H`.

    You shouldn't need to edit the `plot` method to do this. Just change the subplot by calling `plt.subplot(...)` first to switch which subplot it will draw on.

    Use `plt.tight_layout()` to make the spacing of the subplots nicer.
    
    Note how the graph changes as you get more data. With more data you should become more certain of the value.

4. There is a biased coin in `coin.pyc`. You can't see the code to see how biased it is. You can use the coin like this:

    ```python
    from coin import Coin

    mycoin = Coin()
    print mycoin.flip()
    print mycoin.flip()
    ```

    The `flip` method will return either `H` or `T`

    On a single graph, overlay the initial uniform prior with the prior after 1, 2, 10, 50 and 250 flips..

    Use the `color` parameter to give a different color to each layer. Use the `label` parameter to label each label.

This simulation gives us the [***Beta Distribution***](http://en.wikipedia.org/wiki/Beta_distribution)! The shape parameters (alpha and beta) are `1 + # heads` and `1 + # tails`! We'll get into this more later.

## Extra Credit: Simulation

Implementing a simulation can help you gain insight to what you have verified by hand. We'd like to verify that switching is the best strategy for the Monty Hall problem.

If you're unfamiliar with the Monty Hall problem, here's a quick explanation:

Basically, you're on a game show where there are three doors. One door has a prize and two have nothing. You get to pick a door. Then Monty (the show's host) reveals that one of the other two doors doesn't have the prize. Should you keep the same door or should you switch?

1. There are three strategies we'd like to consider:

    1. Stay with your door
    2. Switch doors
    3. Randomly choose a door
    
    Which strategy will perform the best? (This is a simple bayes problem.)

2. Let's verify this with a simulation. Fill in the python code in `monty.py` and
   `playmonty.py`. You will implement the three strategies.

    You can run your code with this command, where 1000 is the number of times
    your simulator will play the Monty Hall game:

    ```python playmonty.py 1000```
