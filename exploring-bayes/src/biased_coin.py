import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from bayes import Bayes
from coin import Coin

'''Create the prior dictionary that has all the keys in 0, 0.01, 0.02, ..., 0.99.
The values should all be the same, as an equal probability of each of these keys occurring.
Technically, each of these corresponds to a range in the pmf, not a specific value.'''

prior_dict = dict()
biases = np.linspace(0,.99,num=100)
for k in biases:
    prior_dict[k] = 1/len(biases)

'''The likelihood function is a Bernoulli. Write the likelihood function.
It should take the data of either 'H' or 'T' and the value for p and return a value between 0 and 1.'''

def likelihood(val, p):
    '''likelihood_func (function):
    takes a new piece of data and the value and
    outputs the likelihood of getting that data'''
    if val == 'H':
        return p
    elif val == 'T':
        return 1-p

'''Make a graph with 8 subplots that has the posterior for each of the following scenarios.
   Make sure to give each graph a title!
        * You get the data: H
        * You get the data: T
        * You get the data: H, H
        * You get the data: T, H
        * You get the data: H, H, H
        * You get the data: T, H, T
        * You get the data: H, H, H, H
        * You get the data: T, H, T, H'''

bayes_uniform = Bayes(prior_dict.copy(),likelihood_func=likelihood)
bayes_uniform.update('H')

fig, axs = plt.subplots(4,2, figsize=(14,8))
scenarios = ['H','T',['H','H'],['T','H'],['H','H','H'],['T','H','T'],['H','H','H','H'],['T','H','T','H']]
i = 1
for scenario, ax in zip(scenarios, axs.flatten()):
    bayes = Bayes(prior_dict.copy(),likelihood_func=likelihood)
    for flip in scenario:
        bayes.update(flip)
    bayes.plot(ax,title='Scenario #'+str(i)+': '+ ', '.join(scenario))
    i+=1
plt.tight_layout()
plt.show()

'''On a single graph, Use the coin.py random coin generator and overlay the initial uniform prior with the prior after 1, 2, 10, 50 and 250 flips..
Use the color parameter to give a different color to each layer. Use the label parameter to label each label.
This simulation gives us the Beta Distribution! The shape parameters (alpha and beta) are 1 + # heads and 1 + # tails!
We'll get into this more later.'''

mycoin = Coin()
# print(mycoin.flip())
# print(mycoin.flip())

fig, ax = plt.subplots(1,1, figsize=(14,9))
scenarios = []
num_flips = [1,2,10,50,250]
colors = ['red','green','blue','orange','purple']
for flips in num_flips:
    scenario = [mycoin.flip() for flip in range(flips)]
    scenarios.append(scenario)
bayes = Bayes(prior_dict.copy(),likelihood_func=likelihood)
bayes.plot(ax,color='black',label='Number of flips: 0')
for flips, color, scenario in zip(num_flips, colors, scenarios):
    bayes = Bayes(prior_dict.copy(),likelihood_func=likelihood)
    for flip in scenario:
        bayes.update(flip)
    bayes.plot(ax,color=color,label='Number of flips: '+str(flips),title='Beta Distribution for Unknown Biased Coin')

plt.legend()
plt.show()
