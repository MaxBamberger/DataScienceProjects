import matplotlib.pyplot as plt

def likelihood_func(data,key):
    key = float(key)
    data = float(data)
    faces = list(range(int(key)+1))[1:]
    if data in faces:
        return 1/key
    else: return 0

class Bayes(object):
    '''
    INPUT:
        prior (dict): key is the value (e.g. 4-sided die),
                      value is the probability

        likelihood_func (function): takes a new piece of data and the value and
                                    outputs the likelihood of getting that data
    '''
    def __init__(self, prior, likelihood_func):
        self.prior = prior
        self.likelihood_func = likelihood_func
        self.posterior = prior

    def normalize(self):
        '''
        INPUT: None
        OUTPUT: None

        Makes the sum of the probabilities equal 1.
        '''
        self.normalizer = sum([v*self.likelihood_func(self.data,k) for k,v in self.prior.items()])

    def update(self, data):
        '''
        INPUT:
            data (int or str): A single observation (data point)

        OUTPUT: None

        Conduct a bayesian update. Multiply the prior by the likelihood and
        make this the new prior.
        '''

        self.data = data
        self.posterior = {}
        self.normalize()
        for key, value in self.prior.items():
            self.posterior[key] = self.likelihood_func(data, key)*self.prior[key] / self.normalizer
            self.prior[key] = self.posterior[key]


    def print_distribution(self):
        '''
        Print the current posterior probability.
        '''
        for key, value in sorted(self.posterior.items()):
            print("{} : {}".format(key, value))

    def plot(self, ax, color=None, title=None, label=None):
        '''
        Plot the current prior.
        '''
        ax.plot(list(self.prior.keys()), list(self.prior.values()),color=color,label=label)
        ax.set_title(title)

if __name__ == '__main__':
    prior = {4:0.2, 6:0.2, 8:0.2, 12:0.2, 20:0.2}
    bayes = Bayes(prior.copy(),likelihood_func=likelihood_func)
    # print(likelihood_func('8','8'))
    bayes.print_distribution()
    print('\n')
    bayes.update(8)
    bayes.print_distribution()
    # print('\n')
    # bayes.update(1)
    # bayes.print_distribution()
    # print('\n')
    # bayes.update(19)
    # bayes.print_distribution()
    # print(prior)
