from numpy.random import rand

class Coin:
    '''
    A Coin with a random weight
    '''
    def __init__(self):
        self.prob = .1+.8*rand()

    def flip(self):
        '''
        flip - get the result of flipping the coint
        Outputs (str) -> either "H" or "T" with probability from the initialized
        variable
        '''
        return "H" if rand() > self.prob else "T"
