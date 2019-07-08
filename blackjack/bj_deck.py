import random

class Card(object):
    '''
    Card object used for evaluating each card in a hand.
    Instantiated by the Deck Object

    Parameters
    ----------
    number: str
        Number of the card. Can be 'A', '2', '3', '4', '5', '6', '7',
                                    '8', '9', '10', 'J', 'Q', 'K'
    suit: str
        Suit of the card. Can be 'c','d','h', or 's'
    '''

    value_dict = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                  '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10,
                  'K': 10, 'A': 11}

    def __init__(self, number, suit):
        self.suit = suit
        self.number = number
        self.is_blackjack = (self.suit in 'cs') and (self.number == 'J')
        self.is_ace = (self.number == 'A')

    def __repr__(self):
        return "%s%s" % (self.number, self.suit)

    def __gt__(self, other):
      return self.value_dict[self.number] > self.value_dict[other.number]

    def __lt__(self, other):
      return self.value_dict[self.number] < self.value_dict[other.number]

    def __eq__(self, other):
      return self.value_dict[self.number] == self.value_dict[other.number]


class Deck(object):
    '''
    Called each blackjack round, Deck class initializes the deck object. 
    Creates a new deck of card objects

    Parameters
    ----------
    None
    '''
    def __init__(self):
        self.cards = []
        for num in Card.value_dict.keys():
            for suit in 'cdhs':
                self.cards.append(Card(num, suit))

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if not self.isempty():
            return self.cards.pop()

    def add_cards(self, cards):
        self.cards.extend(cards)

    def __len__(self):
        return len(self.cards)

    def isempty(self):
        return self.cards == []
