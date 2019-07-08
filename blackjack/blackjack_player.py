import random

class Player(object):
        '''
        Initializes the player object. Blackjack is played between one human player (user)
        and automated Dealer player

        Parameters
        ----------
        name: str
            player's name
        money: int
            amount of money the user player has come to play with
        '''
    def __init__(self, name, money):
        self.name = name
        self.hand = []
        self.money = money
        self.hand_sum = 0

    def receive_card(self, card):
        '''
        Initializes the player's hand with one card.
        Updates hand_sum attribute with the current total in hand for the player

        Parameters
        ----------
        card: object

        Returns
        ----------
        None
        '''
        self.hand.append(card)
        self.hand_sum = sum([card.value_dict[card.number] for card in self.hand])

    def display_hand(self):
        '''
        Prints the current hand and hand total.
        Determines which kind of ace to use.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        print(self.name + "'s cards: " + str(self.hand))
        if self.hand_sum > 21 and self.check_aces:
            for card in self.hand:
                if card.is_ace:
                    card.number = 'A_1'
                    card.value_dict['A_1'] = 1
                    self.hand_sum = sum([card.value_dict[card.number] for card in self.hand])
                    if self.hand_sum <= 21: break
        print(self.name + "'s cards total to " + str(self.hand_sum))


    def check_aces(self):
        ## Checks for aces in the hand
        if any([card.is_ace for card in self.hand]):
            return True
        else: return False

    def lose_money(self,bet):
        ## Subtracts money from the players balance based on how much was bet
        self.money -= bet

    def win_money(self,bet):
        ## Adds money from the players balance based on how much was bet
        self.money += bet

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.hand)
