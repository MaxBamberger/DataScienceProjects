from bj_deck import Deck
from blackjack_player import Player
import sys

class Blackjack(object):
    '''
    Initializes the game object. Blackjack is played between one human player (user)
    and automated Dealer player

    Parameters
    ----------
    None

    '''

    def __init__(self):
        self.player = self.create_player()
        self.dealer = Player('DEALER',1)
        self.winner = None
        self.is_blackjack = False

    def play_game(self):
        '''
        Primary method of game object.
        When called will executes the game from start to finish

        Parameters
        ----------
        self

        Returns
        ----------
        None
        '''
        while self.player.money > 0:
            roundStatus = self.play_round()
            if roundStatus == 'exit':
                print(self.player.name + " walks away with a balance of $"+str(self.player.money))
                print('Goodbye!')
                return None
            self.reset()
            print(self.player.name + " has a balance of $"+str(self.player.money))
        print(self.player.name + " is broke! :-( GAME OVER")


    def create_player(self):
        '''
        Called at the beginning of the game to initalize a Player
        object class found in the blackjack_player.py

        Parameters
        ----------
        self

        Returns
        ----------
        Player: object
        '''

        msg = "What's your name? "
        name = str(raw_input(msg) if sys.version_info[0] < 3 else input(msg))
        msg = "How much $$ did you come to play with? "
        money = raw_input(msg) if sys.version_info[0] < 3 else input(msg)
        while type(money) != int:
            try:
                money = int(money)
                continue
            except:
                money = raw_input("Type an integer number only. " + msg) if sys.version_info[0] < 3 else input("Type an integer number only. " + msg)

        return Player(name, int(money))

    def reset(self):
        '''
        Resets game object Attributes After each round

        Parameters
        ----------
        self

        Returns
        ----------
        None
        '''
        self.winner = None
        self.is_blackjack = False

    def deal(self):
        '''
        Called on each round, the deal method prepares the Deck object,
        creates hands for each player and places the bets

        Parameters
        ----------
        self

        Returns
        ----------
        None or 'exit'
        '''

        self.player.hand = []
        self.dealer.hand = []
        self.deck = Deck()
        self.deck.shuffle()
        msg = "What will you bet on this hand (type `exit` to exit game)? "
        bet = raw_input(msg) if sys.version_info[0] < 3 else input(msg)
        if bet == 'exit': return 'exit'
        while type(bet) != int:
            try:
                bet = int(bet)
                continue
            except:
                bet = raw_input("Type an integer number or `exit` only. " + msg) if sys.version_info[0] < 3 else input("Type an integer number or `exit` only. " + msg)

        while bet > self.player.money:
            print('Cannot bet more money than you have..')
            msg = "Whatcha wanna bet on this hand? "
            bet = int(raw_input(msg) if sys.version_info[0] < 3 else input(msg))
        self.bet = bet

    def play_round(self):
        '''
        The main function that governs the actions and logic of each round

        Parameters
        ----------
        self

        Returns
        ----------
        None or 'exit'
        '''

        #deals
        dealStatus = self.deal()
        if dealStatus == 'exit': return 'exit'

        # Deal 2 cards to each player:
        self.player.receive_card(self.deck.draw_card())
        self.player.receive_card(self.deck.draw_card())
        self.dealer.receive_card(self.deck.draw_card())
        self.dealer.receive_card(self.deck.draw_card())

        # Show one dealer card:
        self.player.display_hand()
        print('Dealer shows ' + str(self.dealer.hand[0]))
        if self.dealer.hand[0].is_blackjack:
            self.is_blackjack = True
            self.winner = self.dealer
            self.evaluate_hand(self.dealer,self.player)

        # Evaluate if there is a winner:
        self.evaluate_hand(self.player,self.dealer)

        # Player actions / hit or stay decisions:
        if self.winner is None:
            msg = 'To Hit press `h`, to Stay press `s`: '
            incorrect_msg = 'Incorrect key. '+ msg
            decision = raw_input(msg) if sys.version_info[0] < 3 else str(input(msg))
            while decision.lower() != 'h' and decision.lower() != 's':
                decision = raw_input(incorrect_msg) if sys.version_info[0] < 3 else str(input(incorrect_msg))
            while decision == 'h':
                new_card = self.deck.draw_card()
                self.player.receive_card(new_card)
                print(self.player.name + " is dealt a: "+str(new_card))
                self.player.display_hand()
                self.evaluate_hand(self.player,self.dealer)
                if self.winner: break
                decision = raw_input(msg) if sys.version_info[0] < 3 else str(input(msg))
                while decision.lower() != 'h' and decision.lower() != 's':
                    decision = raw_input(incorrect_msg) if sys.version_info[0] < 3 else str(input(incorrect_msg))

        # If player decides to stay, and still no winner, execute dealer play:
        if self.winner is None: self.dealer_play()

        # Once dealer play has settled and still no winner - evaluate hands side by side:
        if self.winner is None: self.compare()

    def dealer_play(self):
        '''
        Logic governing how the dealer is to draw cards for himself

        Parameters
        ----------
        self

        Returns
        ----------
        None
        '''
        self.dealer.display_hand()
        while self.dealer.hand_sum <= 16:
            print("Dealers hits...")
            new_card=self.deck.draw_card()
            self.dealer.receive_card(new_card)
            print("Dealer is dealt a: "+str(new_card))
            self.dealer.display_hand()
            self.evaluate_hand(self.dealer,self.player)
        if self.winner is None: print("Dealer stays..")

    def compare(self):
        '''
        Evaluate the player's total compared to the dealers.
        Update winner attribute

        Parameters
        ----------
        self

        Returns
        ----------
        None
        '''
        if self.player.hand_sum > self.dealer.hand_sum:
            self.winner = self.player
        elif self.player.hand_sum < self.dealer.hand_sum:
            self.winner = self.dealer
        elif self.player.hand_sum == self.dealer.hand_sum:
            print("Tie game, money is returned and hand is reset..")
        if self.winner: self.win_lose()


    def evaluate_hand(self, actor, other):
        '''
        Evaluate the player's hand and if it contains a blackjack, is greater than
        21 (bust) or 21 exactly

        Parameters
        ----------
        self

        Returns
        ----------
        None
        '''
        if actor.hand_sum == 21:
            print(actor.name + " GOT 21!")
            self.winner = actor
        elif actor.hand_sum > 21:
            print(actor.name + " BUSTS!")
            self.winner = other
        elif any([card.is_blackjack for card in actor.hand]):
            self.winner = actor
            self.is_blackjack = True

        if self.winner: self.win_lose()

    def win_lose(self):
        '''
        Method when called displays the winner / loser and how much money
        is to change hands

        Parameters
        ----------
        self

        Returns
        ----------
        None
        '''
        if self.is_blackjack:
            if self.winner == self.player:
                self.player.win_money((3/2)*self.bet)
                print(self.player.name + " GOT Blackjack! " + self.player.name + " WINS!")
                print(self.player.name + " WINS $" + str((3/2)*self.bet))
            else:
                self.player.lose_money((3/2)*self.bet)
                print(self.dealer.name + " GOT Blackjack! " + self.dealer.name + " WINS!")
                print(self.player.name + " LOSES $" + str((3/2)*self.bet))
        else:
            if self.winner == self.player:
                self.player.win_money(self.bet)
                print(self.player.name + " WINS $" + str(self.bet))
            else:
                self.player.lose_money(self.bet)
                print(self.player.name + " LOSES $" + str(self.bet))



if __name__ == '__main__':
    game = Blackjack()
    game.play_game()
