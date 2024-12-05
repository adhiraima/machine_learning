from enum import Enum
import random
import string

class IDGenerator:
    @staticmethod
    def id_generator(len:int=8):
        seed = string.ascii_lowercase + string.ascii_uppercase + string.digits
        return "".join(random.choice(seed) for i in range(len))

class Suits(Enum):
    SPADE = "\u2660"
    HEART = "\u2665"
    DIAMOND = "\u2666"
    CLUB = "\u2663"

class FaceVal(Enum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"

class GameState(Enum):


class Card:
    def __init__(self, suit: Suits, face_value: FaceVal):
        self.suit = suit
        self.face_value = face_value
        # print(f"{self.suit} {self.face_value}")
        # print(self)

    def __repr__(self):
        return f"{self.face_value.value} {self.suit.value}"


class Deck:
    def __init__(self):
        self.cards = []
        self._create_deck()
    
    def _create_deck(self):
        for suit in Suits:
            for face_val in FaceVal:
                card = Card(suit=suit, face_value=face_val)
                self.cards.append(card)

    def shuffle(self, num_shuffles:int = 3):
        for _ in range(num_shuffles):
            random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop()

    def __repr__(self):
        return  f"{self.cards}" 

class Player:
    def __init__(self, handle: str):
        self.handle = handle
        self.cards = []
    
    def deal_card(self, card: Card):
        if len(self.cards) > 2:
            raise Exception("No more than 2 cards can be dealt")
        
class Table:
    def __init__(self, id, capacity:int=6):
        self.id = id
        self.players = []
        self.games = {}
        self.capacity = capacity

    def add_player(self, player: Player):
        if len(self.players) > self.capacity:
            raise Exception("Number of player exeeded the table capacity")
        self.players.append(player)

    def start_game(self):
        game_id = IDGenerator.id_generator(len=4)
        while game_id in self.games.keys():
            game_id = IDGenerator.id_generator(len=4)
        game = Game(id=game_id)
        self.games[game_id] = game
        self.play_game(game_id)
    
    def play_game(self, game_id: str):
        pass

class Game:
    def __init__(self, id: str, blind: int = 0, ante: int = 0):
        self.id = id 
        self.deck = Deck()
        self.burn_cards = []
        self.flop = []
        self.turn = None
        self.river = None
        self.fold_pile = []
        self.small_blind = blind/2
        self.big_bling = blind
        self.curret_state = "Started"


    def deal(self, players):
        self.deck.shuffle()
        if len(players) < 2:
            raise Exception("Need minimum 2 players to start")
        for player in players:
            player.deal_card(self.deck.draw())
        for player in players:
            player.deal_card(self.deck.draw())
    
    def _burn(self):
        self.burn_cards.append(self.deck.draw())

    def flop(self):
        self._burn()
        for i in range(3):
            self.flop.append(self.deck.draw())

    def turn(self):
        self._burn()
        self.turn = self.deck.draw()

    def river(self):
        self._brun()
        self.river = self.deck.draw()

def main():
    deck1 = Deck()
    deck2 = Deck()
    # print(deck)
    # print(f"{Suits.SPADE.value}, {FaceVal.ACE.value}")
    deck1.shuffle()
    print(deck1)
    deck2.shuffle()
    print(deck2)



if __name__ == "__main__":
    main()

