#!python3
from tic_tac_toe import TicTacToe
import tensorflow as tf
from random import randint, random, seed

OUT_FILE = "out.log"

class TicTacToePlay:
    """
    Evaluation of a players for playing tic tac toe
    """
    file_cntr = 0

    def __init__(self, grid_size_x, grid_size_y, player_X, player_O, win_strike_length=4, print_chance=60):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.game = TicTacToe(self.grid_size_x, self.grid_size_y)
        self.win_strike_length = win_strike_length
        self.print_chance = print_chance
        self.player_codes = [self.game.F_X, self.game.F_O]
        self.players = [player_X, player_O]
        self.game_history = [[], []]
        self.result = 0

    def play(self, verbosity=0):
        self.game = TicTacToe(self.grid_size_x, self.grid_size_y)
        self.game_history = [[], []]
        
        current_player = 0
        self.result = 0

        while (self.result == 0):
            if (verbosity >= 2):
                print(self.game)
            self.game_history[current_player].append(self.game.copy())

            player_map = self.game.get_player_map(self.player_codes[current_player])
            turn_x, turn_y = self.players[current_player].getNextTurn(player_map)
            self.game.play(turn_x, turn_y, self.player_codes[current_player])
            self.result = self.game.evaluate_win(self.win_strike_length)

            current_player = 1 - current_player

        if (verbosity >= 1):
            print(self.game)

        if (randint(0, 99) < self.print_chance):
            with open(OUT_FILE + "_" + str(TicTacToePlay.file_cntr), "w") as F:
                F.write(f"Game result: {self.getResultDescription()}\n")
                F.write(str(self.game))
        TicTacToePlay.file_cntr += 1
        TicTacToePlay.file_cntr %= 3

        return self.result

    def getResultDescription(self):
        descs = {
            -2 : "invalid state",
            -1 : "draw",
             0 : "unfinished",
             1 : "Player X wins",
             2 : "Player O wins",
        }
        return descs[self.result]

    def accumulate(self, other):
        self.result = 1
        for i in range(len(self.game_history)):
            e = i
            if (other.result != self.result):
                # If the results were different, switch the players' histories
                e = 1 - i
            self.game_history[i] += other.game_history[e]

    def copy(self):
        new = TicTacToePlay(self.grid_size_x, self.grid_size_y, self.players[0], self.players[1], self.win_strike_length)
        new.game = self.game.copy()
        new.game_history = [x.copy() for x in self.game_history]
        new.result = self.result
        return new
