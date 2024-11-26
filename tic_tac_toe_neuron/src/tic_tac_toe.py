#!python3

class TicTacToe:
    """
    Tic Tac Toe game mechanism
    """
    F_EMPTY = 0
    F_X     = 1
    F_O     = 2

    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.grid = [[self.F_EMPTY for _ in range(self.size_x)] for I in range(self.size_y)]

    def _cl_name(self):
        return self.__class__.__qualname__

    def __getitem__(self, y):
        return self.grid[y]

    def __setitem__(self, y, v):
        self.grid[y] = v

    def play(self, x, y, player):
        assert (x >= 0 and y >= 0 and x < self.size_x and y < self.size_y), f"{self._cl_name()}: Coordinates x:{x} y:{y} are out of bounds of grid size x:{self.size_x} y:{self.size_y}"
        assert (player in [self.F_X, self.F_O]), f"{self._cl_name()}: Unknown player code {player}"
        assert (self[y][x] == self.F_EMPTY), f"{self._cl_name()}: Coordinates x:{x} y:{y} are not empty"
        self[y][x] = player

    def copy(self):
        t = type(self)
        c = t(self.size_x, self.size_y)
        c.grid = [l.copy() for l in self.grid]
        return c

    def __str__(self):
        field_map = {
            self.F_EMPTY : " ",
            self.F_X     : "X",
            self.F_O     : "O",
        }

        out = ""
        for i,line in enumerate(self.grid):
            for e,field in enumerate(line):
                out += field_map[field]
                if (e != len(line)-1):
                    out += " | "
            out += "\n"
            if (i != len(self.grid)-1):
                out += "-" * ((len(line)-1) * 4 + 1) + "\n"

        return out

    @classmethod
    def fromString(cls, string):
        """
        Create game from string in a format like this:
        X...X...
        XOO.....
        OX...OOO
        ...XXOOO
        """
        lines = string.split("\n")
        size_y = len(lines)
        size_x = len(lines[0])
        result = cls(size_x, size_y)
        for i,l in enumerate(lines):
            assert (len(l) == size_x), f"Invalid input string format. Line {i} has length {len(l)} which is different from line 0  length {size_x}."
            for e,c in enumerate(l):
                if (c == "X"):
                    result.play(e, i, cls.F_X)
                elif (c == "O"):
                    result.play(e, i, cls.F_O)
                else:
                    assert (c == "."), f"Invalid input string format. Unrecognized character '{c}'. Only characters 'X', 'O' and '.' are supported."
        return result

    def get_player_map(self, player):
        """
        Returns a version of the map grid where fields are encoded in the following way:
        0 - empty field
        1 - field belonging to the player
        2 - field belonging to the other player
        """
        assert (player in [self.F_X, self.F_O]), f"{self._cl_name()}: Unknown player coed {player}"
        other_player = self.F_X if (player == self.F_O) else self.F_O
        field_map = {
            self.F_EMPTY : 0,
            player       : 1,
            other_player : 2,
        }
        return [[field_map[f] for f in line] for line in self.grid]

    def evaluate_win(self, win_strike_length=5):
        """
        Evaluate whether the current state of the game is a final one.
        
        Parameters:
        win_strike_length - number of Xs or Os in a row to constitute a win

        Return values:
        0 - not a final state
        self.F_X - win for player X
        self.F_O - win for player O
        -1 - draw (all fields taken and nobody won)
        -2 - invalid state (both players have a win strike)
        """
        win_x = False
        win_o = False
        empty_space_left = False
        
        # A winning strike can happen in one of 4 directions
        # 1. in one row
        # 2. in one column
        # 3. diagonal to the right bottom
        # 4. diagonal to the left bottom
        for direction in [[1,0], [0,1], [1,1], [-1, 1]]:
            for start_y in range(self.size_y):
                for start_x in range(self.size_x):
                    if (self[start_y][start_x] == self.F_EMPTY):
                        empty_space_left = True

                    w_x = True
                    w_o = True
                    x = start_x
                    y = start_y
                    out_of_bounds = False

                    for step in range(win_strike_length):
                        if (x < 0 or y < 0 or x >= self.size_x or y >= self.size_y):
                            out_of_bounds = True
                            break
                        if (self[y][x] != self.F_X):
                            w_x = False
                        if (self[y][x] != self.F_O):
                            w_o = False
                        x += direction[0]
                        y += direction[1]

                    if (out_of_bounds):
                        continue

                    if (w_x):
                        win_x = True
                    if (w_o):
                        win_o = True

        if (win_x and win_o):
            return -2
        if (win_x):
            return self.F_X
        if (win_o):
            return self.F_O
        if (not empty_space_left):
            return -1
        return 0