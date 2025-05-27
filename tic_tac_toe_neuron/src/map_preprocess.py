#!python3
from tic_tac_toe import TicTacToe
import tensorflow as tf
from random import randint, random, seed

OUT_FILE = "out.log"

class MapVariation:
    def __init__(self, pad_start_x, pad_start_y, coord_switch, mirror_x, mirror_y):
        self.slice_start_x = 0
        self.slice_start_y = 0
        self.size_x = 0
        self.size_y = 0
        self.pad_start_x = pad_start_x
        self.pad_start_y = pad_start_y
        self.coord_switch = coord_switch
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y

    all_variations_dir = {}

    @classmethod
    def getRandomVariations(cls, max_num, game_map_size, cnn_input_size):
        if ((game_map_size, cnn_input_size) in cls.all_variations_dir):
            all_variations = cls.all_variations_dir[(game_map_size, cnn_input_size)]
        else:
            possible_pads = cnn_input_size - game_map_size + 1
            all_variations = []
            for i in range(possible_pads):
                for e in range(possible_pads):
                    for s in [True, False]:
                        for mx in [True, False]:
                            for my in [True, False]:
                                all_variations.append(cls(i, e, s, mx, my))
            cls.all_variations_dir[(game_map_size, cnn_input_size)] = all_variations

        all_variations = all_variations.copy()

        if (max_num >= len(all_variations)):
            return all_variations

        return [all_variations.pop(randint(0, len(all_variations)-1)) for i in range(max_num)]

    def getResizedMap(self, game_map, target_grid_size):
        self.size_x = len(game_map[0])
        self.size_y = len(game_map)
        map_mirrored = mapMirror(game_map, self.mirror_x, self.mirror_y)
        map_resized, self.slice_start_x, self.slice_start_y = mapResize(map_mirrored, target_grid_size, self.pad_start_x, self.pad_start_y, self.coord_switch)
        return map_resized

    def gameToCnn(self, x, y):
        if (self.mirror_x):
            x = self.size_x - x - 1
        if (self.mirror_y):
            y = self.size_y - y - 1
        if (self.coord_switch):
            tmp = x
            x = y
            y = tmp
        return x - self.slice_start_x + self.pad_start_x, y - self.slice_start_y + self.pad_start_y

    def cnnToGame(self, x, y):
        x = x + self.slice_start_x - self.pad_start_x
        y = y + self.slice_start_y - self.pad_start_y
        if (self.coord_switch):
            tmp = x
            x = y
            y = tmp
        if (self.mirror_x):
            x = self.size_x - x - 1
        if (self.mirror_y):
            y = self.size_y - y - 1
        return x, y

    def __str__(self):
        return f"slice_start_x: {self.slice_start_x}, slice_start_y: {self.slice_start_y}, start_x: {self.pad_start_x}, start_y: {self.pad_start_y}, coord_switch: {self.coord_switch}, mirror_x: {self.mirror_x}, mirror_y: {self.mirror_y}"

def mapResize(game_map, target_grid_size, pad_start_x = None, pad_start_y = None, coord_switch = False):
    if (coord_switch):
        game_map = [[game_map[y][x] for y in range(len(game_map))] for x in range(len(game_map[0]))]

    if (len(game_map) == target_grid_size):
        return game_map, 0, 0

    map_resized = [line.copy() for line in game_map]
    slice_start_x = 0
    slice_start_y = 0

    if (len(map_resized) < target_grid_size):
        # If map is too small -> pad with empty fields (at random offset)
        required_pad = target_grid_size - len(map_resized)
        if (pad_start_x == None):
            pad_start_x = randint(0, required_pad)
        if (pad_start_y == None):
            pad_start_y = randint(0, required_pad)
        pad_end_x = required_pad - pad_start_x
        pad_end_y = required_pad - pad_start_y
        for i in range(len(map_resized)):
            map_resized[i] = [0] * pad_start_x + map_resized[i] + [0] * pad_end_x
        map_resized = [[0] * target_grid_size] * pad_start_y + map_resized + [[0] * target_grid_size] * pad_end_y

    if (len(map_resized) > target_grid_size):
        # If map is too large -> find median of non-empty field coordinates and slice around it
        used_xs = []
        used_ys = []
        for y,line in enumerate(map_resized):
            for x,field in enumerate(line):
                if (field != 0):
                    used_xs.append(x)
                    used_ys.append(y)

        if (len(used_xs) == 0):
            # Map empty -> slice randomly
            slice_start_x = randint(0, len(map_resized[0]) - target_grid_size)
            slice_start_y = randint(0, len(map_resized) - target_grid_size)
        else:
            # Map not empty -> find median
            used_xs.sort()
            used_ys.sort()
            slice_mid_x = used_xs[len(used_xs)//2]
            slice_mid_y = used_ys[len(used_ys)//2]
            slice_start_x = slice_mid_x - target_grid_size//2
            slice_start_y = slice_mid_y - target_grid_size//2
            if (slice_start_x < 0):
                slice_start_x = 0
            if (slice_start_y < 0):
                slice_start_y = 0
            if (slice_start_x > len(map_resized[0]) - target_grid_size):
                slice_start_x = len(map_resized[0]) - target_grid_size
            if (slice_start_y > len(map_resized) - target_grid_size):
                slice_start_y = len(map_resized) - target_grid_size

        slice_end_x = slice_start_x + target_grid_size
        slice_end_y = slice_start_y + target_grid_size

        map_resized = map_resized[slice_start_y:slice_end_y]
        map_resized = [line[slice_start_x:slice_end_x] for line in map_resized]

    if (pad_start_x == None):
        pad_start_x = 0
    if (pad_start_y == None):
        pad_start_y = 0
        
    return map_resized, slice_start_x, slice_start_y

def mapMirror(game_map, mirror_x, mirror_y):
    map_mirrored = [line.copy() for line in game_map]

    if (mirror_x):
        map_mirrored = [line[::-1] for line in map_mirrored]
    if (mirror_y):
        map_mirrored = map_mirrored[::-1]

    return map_mirrored

def getPreprocessedValue(game_map, x, y, player, potential=False):
    if ((potential and game_map[y][x] != 0) or (not potential and game_map[y][x] != player)):
        return 0

    max_row = 0
    for direction in range(4):
        row = 1
        for sign in [-1, 1]:
            step_x = sign * [1, 0, 1, 1][direction]
            step_y = sign * [0, 1, 1, -1][direction]
            xx = x + step_x
            yy = y + step_y
            while (xx >= 0 and xx < len(game_map[0]) and yy >= 0 and yy < len(game_map) and game_map[yy][xx] == player):
                row += 1
                xx += step_x
                yy += step_y
        if (row > max_row):
            max_row = row

    return max_row

def packGrids(grids):
    return tf.constant(grids, shape=[1, len(grids), len(grids[0]), len(grids[0][0]), 1], dtype=tf.int32)

def mapToCnnInput(game_map, cnn_input_grids):
    map_value_this_player            = [[0 for e in range(len(game_map[0]))] for i in range(len(game_map))]
    map_value_other_player           = [[0 for e in range(len(game_map[0]))] for i in range(len(game_map))]
    map_value_potential_this_player  = [[0 for e in range(len(game_map[0]))] for i in range(len(game_map))]
    map_value_potential_other_player = [[0 for e in range(len(game_map[0]))] for i in range(len(game_map))]

    #str_map = ""
    #str_value_this_player = ""
    #str_value_other_player = ""
    #str_value_potential_this_player = ""
    #str_value_potential_other_player = ""

    for i in range(len(game_map)):
        for e in range(len(game_map[i])):
            map_value_this_player           [i][e] = getPreprocessedValue(game_map, e, i, 1)
            map_value_other_player          [i][e] = getPreprocessedValue(game_map, e, i, 2)
            map_value_potential_this_player [i][e] = getPreprocessedValue(game_map, e, i, 1, True)
            map_value_potential_other_player[i][e] = getPreprocessedValue(game_map, e, i, 2, True)
        #str_map += " ".join([f"{x:2d}" for x in game_map[i]]) + "\n"
        #str_value_this_player += " ".join([f"{x:2d}" for x in map_value_this_player[i]]) + "\n"
        #str_value_other_player += " ".join([f"{x:2d}" for x in map_value_other_player[i]]) + "\n"
        #str_value_potential_other_player += " ".join([f"{x:2d}" for x in map_value_potential_other_player[i]]) + "\n"
        #str_value_potential_this_player += " ".join([f"{x:2d}" for x in map_value_potential_this_player[i]]) + "\n"

    #with open(OUT_FILE, "a") as F:
    #    F.write("="*20 + "\n")
    #    F.write(f"game_map:\n{str_map}")
    #    F.write(f"this player:\n{str_value_this_player}")
    #    F.write(f"other player:\n{str_value_other_player}")
    #    F.write(f"potential this player:\n{str_value_potential_this_player}")
    #    F.write(f"potential other player:\n{str_value_potential_other_player}")

    if (cnn_input_grids == 1):
        cnn_input = packGrids([game_map])
    elif (cnn_input_grids >= 2 and cnn_input_grids <= 4):
        cnn_input = packGrids([map_value_this_player, map_value_other_player, map_value_potential_this_player, map_value_potential_other_player][:cnn_input_grids])
    elif (cnn_input_grids == 5):
        cnn_input = packGrids([game_map, map_value_this_player, map_value_other_player, map_value_potential_this_player, map_value_potential_other_player])
    else:
        assert (False), f"Unsupported number of input grids: {cnn_input_grids}"

    return cnn_input

def testVariations():
    def test(mv, g, ts, x0, y0, x1, y1):
        g = g.copy()
        go.grid = mv.getResizedMap(g.grid, ts)
        print(go)
        gp = go.copy()
        x, y = mv.gameToCnn(x0, y0)
        gp.play(x, y, g.F_X)
        print(gp)
        x, y = mv.cnnToGame(x1, y1)
        g.play(x, y, g.F_X)
        print(g)

    mv_0_0_0_0_0 = MapVariation(0, 0, 0, 0, 0)
    mv_0_0_1_0_1 = MapVariation(0, 0, 1, 0, 1)
    mv_1_2_0_1_0 = MapVariation(1, 2, 0, 1, 0)
    mv_1_2_1_1_1 = MapVariation(1, 2, 1, 1, 1)
    mv_0_0_0_s_0_1 = MapVariation(0, 0, 0, 0, 1)
    mv_0_0_1_s_1_0 = MapVariation(0, 0, 1, 1, 0)

    g0 = TicTacToe(4, 4)
    g0.play(0, 0, g0.F_X)
    g0.play(0, 1, g0.F_O)
    g0.play(0, 2, g0.F_X)

    g1 = TicTacToe(8, 8)
    g1.play(4, 4, g0.F_X)
    g1.play(4, 5, g0.F_O)
    g1.play(4, 6, g0.F_X)

    ts = 6

    go = TicTacToe(ts, ts)

    print(f"g0:")
    print(g0)
    print(f"g1:")
    print(g1)

    print(f"g0 mv_0_0_0_0_0:")
    test(mv_0_0_0_0_0, g0, ts, 1, 0, 3, 2)
    print(f"g0 mv_0_0_1_0_1:")
    test(mv_0_0_1_0_1, g0, ts, 1, 0, 3, 2)
    print(f"g0 mv_1_2_0_1_0:")
    test(mv_1_2_0_1_0, g0, ts, 1, 0, 3, 2)
    print(f"g0 mv_1_2_1_1_1:")
    test(mv_1_2_1_1_1, g0, ts, 1, 0, 3, 4)

    print(f"g1 mv_0_0_0_s_0_1:")
    test(mv_0_0_0_s_0_1, g1, ts, 1, 5, 2, 4)
    print(f"g1 mv_0_0_1_s_1_0:")
    test(mv_0_0_1_s_1_0, g1, ts, 2, 5, 2, 4)
