#!python3

from random import randint, seed, random
import tensorflow as tf

from map_preprocess import *
from model import *

OUT_FILE = "out.log"

class TicTacToePlayer:
    def getNextTurn(self, game_map):
        assert (False), "Player behavior not defined"

    def get_last_map_resize_paddings(self):
        return 0, 0

class TTTPlayerUser:
    def __init__(self, name):
        self.name = name
        self.random_player = TTTPlayerRandom()

    def getNextTurn(self, game_map):
        print(f"Player {self.name} turn:")
        try:
            x = int(input("x: "))
            y = int(input("y: "))
            return x, y
        except:
            return self.random_player.getNextTurn(game_map)

class TTTPlayerRandom:
    def __init__(self, wait_for_user_input=False):
        self.wait = wait_for_user_input

    def getNextTurn(self, game_map):
        empty_cells = []
        for i,line in enumerate(game_map):
            for e,field in enumerate(line):
                if (game_map[i][e] == 0):
                    empty_cells.append((e, i))

        x, y = empty_cells[randint(0, len(empty_cells)-1)]

        if (self.wait):
            input(f"Going to play {x}, {y}. Waiting...")

        return x, y

class TTTPlayerFred:
    def __init__(self, mistake_rate=0.0):
        self.mistake_rate = mistake_rate
        self.random_player = TTTPlayerRandom()

    class CoordSet:
        def __init__(self):
            self.coords = []

        def getValue(self, map):
            x, y = self.coords[0]
            return int(map[y][x][0])

        def getRandomCoord(self):
            return self.coords[randint(0, len(self.coords)-1)]

        def getRandomMaxValueCoord(self, map):
            max_value = 0
            max_coords = []
            for coord in self.coords:
                x, y = coord
                if (map[y][x][0] > max_value):
                    max_coords = [coord]
                    max_value  = map[y][x][0]
                elif (map[y][x][0] == max_value):
                    max_coords.append(coord)
            return max_coords[randint(0, len(max_coords)-1)]

        def addCoord(self, x, y):
            self.coords.append((x, y))

        def replaceCoords(self, coords):
            self.coords = coords

        def intersection(self, other):
            new_set = TTTPlayerFred.CoordSet()
            for coord in self.coords:
                if (coord in other.coords):
                    new_set.addCoord(*coord)
            return new_set

        def __len__(self):
            return len(self.coords)

    def getMaxCoordSet(self, map):
        max_coord_set = TTTPlayerFred.CoordSet()
        max_value = 0
        for y,line in enumerate(map):
            for x,field in enumerate(line):
                if (map[y][x][0] > max_value):
                    max_coord_set.replaceCoords([(x, y)])
                    max_value = map[y][x][0]
                elif (map[y][x][0] == max_value):
                    max_coord_set.addCoord(x, y)
        return max_coord_set

    def getNextTurn(self, game_map):
        preprocessed_map = mapToCnnInput(game_map, 4)

        map_value_potential_this_player  = preprocessed_map[0][2]
        map_value_potential_other_player = preprocessed_map[0][3]

        this_player_max_coord_set  = self.getMaxCoordSet(map_value_potential_this_player)
        other_player_max_coord_set = self.getMaxCoordSet(map_value_potential_other_player)

        max_coord_set_intersection = this_player_max_coord_set.intersection(other_player_max_coord_set)

        if (len(max_coord_set_intersection) > 0 and random() > self.mistake_rate):
            # There are fields which have maximum potential for both players
            # -> Select one of those fields
            return max_coord_set_intersection.getRandomCoord()

        if (other_player_max_coord_set.getValue(map_value_potential_other_player) > this_player_max_coord_set.getValue(map_value_potential_this_player) and random() > self.mistake_rate):
            # Other player has fields with higher potential than this player
            # -> Select one of those fields where there is also maximum potential for this player
            if (random() > self.mistake_rate):
                return other_player_max_coord_set.getRandomMaxValueCoord(map_value_potential_this_player)
            else:
                return other_player_max_coord_set.getRandomCoord()
        elif (random() > self.mistake_rate):
            # This player has fields with same or higher potential than other player
            # -> Select one of those fields where there is also maximum potential for other player
            if (random() > self.mistake_rate):
                return this_player_max_coord_set.getRandomMaxValueCoord(map_value_potential_other_player)
            else:
                return this_player_max_coord_set.getRandomCoord()

        # Complete mistake
        # -> Select random field
        return self.random_player.getNextTurn(game_map)

def cycleRelevantCoordinates(game_size, cnn_output_size, resize_variant):
    if (cnn_output_size < game_size):
        for y in range(cnn_output_size):
            for x in range(input_grid_size):
                game_x, game_y = resize_variant.cnnToGame(x, y)
                cnn_output_x, cnn_output_y = x, y
                yield game_x, game_y, cnn_output_x, cnn_output_y
    else:
        for y in range(game_size):
            for x in range(game_size):
                game_x, game_y = x, y
                cnn_output_x, cnn_output_y = resize_variant.gameToCnn(x, y)
                yield game_x, game_y, cnn_output_x, cnn_output_y

class TTTPlayerCNN:

    class InputVariant:
        def __init__(self):
            self.game_map_resized = []
            self.slice_start_x = 0
            self.slice_start_y = 0
            self.pad_start_x = 0
            self.pad_start_y = 0
            self.cnn_input = None

    def __init__(self, cnn_model, input_transform_func, training_variants=None, winner_wight=1.0, loser_weight=1.0, top_random_select_size=1):
        self.cnn_model = cnn_model
        self.input_transform_func = input_transform_func
        self.training_variants = training_variants
        self.winner_weight = winner_wight
        self.loser_weight = loser_weight
        self.top_random_select_size = top_random_select_size
        self.random_player = TTTPlayerRandom()
        self.cnn_input_history = tf.constant([], shape=[0] + list(self.cnn_model.input_shape[1:]),dtype=tf.int32)
        self.result_history = []
        self.turns_played = 0

    def getNextTurn(self, game_map):
        all_empty = True
        for line in game_map:
            for field in line:
                if (field != 0):
                    all_empty = False
                    break
            if (not all_empty):
                break

        self.turns_played += 1

        # Make first move always random
        if (all_empty):
            return self.random_player.getNextTurn(game_map)

        input_grid_size = self.cnn_model.input_shape[2]

        # Only generate actual maps for a desired subset (to reduce complexity)
        training_variants = len(variants_shifts) if self.training_variants == None else self.training_variants
        variants = MapVariation.getRandomVariations(training_variants, len(game_map), input_grid_size)

        # Select one of the variants to actually produce the next turn
        selected_variant = variants[0]

        cnn_input  = self.input_transform_func(selected_variant.getResizedMap(game_map, input_grid_size), self.cnn_model.input_shape[1])
        cnn_output = self.cnn_model(cnn_input)[0]

        S = ""
        #for line in game_map:
        #    S += str(line) + "\n"
        #S += str(selected_variant) + "\n"
        #S += dataToStr(cnn_input[0])
        #S += dataToStr(cnn_output)

        # Find highest values which don't point to
        # an occupied field on the game map
        max_v = None
        empty_fields_cnt = 0
        top_select = [(0, 0, 0, 0, 0) for i in range(self.top_random_select_size)]
        for game_x, game_y, cnn_output_x, cnn_output_y in cycleRelevantCoordinates(len(game_map), input_grid_size, selected_variant):
            v = float(cnn_output[cnn_output_y][cnn_output_x])
            if (game_map[game_y][game_x] == 0 and v > top_select[0][4]):
                max_v = v
                top_select[0] = (cnn_output_x, cnn_output_y, game_x, game_y, max_v)
                top_select.sort(key=lambda x: x[4])
                #S += f"game_x: {game_x}, game_y: {game_y}, cnn_output_x: {cnn_output_x}, cnn_output_y: {cnn_output_y}, v: {v}\n"

        # Select one of the top values randomly
        result = top_select[randint(0, len(top_select)-1)]
        cnn_output_x, cnn_output_y, game_x, game_y, max_v = result

        # Generate referential data for all variants with the same turn
        for i,variant in enumerate(variants):
            variant_cnn_input = self.input_transform_func(variant.getResizedMap(game_map, input_grid_size), self.cnn_model.input_shape[1])
            self.cnn_input_history = tf.concat([self.cnn_input_history, variant_cnn_input], axis=0)
            variant_output = variant.gameToCnn(game_x, game_y)
            self.result_history.append(variant_output)

        if (randint(0, 20) == 0):
            S += f"output:\n"
            S += dataToStr(cnn_output)
            inspected_values = [float(x[0]) for x in cnn_output]
            S += f"turn: {game_x}:{game_y}\n"
            avg_v = sum(inspected_values) / len(inspected_values)
            min_v = min(inspected_values)
            max_v = max(inspected_values)
            S += f"avg: {avg_v:.05f}, min: {min_v:.05f}, max: {max_v:.05f}, variance: {max_v-min_v:.05f}\n"
            
            with open(OUT_FILE, "w") as F:
                F.write(S)

        return game_x, game_y

    def _getWeightData(self, win_strike_length, won, weights_scale_coef=0.0):
        # The weight is devided by length of the game to get heigher weight for shorter games and lower weight for longer games
        # The largest weight is for a game with minimum possible turns
        target_weight = self.winner_weight if won else self.loser_weight
        min_turns = win_strike_length
        steps = self.turns_played
        if (weights_scale_coef == 0.0):
            weight = target_weight
        else:
            weight = target_weight * (min_turns / steps)**weights_scale_coef

        weight_data = tf.constant([[weight] for i in range(1, len(self.result_history) + 1)], dtype=tf.float32)
        return weight_data

    def _getRefOutputData(self, won):
        ref_output_data = []
        output_grid_size = self.cnn_model.output_shape[1]
        for r in self.result_history:
            x = r[0]
            y = r[1]
            if (won):
                # If player won, set expected output to 1 on index which was played
                # and 0 everywhere else
                exp_output = [[0 for ee in range(output_grid_size)] for i in range(output_grid_size)]
                exp_output[y][x] = 1
            else:
                # If player lost, set expected output to 0 on index which was played
                # and N everywhere else, where N is a value, whose sum over all fields
                # is equal to 1
                N = 1 / (output_grid_size * output_grid_size - 1)
                exp_output = [[N for ee in range(output_grid_size)] for i in range(output_grid_size)]
                exp_output[y][x] = 0
            ref_output_data.append(exp_output)
        return tf.constant(ref_output_data, dtype=tf.float32)

    def getTrainingData(self, win_strike_length, won=True, weights_scale_coef=0.0):
        input_data      = self.cnn_input_history
        weight_data     = self._getWeightData(win_strike_length, won, weights_scale_coef)
        ref_output_data = self._getRefOutputData(won)
        #if (randint(0, 30) == 0):
        #    S = ""
        #    turn = randint(0, len(self.result_history)-4-1)
        #    for i in range(4):
        #        S += "----------------\n"
        #        S += f"turn: {turn}\n"
        #        S += "----------------\n"
        #        S += f"input:\n"
        #        S += dataToStr(input_data[turn])
        #        S += f"weight:\n"
        #        S += dataToStr(weight_data[turn])
        #        S += f"\noutput:\n"
        #        S += dataToStr(ref_output_data[turn])
        #        turn += 1
        #    print("Writing tmp file")
        #    with open("tmp", "w") as F:
        #        F.write(S)
        return trainingData(input_data, weight_data, ref_output_data)
