#!python3

from random import randint, seed, random
import tensorflow as tf

from map_preprocess import *
from model import *
from print_log import *

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
        preprocessed_map = packGrids(mapToCnnInput(game_map, 4))

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

    def __init__(self, cnn_model, training_variants=None, winner_wight=1.0, loser_weight=1.0, top_random_select_size=1, top_select_equal=False, plays_first=None):
        self.cnn_model = cnn_model
        self.input_transform_func = mapToCnnInput
        self.training_variants = training_variants
        self.winner_weight = winner_wight
        self.loser_weight = loser_weight
        self.top_random_select_size = top_random_select_size
        self.top_select_equal = top_select_equal
        self.plays_first = plays_first
        self.random_player = TTTPlayerRandom()
        self.cnn_input_history = tf.constant([], shape=[0] + list(self.cnn_model.input_shape[1:]),dtype=tf.int32)
        self.result_history = []
        self.turns_played = 0
        self.recorded_game = CNNPlayedGameInfo()

    def copy(self):
        new_player = TTTPlayerCNN(self.cnn_model, self.training_variants, self.winner_weight, self.loser_weight, self.top_random_select_size, self.top_select_equal, self.plays_first)
        new_player.cnn_input_history = tf.identity(self.cnn_input_history)
        new_player.result_history = list(self.result_history)
        new_player.turns_played = self.turns_played
        return new_player

    def selectTopRandom(self, game_map, cnn_output, input_grid_size, map_variation):
        # Get all values which don't point to
        # an occupied field on the game map sorted by value
        selected_next_turns = []
        for game_x, game_y, cnn_output_x, cnn_output_y in cycleRelevantCoordinates(len(game_map), input_grid_size, map_variation):
            v = float(cnn_output[cnn_output_y][cnn_output_x])
            if (game_map[game_y][game_x] == 0):
                selected_next_turns.append((cnn_output_x, cnn_output_y, game_x, game_y, v))
        selected_next_turns.sort(key=lambda x: x[4])

        if (self.top_random_select_size > 0):
            # Limit the selection to the top choices
            selected_next_turns = selected_next_turns[-self.top_random_select_size:]

        if (len(selected_next_turns) == 1):
            return selected_next_turns[0]

        # Select one of the values randomly
        # with distribution proportional to their value (higher value -> higher chance to be selected)
        sum_v = sum([x[4] for x in selected_next_turns])
        if (sum_v > 0 and (not self.top_select_equal)):
            r = random() * sum_v
            for s in selected_next_turns:
                r -= s[4]
                if (r <= 0):
                    result = s
                    break
            result = result if r <= 0 else selected_next_turns[-1]
        else:
            # If all values are 0 or top_select_equal is set, select random field with equal probability
            result = selected_next_turns[randint(0, len(selected_next_turns)-1)]

        return result

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
        training_variants = self.training_variants
        variants = MapVariation.getRandomVariations(training_variants, len(game_map), input_grid_size)

        # Select one of the variants to actually produce the next turn
        selected_variant = variants[0]

        cnn_input_transformed = self.input_transform_func(game_map, self.cnn_model.input_shape[1])
        cnn_input_resized     = [selected_variant.getResizedMap(m, input_grid_size) for m in cnn_input_transformed]
        cnn_input_packed      = packGrids(cnn_input_resized)
        cnn_input             = cnn_input_packed
        cnn_output = self.cnn_model(cnn_input)[0]

        S = ""
        #for line in game_map:
        #    S += str(line) + "\n"
        #S += str(selected_variant) + "\n"
        #S += dataToStr(cnn_input[0])
        #S += dataToStr(cnn_output)

        cnn_output_x, cnn_output_y, game_x, game_y, max_v = self.selectTopRandom(game_map, cnn_output, input_grid_size, selected_variant)

        self.recorded_game.played_turns.append(CNNPlayedTurnInfo(game_map, cnn_input_transformed, game_x, game_y))

        # Generate referential data for all variants with the same turn
        turn_index = self.turns_played-1
        for i,variant in enumerate(variants):
            variant_cnn_input_resized = [variant.getResizedMap(m, input_grid_size) for m in cnn_input_transformed]
            variant_cnn_input_packed  = packGrids(variant_cnn_input_resized)
            self.cnn_input_history = tf.concat([self.cnn_input_history, variant_cnn_input_packed], axis=0)
            variant_output = variant.gameToCnn(game_x, game_y)
            self.result_history.append((turn_index, variant_output))

        if (randint(0, 20) == 0):
            S += f"output:\n"

            S_map = dataToStr(cnn_output)
            S_map = S_map.split("\n")
            for i in range(len(S_map)):
                S_map[i] = S_map[i].split(",")

            # Add colored markers for input game state
            for y,line in enumerate(selected_variant.getResizedMap(game_map, input_grid_size)):
                for x,field in enumerate(line):
                    if (field == 1):
                        # Green X for "self"
                        S_map[y][x] = "\033[38;5;46m   X\033[0m"
                    if (field == 2):
                        # Red O for "other"
                        S_map[y][x] = "\033[38;5;196m   O\033[0m"
                S += ",".join(S_map[y]) + "\n"

            inspected_values = []
            for line in cnn_output:
                inspected_values += [float(x[0]) for x in line]
            S += f"turn: {game_x}:{game_y}\n"
            avg_v = sum(inspected_values) / len(inspected_values)
            min_v = min(inspected_values)
            max_v = max(inspected_values)
            S += f"avg: {avg_v:.05f}, min: {min_v:.05f}, max: {max_v:.05f}, variance: {max_v-min_v:.05f}\n"

            # Add colored markers for output values
            S_colored = S.split("\n")
            S_colored = process_log(S_colored)
            S_colored = "\n".join(S_colored)

            with open(OUT_FILE, "w") as F:
                F.write(S_colored)

        return game_x, game_y

class CNNPlayedTurnInfo:
    """
    Information about a single turn played by a CNN player for the purpose
    of generating training data later.
    """

    def __init__(self, game_map, game_map_transformed, turn_x, turn_y):
        self.game_map             = game_map
        self.game_map_transformed = game_map_transformed
        self.turn_x               = turn_x
        self.turn_y               = turn_y

class CNNPlayedGameInfo:
    """
    Information about a series of turns played by a CNN player within a single game
    for the purpose of generating training data later.
    """

    def __init__(self, played_turns=[], played_first=False, won=False):
        self.played_turns = played_turns
        self.played_first = played_first
        self.won          = won

def getRefOutputData(won, output_grid_size, turn_x, turn_y):
    x = turn_x
    y = turn_y
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
    return exp_output

def getTrainingData(
    recorded_games_info,
    cnn_input_shape,
    training_variants,
    winner_weight,
    loser_weight,
    weights_scale_coef=0.0,
    weights_scale_uniform=False
    ):
    """
    Generate proper CNN model training data from a list of recorded games played by the CNN player.

    Weight calculation:
      For each turn, the weight is devided by the number of remaining turns in the game.
      The largest weight (coefficient 1.0) is for the last turn.
      Each turn before that has its weight lowered based on the weights_scale_coef.
      For weights_scale_coef = 0.0, all turns have weight 1.0.
      For weights_scale_coef = 1.0, the wights from last turn go 1/1, 1/2, 1/3, ..., 1/N, where N is the number of turns in the game.
      For higher weights_scale_coef, the weights are lowered slower and slower for the earlier turns.
      For weights_scale_uniform==True, all turns are weighted the same as the first turns (a long game turns get lower weight than short dame turns).
    """
    #training_data_input      = tf.constant([], shape=[0] + list(cnn_input_shape[1:]), dtype=tf.int32)
    #return tf.constant(training_data_ref_output, dtype=tf.float32)
    training_data_input      = []
    training_data_weight     = []
    training_data_ref_output = []

    weights_scale = (weights_scale_coef > 0.0)
    if (weights_scale):
        reverted_coef = 1.0 / weights_scale_coef

    for game in recorded_games_info:
        # Pre-calculate some weight variables
        weight    = winner_weight if game.won else loser_weight
        turns_cnt = len(game.played_turns)
        if (weights_scale and weights_scale_uniform):
            # Same weight for every turn
            turns_remaining = turns_cnt
            weight = weight / (turns_remaining ** reverted_coef)

        for turn_index,turn in enumerate(game.played_turns):
            if (weights_scale and (not weights_scale_uniform)):
                # Calculate weight for this specific turn
                turns_remaining = turns_cnt - turn_index
                weight = weight / (turns_remaining ** reverted_coef)
            training_data_weight.append(weight)

            game_grid_size = len(turn.game_map)
            cnn_grid_size  = cnn_input_shape[2]

            # Define a set of random variants of the turn with different shifts, rotations and mirroring
            variants = MapVariation.getRandomVariations(training_variants, game_grid_size, cnn_grid_size)

            for variant in variants:
                # Apply variant to turn input map
                input_map_variant    = [variant.getResizedMap(m, cnn_grid_size) for m in turn.game_map_transformed]
                training_data_input.append(input_map_variant)

                # Apply variant to turn output
                variant_turn_x, variant_turn_y = variant.gameToCnn(turn.turn_x, turn.turn_y)
                training_data_ref_output.append(getRefOutputData(game.won, cnn_grid_size, variant_turn_x, variant_turn_y))

    # Transform data to TensorFlow tensors
    training_data_input      = tf.constant([packGrids(d) for d in training_data_input], dtype=tf.int32)
    training_data_weight     = tf.constant(training_data_weight, dtype=tf.float32)
    training_data_ref_output = tf.constant(training_data_ref_output, dtype=tf.float32)

    return TrainingData(training_data_input, training_data_weight, training_data_ref_output)
