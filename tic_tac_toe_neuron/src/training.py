import os
import time
import multiprocessing

from model import *
from play import *
from player import *
from storage import *
from abort import *
from global_params import *

# Enable cProfile performance profiling
PROFILE_ENABLE = False

if (PROFILE_ENABLE):
    import cProfile
    import pstats

# Process for generating training data
# Periodically loads the latest model, plays a set of games against a random recent model,
# collects training data from the winners and losers, and appends it to the output list.
def generateTrainingDataProcess(
    model_file,
    serial_rounds,
    model_file_lock,
    produced_data_lock,
    produced_data_list,
    kept_models,
    top_random_select_size,
    top_select_equal,
    train_against_top_random_select_1
    ):
    print(f"Generator {multiprocessing.current_process().name} started", flush=True)
    # Initialize random seed
    seed()

    # Add random delay to avoid all generators accessing resources at the same time
    time.sleep(random()*4)

    if (PROFILE_ENABLE):
        pr = cProfile.Profile()
        pr.enable()

    while (not os.path.exists(abort_file)):
        # Pause if pause file is present
        if (os.path.exists(pause_file)):
            print(f"Pausing generator {multiprocessing.current_process().name}", flush=True)
            while (os.path.exists(pause_file)):
                time.sleep(5)
            print(f"Resuming generator {multiprocessing.current_process().name}", flush=True)

        with model_file_lock:
            # Load latest model
            model_trained = tf.keras.models.load_model(f"{model_file}0.keras")
            # Load random model as other
            model_num = randint(0, kept_models-1)
            model_other = tf.keras.models.load_model(f"{model_file}{model_num}.keras")

        # Play a set of games
        recorded_games_info = []
        p1_top_random_select_size = 1 if train_against_top_random_select_1 else top_random_select_size
        use_p1_result = (not train_against_top_random_select_1)
        for _ in range(serial_rounds):
            p0 = TTTPlayerCNN(model_trained, top_random_select_size, top_select_equal)
            p1 = TTTPlayerCNN(model_other  , p1_top_random_select_size, top_select_equal)

            pp = [p0, p1]
            switched = randint(0,1)
            if (switched):
                pp = [p1, p0]
            pp[0].recorded_game.played_first = True
            pp[1].recorded_game.played_first = False

            ttt_play = TicTacToePlay(grid_size, grid_size, *pp, win_strike_length)
            result = ttt_play.play(0)

            if (result == 1+switched):
                p0.recorded_game.won = True
                p1.recorded_game.won = False
                recorded_games_info.append(p0.recorded_game)
                if (use_p1_result):
                    recorded_games_info.append(p1.recorded_game)
            elif (result == 2-switched):
                p0.recorded_game.won = False
                p1.recorded_game.won = True
                recorded_games_info.append(p0.recorded_game)
                if (use_p1_result):
                    recorded_games_info.append(p1.recorded_game)

            if (os.path.exists(abort_file) or os.path.exists(pause_file)):
                break

        if (len(recorded_games_info)):
            # Append to output list
            with produced_data_lock:
                produced_data_list += recorded_games_info

    if (PROFILE_ENABLE):
        identifier = f"gen_{multiprocessing.current_process().name}"
        pr.disable()
        pr.dump_stats(f"output_{identifier}.prof")
        with open(f"output_{identifier}.txt", "w") as stream:
            stats = pstats.Stats(f"output_{identifier}.prof", stream=stream)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats(0.02)
        os.remove(f"output_{identifier}.prof")

    print(f"Generator {multiprocessing.current_process().name} finished", flush=True)

class TrainingDataStats:
    def __init__(self, recorded_games_info, winner_weight, loser_weight):
        # Extract useful statistics data from list of recorded games intendend for training data
        w_turns = []
        l_turns = []
        self.w_first_cnt  = 0
        self.l_first_cnt  = 0
        for g in recorded_games_info:
            if (g.won):
                first = int(g.played_first)
                self.w_first_cnt += first
                w_turns.append(len(g.played_turns) + first) # Add 1 for first player because its first turn is not recorded in the set
            else:
                first = int(g.played_first)
                self.l_first_cnt += int(g.played_first)
                l_turns.append(len(g.played_turns) + first + 1) # Add 1 for first player and another 1 to account for the last turn that was not played by the loser

        self.w_cnt = len(w_turns)
        self.l_cnt = len(l_turns)

        if (self.w_cnt):
            games_turns = w_turns
        else:
            games_turns = l_turns

        self.games_cnt = len(games_turns)
        self.turns_min = 0
        self.turns_max = 0
        self.turns_avg = 0
        if (self.games_cnt):
            self.turns_min = min(games_turns)
            self.turns_max = max(games_turns)
            self.turns_avg = sum(games_turns) / self.games_cnt

        self.w_second_cnt = self.w_cnt - self.w_first_cnt
        self.l_second_cnt = self.l_cnt - self.l_first_cnt

        # Weigh training data based on number of games played as first player or second player.
        # If a majority of games were played as first player or as second player,
        # then the weight of these majority games will be lower to avoid overtraining on only one of these roles.
        self.w_second_weight = (self.w_first_cnt / self.w_cnt) * winner_weight if (self.w_cnt > 0) else 0
        self.w_first_weight  = (1 - self.w_second_weight) * winner_weight
        self.l_second_weight = (self.l_first_cnt / self.l_cnt) * loser_weight  if (self.l_cnt > 0) else 0
        self.l_first_weight  = (1 - self.l_second_weight) * loser_weight

# Process for training the model
# Periodically loads the latest model, checks for new training data, trains the model on it,
# saves the new model for generators to use, and periodically stores the model in the storage.
def trainModelProcess(
    model_file,
    model_name,
    model_games,
    model_inputs_trained,
    max_training_data_size,
    train_interval,
    batch_size,
    store_interval,
    model_file_lock,
    produced_data_lock,
    produced_data_list,
    kept_models,
    learning_rate,
    training_variants,
    winner_weight,
    loser_weight,
    weights_scale_coef,
    weights_scale_uniform
    ):
    print(f"Model training {multiprocessing.current_process().name} started", flush=True)
    # Load initial model
    with model_file_lock:
        model = tf.keras.models.load_model(f"{model_file}0.keras")

    # Set model optimizer learning rate
    model.optimizer.learning_rate.assign(learning_rate)

    training_data = None
    last_store_time = time.time()
    last_stat_time  = last_store_time
    last_train_time = last_store_time
    storage = ModelStorage()

    stat_history = []

    if (PROFILE_ENABLE):
        pr = cProfile.Profile()
        pr.enable()

    while (not os.path.exists(abort_file)):
        # Pause if pause file is present
        if (os.path.exists(pause_file)):
            print(f"Pausing model training {multiprocessing.current_process().name}", flush=True)
            while (os.path.exists(pause_file)):
                time.sleep(5)
            print(f"Resuming model training {multiprocessing.current_process().name}", flush=True)

        t = time.time()
        if (t - last_train_time < train_interval):
            time.sleep(1)
            continue

        if (len(produced_data_list) == 0):
            time.sleep(1)
            continue

        last_train_time = t

        # Get training data
        with produced_data_lock:
            pdl = produced_data_list[:]
            produced_data_list[:] = []

        # Calculate and print statistics
        stats = TrainingDataStats(pdl, winner_weight, loser_weight)
        time_passed = t - last_stat_time
        last_stat_time = t
        games_per_sec = stats.games_cnt / time_passed

        training_data = getTrainingData(
            pdl,
            model.input_shape,
            training_variants,
            stats.w_first_weight,
            stats.w_second_weight,
            stats.l_first_weight,
            stats.l_second_weight,
            weights_scale_coef,
            weights_scale_uniform
            )

        # Truncate training data to maximum size
        training_data = training_data.truncate(max_training_data_size)

        print(
            f"   Time passed: {time_passed:.2f} s, " \
           +f"Games: {stats.games_cnt:3}, " \
           +f"Winners/first/second: {stats.w_cnt:3}/{stats.w_first_cnt:3}/{stats.w_second_cnt:3}, " \
           +f"Losers/first/second: {stats.l_cnt:3}/{stats.l_first_cnt:3}/{stats.l_second_cnt:3}\n" \
           +f"   Turns min/avg/max: {stats.turns_min:3}/{stats.turns_avg:.2f}/{stats.turns_max}, " \
           +f"Games/sec: {games_per_sec:5.2f}, " \
           +f"Training data size: {len(training_data.input_data)}"
           , flush=True)

        # Train model
        training_data.trainModel(model, batch_size, 1, MAX_TRAINING_DATA_SIZE)

        with model_file_lock:
            # Rotate older model files
            os.remove(f"{model_file}{kept_models-1}.keras")
            for i in range(kept_models-1, 0, -1):
                if (os.path.exists(f"{model_file}{i-1}.keras")):
                    os.rename(f"{model_file}{i-1}.keras", f"{model_file}{i}.keras")
            # Save new model for generators
            model.save(f"{model_file}0.keras")

        # Save model for storage
        model_games += stats.games_cnt
        model_inputs_trained += len(training_data.input_data)
        if (t - last_store_time > store_interval):
            name = f"{model_name}_{stats.turns_avg:.1f}avg_{model_games}_{model_inputs_trained}"
            print(f"Storing model {name}", flush=True)
            last_store_time = t
            storage.storeModel(model, name)

    if (PROFILE_ENABLE):
        identifier = f"train_{multiprocessing.current_process().name}"
        pr.disable()
        pr.dump_stats(f"output_{identifier}.prof")
        with open(f"output_{identifier}.txt", "w") as stream:
            stats = pstats.Stats(f"output_{identifier}.prof", stream=stream)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats(0.02)
        os.remove(f"output_{identifier}.prof")

    print(f"Model training {multiprocessing.current_process().name} finished", flush=True)

# Process for testing the model
# Periodically loads the latest model, plays a set of games against a simple opponent, and prints the results.
# Servest to keep track of the model's ability against a stable reference opponent.
def testModelProcess(
    model_file,
    model_name,
    model_file_lock,
    fred_mistake_rate,
    test_runs
    ):
    print(f"Model testing {multiprocessing.current_process().name} started", flush=True)
    p_fred = TTTPlayerFred(fred_mistake_rate)

    while (not os.path.exists(abort_file)):
        # Pause if pause file is present
        if (os.path.exists(pause_file)):
            print(f"Pausing tester {multiprocessing.current_process().name}", flush=True)
            while (os.path.exists(pause_file)):
                time.sleep(5)
            print(f"Resuming tester {multiprocessing.current_process().name}", flush=True)

        with model_file_lock:
            # Load latest model
            model = tf.keras.models.load_model(f"{model_file}0.keras")

        # Test model
        model_player = TTTPlayerCNN(model, 1)
        ttt_play0 = TicTacToePlay(grid_size, grid_size, p_fred, model_player, win_strike_length, 0)
        ttt_play1 = TicTacToePlay(grid_size, grid_size, model_player, p_fred, win_strike_length, 0)

        model_wins = 0
        enemy_wins = 0
        draws      = 0

        for _ in range(test_runs//2):
            result = ttt_play0.play(0)

            model_wins += int(result ==  2)
            enemy_wins += int(result ==  1)
            draws      += int(result == -1)

            result = ttt_play1.play(0)

            model_wins += int(result ==  1)
            enemy_wins += int(result ==  2)
            draws      += int(result == -1)

            if (os.path.exists(abort_file) or os.path.exists(pause_file)):
                break

        print(f"              ======== {model_wins} wins, {enemy_wins} losses, {draws} draws ========", flush=True)

    print(f"Model testing {multiprocessing.current_process().name} finished", flush=True)

# Main training function
# Runs multiple parallel processes to generate training data, train the model, and optionally test the model.
def train(
    model,
    model_name,
    model_games,
    model_inputs_trained,
    max_training_data_size,
    train_interval,
    store_interval,
    batch_size,
    serial_rounds,
    threads_num,
    use_testing_thread=False,
    winner_weight=1.0,
    loser_weight=1.0,
    learning_rate=0.00001,
    kept_models=5,
    player_training_variants=None,
    top_random_select_size=1,
    top_select_equal=False,
    weights_scale_coef=0.0,
    weights_scale_uniform=False,
    fred_mistake_rate=0.0,
    test_runs=100,
    train_against_top_random_select_1=False
    ):
    use_winners = winner_weight > 0
    use_losers = loser_weight > 0
    assert (use_winners or use_losers), "At least one of winner_weight or loser_weight must be over 0.0."
    assert (threads_num >= 2 + int(use_testing_thread)), f"At least {2 + int(use_testing_thread)} threads are required for the training."
    tmp_model_file = "tmp"
    num_producers = threads_num - 1 - int(use_testing_thread)
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    model_file_lock = multiprocessing.Lock()
    produced_data_lock = multiprocessing.Lock()
    produced_data_list = manager.list()

    # Initialize input files (all the same at the start)
    for i in range(kept_models):
        model.save(f"{tmp_model_file}{i}.keras")

    # Clear abort file
    if (os.path.exists(abort_file)):
        os.remove(abort_file)

    print(f"Training started. To abort, create a file named '{abort_file}' in the working directory.")

    # Create producer processes
    producer_processes = []
    for i in range(num_producers):
        p = multiprocessing.Process(target=funcAbortWrapper, args=(generateTrainingDataProcess,
            tmp_model_file, serial_rounds, model_file_lock, produced_data_lock, produced_data_list, kept_models, top_random_select_size, top_select_equal, train_against_top_random_select_1))
        producer_processes.append(p)
        p.start()
    
    # Create consumer process
    consumer_process = multiprocessing.Process(target=funcAbortWrapper, args=(trainModelProcess,
        tmp_model_file, model_name, model_games, model_inputs_trained, max_training_data_size, train_interval, batch_size, store_interval, model_file_lock, produced_data_lock, produced_data_list, kept_models, learning_rate, player_training_variants, winner_weight, loser_weight, weights_scale_coef, weights_scale_uniform))
    consumer_process.start()

    if (use_testing_thread):
        # Create testing process
        testing_process = multiprocessing.Process(target=funcAbortWrapper, args=(testModelProcess,
            tmp_model_file, model_name, model_file_lock, fred_mistake_rate, test_runs))
        testing_process.start()
    
    # Join processes
    for p in producer_processes:
        p.join()
    consumer_process.join()
    if (use_testing_thread):
        testing_process.join()

    print(f"Training finished.")

    # Delete temporary model files
    for i in range(kept_models):
        if (os.path.exists(f"{tmp_model_file}{i}.keras")):
            os.remove(f"{tmp_model_file}{i}.keras")
