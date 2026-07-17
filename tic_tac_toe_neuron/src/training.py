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

# Sorts players by the number of turns played and leaves only those with the shortest histories
# appearing in the set, up to the cutoff number of different history lengths.
def cutOffShortest(players, cutoff):
    if (len(players) == 0):
        return players

    # Leave only players with the shortest 'cutoff' histories appearing in the set
    turns = {p.turns_played for p in players}
    turns_sorted = sorted(list(turns))
    if (cutoff > len(turns_sorted) or cutoff == 0):
        cutoff = len(turns_sorted)

    shortest = turns_sorted[cutoff-1]

    result = []
    for p in players:
        if (p.turns_played <= shortest):
            result.append(p)
    return result

# Process for generating training data
# Periodically loads the latest model, plays a set of games against a random recent model,
# collects training data from the winners and losers, and appends it to the output list.
def generateTrainingDataProcess(
    model_file,
    serial_rounds,
    shortest_cutoff,
    model_file_lock,
    produced_data_lock,
    produced_data_list,
    winner_weight,
    loser_weight,
    kept_models,
    player_training_variants,
    top_random_select_size,
    top_select_equal,
    weights_scale_coef,
    weights_scale_uniform,
    train_against_top_random_select_1
    ):
    use_winners = winner_weight > 0
    use_losers = loser_weight > 0
    assert (use_winners or use_losers), "At least one of winner_weight or loser_weight must be over 0.0."
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
        winners = []
        losers = []
        stat_turns = []
        stat_cutoff = 0
        p1_top_random_select_size = 1 if train_against_top_random_select_1 else top_random_select_size
        use_p1_result = (not train_against_top_random_select_1)
        for _ in range(serial_rounds):
            p0 = TTTPlayerCNN(model_trained, player_training_variants, winner_weight, loser_weight, top_random_select_size, top_select_equal)
            p1 = TTTPlayerCNN(model_other  , player_training_variants, winner_weight, loser_weight, p1_top_random_select_size, top_select_equal)
            pp = [p0, p1]
            switched = randint(0,1)
            if (switched):
                pp = [p1, p0]
            pp[0].plays_first = True
            pp[1].plays_first = False
            ttt_play = TicTacToePlay(grid_size, grid_size, *pp, win_strike_length)
            result = ttt_play.play(0)
            stat_turns.append(p0.turns_played + p1.turns_played)

            if (result == 1+switched):
                if (use_winners):
                    winners.append(p0)
                if (use_losers and use_p1_result):
                    losers.append(p1)
            elif (result == 2-switched):
                if (use_winners and use_p1_result):
                    winners.append(p1)
                if (use_losers):
                    losers.append(p0)

            if (os.path.exists(abort_file) or os.path.exists(pause_file)):
                break

        for p in winners:
            p.recorded_game.played_first = p.plays_first
            p.recorded_game.won = True
        for p in losers:
            p.recorded_game.played_first = p.plays_first
            p.recorded_game.won = False

        # Cut off to only get players with the shortest histories (fastest win/lose)
        len_prev_win = len(winners)
        winners = cutOffShortest(winners, shortest_cutoff)

        len_prev_lose = len(losers)
        losers = cutOffShortest(losers, shortest_cutoff)

        # Get training data
        win_lose_stat = {}
        training_data_list = getTrainingDataList(winners, losers, win_strike_length, weights_scale_coef, weights_scale_uniform, win_lose_stat)

        stat_winners = len(winners)
        stat_losers  = len(losers)
        stat_cutoff += len_prev_win - stat_winners
        stat_cutoff += len_prev_lose - stat_losers

        win_lose_stat["winners"] = stat_winners
        win_lose_stat["losers"]  = stat_losers

        if (len(training_data_list)):
            training_data = training_data_list[0]
            for td in training_data_list[1:]:
                training_data = training_data.concat(td)

            # Append to output list
            with produced_data_lock:
                produced_data_list.append((training_data, stat_turns, stat_cutoff, win_lose_stat))

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

# Gets training data from the winners and losers, applies weights scaling if needed, and concatenates it into a single list
def getTrainingDataList(winners, losers, win_strike_length, weights_scale_coef, weights_scale_uniform, win_lose_stat={}):
    training_data_list_win  = []
    training_data_list_lose = []
    for winner in winners:
        training_data_list_win.append(getTrainingData(
            [winner.recorded_game],
            winner.cnn_model.input_shape,
            winner.training_variants,
            winner.winner_weight,
            winner.loser_weight,
            weights_scale_coef,
            weights_scale_uniform
            ))
    for loser in losers:
        training_data_list_lose.append(getTrainingData(
            [loser.recorded_game],
            loser.cnn_model.input_shape,
            loser.training_variants,
            loser.winner_weight,
            loser.loser_weight,
            weights_scale_coef,
            weights_scale_uniform))

    w_first, w_second = normalizeTrainingDataWeights(training_data_list_win , winners)
    l_first, l_second = normalizeTrainingDataWeights(training_data_list_lose, losers )

    win_lose_stat["w_first"]  = win_lose_stat.get("w_first" , 0) + w_first
    win_lose_stat["w_second"] = win_lose_stat.get("w_second", 0) + w_second
    win_lose_stat["l_first"]  = win_lose_stat.get("l_first" , 0) + l_first
    win_lose_stat["l_second"] = win_lose_stat.get("l_second", 0) + l_second

    return training_data_list_win + training_data_list_lose

# Modify training data weights so that the combined weight of games played as first player
# is the same as the combined weight of games played as second player, to avoid overtraining for only one of the roles.
def normalizeTrainingDataWeights(training_data_list, players):
    assert (len(training_data_list) == len(players)), "The number of training data items must be the same as the number of players for weight normalization."
    if (len(training_data_list) == 0):
        return 0, 0

    #S = "Normalizing training data weights:\n"
    #S += f"  Weights 0 original: {[float(x[0]) for x in training_data_list[0].weight_data]}\n"

    games_num = len(players)
    games_played_first_num = sum([1 for p in players if (p.plays_first)])

    played_second_weight = games_played_first_num / games_num
    played_first_weight  = 1 - played_second_weight

    #S += f"  Total games: {games_num}, played first: {games_played_first_num} ({played_first_weight:.2f}), played second: {games_num - games_played_first_num} ({played_second_weight:.2f})\n"

    if (played_first_weight == 0 or played_second_weight == 0):
        # There is only one type of data present, that will have weight 0.
        # There is no point in training on this, so remove all the data instead.
        training_data_list.clear()
        players.clear()
        #S += "  Clearing.\n"
        #print(S)
        return games_played_first_num, games_num - games_played_first_num

    for td, p in zip(training_data_list, players):
        coef = played_first_weight if (p.plays_first) else played_second_weight
        td_w_float = [[float(w[0]) * coef] for w in td.weight_data]
        td.weight_data = tf.constant(td_w_float, dtype=tf.float32)

    #S += f"  Weights 1 modified: {[float(x[0]) for x in training_data_list[0].weight_data]}\n"
    #print(S)

    return games_played_first_num, games_num - games_played_first_num

# Process for training the model
# Periodically loads the latest model, checks for new training data, trains the model on it,
# saves the new model for generators to use, and periodically stores the model in the storage.
def trainModelProcess(
    model_file,
    model_name,
    model_games,
    model_inputs_trained,
    max_training_data_size,
    accumulate_training_data,
    train_interval,
    batch_size,
    store_interval,
    model_file_lock,
    produced_data_lock,
    produced_data_list,
    kept_models,
    learning_rate
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
        new_data_size = sum([len(pd[0].input_data) for pd in pdl])

        # Concatenate statistics
        stat_turns    = sum([pd[1] for pd in pdl], [])
        stat_cutoff   = sum([pd[2] for pd in pdl])
        stat_win_lose = {}
        for w_l_s in [pd[3] for pd in pdl]:
            for k, v in w_l_s.items():
                stat_win_lose[k] = stat_win_lose.get(k, 0) + v

        # Concatenate training data
        if (training_data == None or (not accumulate_training_data)):
            training_data = pdl[0][0]
            pdl = pdl[1:]
        for pd in pdl:
            training_data = training_data.concat(pd[0])

        # Truncate oldest training data
        training_data = training_data.truncate(max_training_data_size)

        # Calculate and print statistics
        stat_games     = len(stat_turns)
        stat_avg_turns = sum(stat_turns) / stat_games
        stat_min_turns = min(stat_turns)
        stat_max_turns = max(stat_turns)
        stat_winners   = stat_win_lose.get("winners" , 0)
        stat_losers    = stat_win_lose.get("losers"  , 0)
        stat_w_first   = stat_win_lose.get("w_first" , 0)
        stat_w_second  = stat_win_lose.get("w_second", 0)
        stat_l_first   = stat_win_lose.get("l_first" , 0)
        stat_l_second  = stat_win_lose.get("l_second", 0)
        winners_losers_total = stat_cutoff + stat_winners + stat_losers
        stat_cutoff_perc = 100 * stat_cutoff / winners_losers_total if winners_losers_total > 0 else 0
        time_passed = t - last_stat_time
        last_stat_time = t
        games_per_sec = stat_games / time_passed
        print(f"   Time passed: {time_passed:.2f} s, Games: {stat_games:3}, Winners/first/second: {stat_winners:3}/{stat_w_first:3}/{stat_w_second:3}, Losers/first/second: {stat_losers:3}/{stat_l_first:3}/{stat_l_second:3}, Cut off games: {stat_cutoff:3} ({stat_cutoff_perc:.2f}%)\n   Turns min/avg/max: {stat_min_turns:3}/{stat_avg_turns:.2f}/{stat_max_turns}, Games/sec: {games_per_sec:5.2f}, New data size: {new_data_size}, Training data size: {len(training_data.input_data)}", flush=True)

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
        model_games += stat_games
        model_inputs_trained += len(training_data.input_data)
        if (t - last_store_time > store_interval):
            name = f"{model_name}_{stat_avg_turns:.1f}avg_{model_games}_{model_inputs_trained}"
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
    accumulate_training_data,
    train_interval,
    store_interval,
    batch_size,
    serial_rounds,
    threads_num,
    use_testing_thread=False,
    shortest_cutoff=0,
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
            tmp_model_file, serial_rounds, shortest_cutoff, model_file_lock, produced_data_lock, produced_data_list, winner_weight, loser_weight, kept_models, player_training_variants, top_random_select_size, top_select_equal, weights_scale_coef, weights_scale_uniform, train_against_top_random_select_1))
        producer_processes.append(p)
        p.start()
    
    # Create consumer process
    consumer_process = multiprocessing.Process(target=funcAbortWrapper, args=(trainModelProcess,
        tmp_model_file, model_name, model_games, model_inputs_trained, max_training_data_size, accumulate_training_data, train_interval, batch_size, store_interval, model_file_lock, produced_data_lock, produced_data_list, kept_models, learning_rate))
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
