import os
# Disable tensorflow warning message about missing high-performance instructions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Allow GPU memory growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from pprint import pprint
import math
import time
import multiprocessing
import numpy as np

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#exit(0)

from model import *
from tic_tac_toe import *
from play import *
from player import *
from storage import *

# Tic Tac Toe game parameters
grid_size = 12
win_strike_length = 5

# Maximum number of training data for each training call
# (influences memory consumption)
MAX_TRAINING_DATA_SIZE = 1500
# Maximum number of training data samples to keep
MAX_TRAINING_DATA_KEPT = MAX_TRAINING_DATA_SIZE * 50

# Abort file name
# If this file is present in the working directory, the training will be stopped
abort_file = "abort"

# Pause file name
# If this file is present in the working directory, the training will be paused
pause_file = "pause"

def abort():
    with open(abort_file, "w") as F:
        F.write("")

# The multiprocessing library is not compatible with decorated functions,
# so this is a workaround solution
def funcAbortWrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Generator {multiprocessing.current_process().name} raised an error '{e}'. Aborting now.", flush=True)
        abort()
        raise

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

def generateTrainingData(model_file, model_input_map_func, serial_rounds, shortest_cutoff, model_file_lock, produced_data_lock, produced_data_list, winner_weight, loser_weight, kept_models, player_training_variants, top_random_select_size, weights_scale_coef, weights_scale_uniform, train_against_top_random_select_1):
    use_winners = winner_weight > 0
    use_losers = loser_weight > 0
    assert (use_winners or use_losers), "At least one of winner_weight or loser_weight must be over 0.0."
    print(f"Generator {multiprocessing.current_process().name} started", flush=True)
    # Initialize random seed
    seed()

    # Add random delay to avoid all generators accessing resources at the same time
    time.sleep(random()*10)

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
            p0 = TTTPlayerCNN(model_trained, model_input_map_func, player_training_variants, winner_weight, loser_weight, top_random_select_size)
            p1 = TTTPlayerCNN(model_other  , model_input_map_func, player_training_variants, winner_weight, loser_weight, p1_top_random_select_size)
            pp = [p0, p1]
            switched = randint(0,1)
            if (switched):
                pp = [p1, p0]
            ttt_play = TicTacToePlay(grid_size, grid_size, *pp, win_strike_length)
            result = ttt_play.play(0)
            stat_turns.append(p0.turns_played + p1.turns_played)

            if (result == 2-switched):
                if (use_winners):
                    winners.append(p0)
                if (use_losers and use_p1_result):
                    losers.append(p1)
            elif (result == 1+switched):
                if (use_winners and use_p1_result):
                    winners.append(p1)
                if (use_losers):
                    losers.append(p0)

            if (os.path.exists(abort_file) or os.path.exists(pause_file)):
                break

        # Cut off to only get players with the shortest histories (fastest win/lose)
        len_prev = len(winners)
        winners = cutOffShortest(winners, shortest_cutoff)
        stat_winners = len(winners)
        stat_cutoff = len_prev - stat_winners

        len_prev = len(losers)
        losers = cutOffShortest(losers, shortest_cutoff)
        stat_losers = len(losers)
        stat_cutoff += len_prev - stat_losers

        # Get training data
        if (len(winners)):
            training_data = winners[0].getTrainingData(win_strike_length, True, weights_scale_coef, weights_scale_uniform)
            winners = winners[1:]
        elif (len(losers)):
            training_data = losers[0].getTrainingData(win_strike_length, False, weights_scale_coef, weights_scale_uniform)
            losers = losers[1:]

        for winner in winners:
            training_data = training_data.concat(winner.getTrainingData(win_strike_length, True, weights_scale_coef, weights_scale_uniform))
        for loser in losers:
            training_data = training_data.concat( loser.getTrainingData(win_strike_length, False, weights_scale_coef, weights_scale_uniform))

        # Append to output list
        with produced_data_lock:
            produced_data_list.append((training_data, stat_turns, stat_cutoff, stat_winners, stat_losers))

    print(f"Generator {multiprocessing.current_process().name} finished", flush=True)

def trainModel(model_file, model_name, model_games, model_inputs_trained, max_training_data_size, train_interval, batch_size, store_interval, model_file_lock, produced_data_lock, produced_data_list, kept_models, learning_rate):
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
        stat_turns   = sum([pd[1] for pd in pdl], [])
        stat_cutoff  = sum([pd[2] for pd in pdl])
        stat_winners = sum([pd[3] for pd in pdl])
        stat_losers  = sum([pd[4] for pd in pdl])

        # Concatenate training data
        if (training_data == None):
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
        time_passed = t - last_stat_time
        last_stat_time = t
        games_per_sec = stat_games / time_passed
        winners_losers_total = stat_cutoff + stat_winners + stat_losers
        stat_cutoff_perc = 100 * stat_cutoff / winners_losers_total if winners_losers_total > 0 else 0
        print(f"   Time passed: {time_passed:.2f} s, Games: {len(stat_turns):3}, Winners: {stat_winners:3}, Losers: {stat_losers:3}, Cut off winners/losers: {stat_cutoff:3} ({stat_cutoff_perc:.2f}%)\n   Avg turns: {stat_avg_turns:.2f}, Min turns: {stat_min_turns}, Max turns: {stat_max_turns}, Games/sec: {games_per_sec:5.2f}, New data size: {new_data_size}, Training data size: {len(training_data.input_data)}", flush=True)

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
        model_games += len(stat_turns)
        model_inputs_trained += len(training_data.input_data)
        if (t - last_store_time > store_interval):
            print(f"Storing model {model_name}_{model_games}_{model_inputs_trained}", flush=True)
            last_store_time = t
            storage.storeModel(model, f"{model_name}_{model_games}_{model_inputs_trained}")

    print(f"Model training {multiprocessing.current_process().name} finished", flush=True)

def testModel(model_file, model_input_map_func, model_name, model_file_lock, fred_mistake_rate, test_runs):
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
        model_player = TTTPlayerCNN(model, model_input_map_func, 1)
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

def train(model, model_name, model_games, model_inputs_trained, model_input_map_func, max_training_data_size, train_interval, store_interval, batch_size, serial_rounds, threads_num, use_testing_thread=False, shortest_cutoff=0, winner_weight=1.0, loser_weight=1.0, learning_rate=0.00001, kept_models=5, player_training_variants=None, top_random_select_size=1, weights_scale_coef=0.0, weights_scale_uniform=False, fred_mistake_rate=0.0, test_runs=100, train_against_top_random_select_1=False):
    assert (threads_num >= 2 + int(use_testing_thread)), f"At least {2 + int(use_testing_thread)} threads are required for the training."
    tmp_model_file = "tmp"
    num_producers = threads_num - 1
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

    #if (os.path.exists(OUT_FILE)):
    #    os.remove(OUT_FILE)

    print(f"Training started. To abort, create a file named '{abort_file}' in the working directory.")

    # Create producer processes
    producer_processes = []
    for i in range(num_producers):
        p = multiprocessing.Process(target=funcAbortWrapper, args=(generateTrainingData, tmp_model_file, model_input_map_func, serial_rounds, shortest_cutoff, model_file_lock, produced_data_lock, produced_data_list, winner_weight, loser_weight, kept_models, player_training_variants, top_random_select_size, weights_scale_coef, weights_scale_uniform, train_against_top_random_select_1))
        producer_processes.append(p)
        p.start()
    
    # Create consumer process
    consumer_process = multiprocessing.Process(target=funcAbortWrapper, args=(trainModel, tmp_model_file, model_name, model_games, model_inputs_trained, max_training_data_size, train_interval, batch_size, store_interval, model_file_lock, produced_data_lock, produced_data_list, kept_models, learning_rate))
    consumer_process.start()

    if (use_testing_thread):
        # Create testing process
        testing_process = multiprocessing.Process(target=funcAbortWrapper, args=(testModel, tmp_model_file, model_input_map_func, model_name, model_file_lock, fred_mistake_rate, test_runs))
        testing_process.start()
    
    # Join processes
    for p in producer_processes:
        p.join()
    consumer_process.join()
    if (use_testing_thread):
        testing_process.join()

    print(f"Training finished.")

    # Delete temporary model file
    if (os.path.exists(tmp_model_file)):
        os.remove(tmp_model_file)

def testAgainstPlayer(model, enemy_player, rounds, verbosity=0, map_func=mapToCnnInput):
    model_player = TTTPlayerCNN(model, map_func, 1)

    model_wins = 0
    enemy_wins = 0
    draws      = 0
    invalids   = 0
    none       = 0

    if (verbosity > 0):
        print(f"Testing against enemy player X", flush=True)
    ttt_play = TicTacToePlay(grid_size, grid_size, enemy_player, model_player, win_strike_length)
    results = []
    for _ in range(rounds//2):
        if (os.path.exists(abort_file)):
            print(f"Aborting", flush=True)
            return 0, 0, 0
        results.append(ttt_play.play(verbosity))

    model_wins += sum([1 for r in results if (r ==  2)])
    enemy_wins += sum([1 for r in results if (r ==  1)])
    draws      += sum([1 for r in results if (r == -1)])
    invalids   += sum([1 for r in results if (r == -2)])
    none       += sum([1 for r in results if (r ==  0)])

    if (verbosity > 0):
        print(f"Testing against enemy player O", flush=True)
    ttt_play = TicTacToePlay(grid_size, grid_size, model_player, enemy_player, win_strike_length)
    results = []
    for _ in range(rounds//2):
        if (os.path.exists(abort_file)):
            print(f"Aborting", flush=True)
            return 0, 0, 0
        results.append(ttt_play.play(verbosity))

    model_wins += sum([1 for r in results if (r ==  1)])
    enemy_wins += sum([1 for r in results if (r ==  2)])
    draws      += sum([1 for r in results if (r == -1)])
    invalids   += sum([1 for r in results if (r == -2)])
    none       += sum([1 for r in results if (r ==  0)])

    if (verbosity > 0):
        print(f"Results:\n"\
           +f"  model_wins: {model_wins}\n"\
           +f"  enemy_wins: {enemy_wins}\n"\
           +f"  draws: {draws}", flush=True)

    return model_wins, enemy_wins, draws

def testAgainstPlayerParallelProcess(model_file, enemy_model_file, rounds, verbosity, map_func, model_file_lock, produced_data_lock, produced_data_dict, model_name, enemy_name):
    # Initialize random seed
    seed()

    # Add random delay to avoid all processes accessing resources at the same time
    time.sleep(random()*10)

    storage = ModelStorage()

    with model_file_lock:
        model       = storage.loadModel(model_file)
        enemy_model = storage.loadModel(enemy_model_file)
    enemy_player = TTTPlayerCNN(enemy_model, map_func, 1)

    m_wins, e_wins, draws = testAgainstPlayer(model, enemy_player, rounds, verbosity, map_func)

    with produced_data_lock:
        produced_data_dict[model_name][enemy_name] = m_wins
        produced_data_dict[enemy_name][model_name] = e_wins

def allToAllTest(model_files, names=None, rounds=2, threads_num=1, verbosity=0, map_funcs=None):
    if (names == None):
        names = [str(i) for i in range(len(model_files))]

    if (map_funcs == None):
        map_funcs = [mapToCnnInput for _ in model_files]

    # Clear abort file
    if (os.path.exists(abort_file)):
        os.remove(abort_file)

    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    model_file_lock = multiprocessing.Lock()
    produced_data_lock = multiprocessing.Lock()
    produced_data_dict = manager.dict()
    produced_data_dict.update({n0:manager.dict({n1:0 for n1 in names}) for n0 in names})

    producer_processes_awaiting = {n0:{n1:None for n1 in names} for n0 in names}
    producer_processes_started  = {}

    # Create testing processes
    for i,mi in enumerate(model_files):
        ni = names[i]
        produced_data_dict[ni][ni] = "X"
        for e in range(i+1, len(model_files)):
            ne = names[e]
            me = model_files[e]
            p = multiprocessing.Process(target=funcAbortWrapper, args=(testAgainstPlayerParallelProcess, me, mi, rounds, verbosity-1, map_funcs[e], model_file_lock, produced_data_lock, produced_data_dict, ne, ni))
            producer_processes_awaiting[ni][ne] = p

    # Run processes in batches of threads_num at a time
    while (len(producer_processes_awaiting)+len(producer_processes_started.keys()) > 0):
        if (os.path.exists(abort_file)):
            print(f"Aborting", flush=True)
            return {}

        while (len(producer_processes_started.keys()) < threads_num and len(producer_processes_awaiting) > 0):
            ni = next(iter(producer_processes_awaiting))
            ne = next(iter(producer_processes_awaiting[ni]))
            p = producer_processes_awaiting[ni][ne]
            del producer_processes_awaiting[ni][ne]
            if (len(producer_processes_awaiting[ni]) == 0):
                del producer_processes_awaiting[ni]
            if (p == None):
                continue
            p.start()
            producer_processes_started[(ni, ne)] = p
            print(f"Started player {ne} against enemy {ni}")

        pps = list(producer_processes_started.items())
        for (ni, ne), p in pps:
            if (not p.is_alive()):
                p.join()
                producer_processes_started.pop((ni, ne))
                print(f"Finished player {ne} against enemy {ni}")

        time.sleep(1)

    results_dict = {}
    results_dict.update(produced_data_dict)

    if (verbosity > 0):
        for k,v in results_dict.items():
            print(f"{k}: {list(v.values())}")
        for k,v in results_dict.items():
            filtered_values = [x for x in v.values() if x != "X"]
            print(f"{k}: {sum(filtered_values)}")

    return results_dict

def testModelVersions(model_name, fractions=8, rounds=10, threads_num=1):
    storage = ModelStorage()
    model_files = [fn for fn in storage.getStorageList() if model_name in fn]

    indexes = set()
    for mi in range(fractions):
        indexes.add(mi*len(model_files)//fractions)
    indexes.add(len(model_files)-1)
    indexes = list(indexes)
    indexes.sort()

    model_files_selected = []
    names_selected = []
    for mi in indexes:
        model_files_selected.append(model_files[mi])
        names_selected.append(model_files[mi].split("_")[6])
    print(names_selected)

    allToAllTest(model_files_selected, names_selected, rounds=rounds, threads_num=threads_num, verbosity=1)

def testOnUser(model_name, user_player, rounds, verbosity=2):
    storage = ModelStorage()
    model_files = [fn for fn in storage.getStorageList() if model_name in fn]
    model = storage.loadModel(model_files[-1])
    testAgainstPlayer(model, user_player, rounds, verbosity=verbosity)

def trainHero4(start_new, load_only=False, skip_training=False):
    model_grid_size = 16
    model_name = f"hero_4_{model_grid_size}x{model_grid_size}"

    if (start_new):
        conv_layers = 5
        depth = 3
        width = 4
        func0 = "elu"
        func1 = "tanh"
        func2 = "relu"
        dense_sizes = [
            model_grid_size * model_grid_size * width,
        ] * depth
        mid_activations = [func1] * (depth)
        m_hero = createCnnModel(
            model_grid_size,
            model_grid_size,
            conv_layers=conv_layers,
            dense_sizes=dense_sizes,
            conv_activation="swish",
            mid_activations=mid_activations,
            out_activation="softmax",
            input_grids=width,
        )
        model_games = 0
        model_inputs_trained = 0
    else:
        storage = ModelStorage()
        m_hero, fn_hero = storage.loadLatesModelContaining(model_name)
        print(f"Loading model {fn_hero}")
        numbering = fn_hero.split(model_name)[1].split(".")[0]
        model_games = int(numbering.split("_")[1])
        model_inputs_trained = int(numbering.split("_")[2])

    if (load_only):
        return m_hero

    if (not skip_training):
        train(
            model = m_hero,
            model_name = model_name,
            model_games = model_games,
            model_inputs_trained = model_inputs_trained,
            model_input_map_func = mapToCnnInput,
            max_training_data_size = MAX_TRAINING_DATA_KEPT,
            train_interval = 3*60,
            store_interval = 10*60,
            batch_size = 150,
            serial_rounds = 40,
            threads_num = 4,
            use_testing_thread = True,
            shortest_cutoff = 0,
            winner_weight = 1.0,
            loser_weight = 0.0,
            learning_rate = 0.0001,
            kept_models = 3,
            player_training_variants = 20,
            top_random_select_size = 0,
            weights_scale_coef = 1.0,
            weights_scale_uniform = False,
            fred_mistake_rate = 0.2,
            test_runs = 50,
            train_against_top_random_select_1 = True
        )

if __name__ == "__main__":
    #testVariations()
    #exit(0)
    #import pdb; pdb.set_trace()
    p_user = TTTPlayerUser("User")
    p_fred = TTTPlayerFred(0.2)
    #storage = ModelStorage()
    #p_other, _ = storage.loadLatesModelContaining("2025-03-09_07-02-28_hero_4_16x16_14720_6220000_depth-128_width-4_func-tanh_learnrate-0p00001_batch-400")
    #p_other = TTTPlayerCNN(p_other, mapToCnnInput)

    m_hero_4 = trainHero4(True, False, False)

    #testModelVersions("hero_4", 6, 40, 8)

    #testOnUser("hero_4_16x16_44240", p_user, 4)
    #testOnUser("hero_4_16x16_44240", p_fred, 30, 1)

    exit(0)
