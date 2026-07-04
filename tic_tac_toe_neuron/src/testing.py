import os
import multiprocessing
import time

from abort import *
from model import *
from play import *
from player import *
from storage import *
from global_params import *

# Test a player against an enemy player
# Runs given number of rounds (half of them with player as X and half with player as O)
# and returns player wins, losses and draws
def testAgainstPlayer(player, enemy_player, rounds, verbosity=0):
    model_wins = 0
    enemy_wins = 0
    draws      = 0
    invalids   = 0
    none       = 0

    if (verbosity > 0):
        print(f"Testing against enemy player X", flush=True)
    ttt_play = TicTacToePlay(grid_size, grid_size, enemy_player, player, win_strike_length)
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
    ttt_play = TicTacToePlay(grid_size, grid_size, player, enemy_player, win_strike_length)
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
           +f"  player wins: {model_wins}\n"\
           +f"   enemy wins: {enemy_wins}\n"\
           +f"        draws: {draws}", flush=True)

    return model_wins, enemy_wins, draws

# Parallel process wrapper around testAgainstPlayer
def testAgainstPlayerParallelProcess(
    model_file,
    enemy_model_file,
    rounds,
    verbosity,
    model_file_lock,
    produced_data_lock,
    produced_data_dict,
    model_name,
    enemy_name,
    ):
    # Initialize random seed
    seed()

    # Add random delay to avoid all processes accessing resources at the same time
    time.sleep(random()*4)

    storage = ModelStorage()

    with model_file_lock:
        model       = storage.loadModel(model_file)
        enemy_model = storage.loadModel(enemy_model_file)

    player       = TTTPlayerCNN(model, 1)
    enemy_player = TTTPlayerCNN(enemy_model, 1)

    m_wins, e_wins, draws = testAgainstPlayer(player, enemy_player, rounds, verbosity)

    with produced_data_lock:
        produced_data_dict[model_name][enemy_name] = m_wins
        produced_data_dict[enemy_name][model_name] = e_wins

# Run parallel testing of given models all-against-all and return results in a dictionary
# with format {model_name_A:{model_name_B:wins_of_A_over_B, ...}, ...}
def allToAllTestParallel(model_files, model_names=None, rounds=2, threads_num=1, verbosity=0):
    assert (threads_num >= 1), "Number of threads must be at least 1"

    if (model_names == None):
        model_names = [f"model_{i}" for i in range(len(model_files))]

    # Clear abort file
    if (os.path.exists(abort_file)):
        os.remove(abort_file)

    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    model_file_lock = multiprocessing.Lock()
    produced_data_lock = multiprocessing.Lock()
    produced_data_dict = manager.dict()
    produced_data_dict.update({n0:manager.dict({n1:0 for n1 in model_names}) for n0 in model_names})

    producer_processes_awaiting = {n0:{n1:None for n1 in model_names} for n0 in model_names}
    producer_processes_started  = {}

    # Create testing processes
    for i,mi in enumerate(model_files):
        ni = model_names[i]
        produced_data_dict[ni][ni] = "X"
        for e in range(i+1, len(model_files)):
            ne = model_names[e]
            me = model_files[e]
            p = multiprocessing.Process(target=funcAbortWrapper, args=(testAgainstPlayerParallelProcess,
                me, mi, rounds, verbosity-1, model_file_lock, produced_data_lock, produced_data_dict, ne, ni))
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
        max_name_length = max([len(n) for n in model_names])
        max_value_length = max([max([0]+[len(str(v)) for v in d.values()]) for d in results_dict.values()])
        for k,v in results_dict.items():
            print(f"{k.ljust(max_name_length)}: " + ", ".join([str(vv).rjust(max_value_length) for vv in v.values()]))
        for k,v in results_dict.items():
            filtered_values = [x for x in v.values() if x != "X"]
            print(f"{k.ljust(max_name_length)}: {sum(filtered_values)}")

#        for k,v in results_dict.items():
#            print(f"{k}: {list(v.values())}")
#        for k,v in results_dict.items():
#            filtered_values = [x for x in v.values() if x != "X"]
#            print(f"{k}: {sum(filtered_values)}")

    return results_dict

# Test different version of model against each other
# Selects given number of samples of model versions (with last and first versions always included) and runs all-against-all testing on them
def testModelVersions(model_name, samples=8, rounds=10, threads_num=1, names_extraction_func=lambda x: x.split("_")[9], verbosity=1):
    assert (samples >= 2), "At least 2 samples must be selected to enable testing"

    storage = ModelStorage()
    model_files = [fn for fn in storage.getStorageList() if model_name in fn]

    indexes = set()
    for mi in range(samples-1):
        indexes.add(mi*len(model_files)//samples)
    indexes.add(len(model_files)-1)
    indexes = list(indexes)
    indexes.sort()

    model_files_selected = []
    names_selected = []
    for mi in indexes:
        model_files_selected.append(model_files[mi])
        names_selected.append(names_extraction_func(model_files[mi]))

    if (verbosity > 0):
        print(f"Testing following model versions: {names_selected}")

    allToAllTestParallel(model_files_selected, names_selected, rounds=rounds, threads_num=threads_num, verbosity=verbosity)

# Run a series of tests of a model against a user player and print results
def testAgainstUser(model_name, user_player, rounds, verbosity=2):
    storage = ModelStorage()
    model_files = [fn for fn in storage.getStorageList() if model_name in fn]
    model  = storage.loadModel(model_files[-1])
    player = TTTPlayerCNN(model, 1)
    testAgainstPlayer(player, user_player, rounds, verbosity=verbosity)

