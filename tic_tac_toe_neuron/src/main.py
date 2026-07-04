import os
# Disable tensorflow warning message about missing high-performance instructions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Allow GPU memory growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#exit(0)

from argparse import ArgumentParser

from global_params import *
from model import *
from play import *
from player import *
from storage import *
from abort import *
from testing import *
from training import *

# Customized training of a model with some specific parameters
def trainCustomModel0(
    name="model0",
    input_width = 4,
    conv3x3_layers=10,
    conv3x3_channels=4,
    conv5x5_layers=10,
    conv5x5_channels=4,
    input_to_dense_propagate=True,
    dense_width_multiplier=1,
    dense_layers=3,

    max_training_data_size=50,
    train_interval=3*60,
    serial_rounds=40,
    threads_num=4,
    shortest_cutoff=0,
    winner_weight=1.0,
    loser_weight=0.0,
    learning_rate=0.0001,
    player_training_variants=20,
    top_random_select_size=0,
    top_select_equal=True,
    weights_scale_coef=1.0,
    weights_scale_uniform=False,
    train_against_best = True,

    fred_mistake_rate=0.2,
    test_runs=50,

    start_new=True,
    load_only=False,
    ):

    model_grid_size = 16
    model_name = f"{name}_{model_grid_size}x{model_grid_size}"

    if (start_new):
        dense_layers = dense_layers
        width = input_width
        func0 = "elu"
        func1 = "tanh"
        func2 = "relu"
        func3 = "swish"
        func4 = "softmax"
        dense_sizes = [
            model_grid_size * model_grid_size * width * dense_width_multiplier,
        ] * dense_layers
        mid_activations = [func1] * (dense_layers)
        m_hero = createCnnModel(
            model_grid_size,
            model_grid_size,
            conv3x3_layers=conv3x3_layers,
            conv3x3_channels=conv3x3_channels,
            conv5x5_layers=conv5x5_layers,
            conv5x5_channels=conv5x5_channels,
            input_to_dense_propagate=input_to_dense_propagate,
            dense_sizes=dense_sizes,
            conv_activation=func3,
            mid_activations=mid_activations,
            out_activation=func4,
            input_grids=width,
        )
        model_games = 0
        model_inputs_trained = 0
    else:
        storage = ModelStorage()
        m_hero, fn_hero = storage.loadLatesModelContaining(model_name)
        print(f"Loading model {fn_hero}")
        numbering = fn_hero.split(model_name)[1].split(".")[1]
        model_games = int(numbering.split("_")[1])
        model_inputs_trained = int(numbering.split("_")[2])

    if (load_only):
        return m_hero

    train(
        model = m_hero,
        model_name = model_name,
        model_games = model_games,
        model_inputs_trained = model_inputs_trained,
        max_training_data_size = MAX_TRAINING_DATA_SIZE * max_training_data_size,
        accumulate_training_data=False,
        train_interval = train_interval,
        store_interval = 10*60,
        batch_size = 150,
        serial_rounds = serial_rounds,
        threads_num = threads_num,
        use_testing_thread = True,
        shortest_cutoff = shortest_cutoff,
        winner_weight = winner_weight,
        loser_weight = loser_weight,
        learning_rate = learning_rate,
        kept_models = 3,
        player_training_variants = player_training_variants,
        top_random_select_size = top_random_select_size,
        top_select_equal = top_select_equal,
        weights_scale_coef = weights_scale_coef,
        weights_scale_uniform = weights_scale_uniform,
        train_against_top_random_select_1 = train_against_best,
        fred_mistake_rate = fred_mistake_rate,
        test_runs = test_runs,
    )

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Tic Tac Toe neural network training and testing",
        description="This program trains a neural network model to play Tic Tac Toe using self-play algorithm. The training is heavily customized and parallelized.")
    parser.add_argument("--test-versions",
        nargs=4, metavar=("MODEL_NAME_PREFIX", "VERSIONS_NUM", "ROUNDS_PER_TEST", "PARALLEL_THREADS"),
                        help="Instead of training, test different versions of the model against each other. Requires 4 parameter values: model name prefix, number of versions, rounds per test, number of parallel threads to use for testing. Overrides all training-related arguments.")
    parser.add_argument("--model-name",
        default="model0", help="Model name prefix to use for training or testing. Default value is 'model0'.")
    parser.add_argument("--max-training-data-size",
        type=int, default=50, help="Maximum training data size multiplier. Default value is 50.")
    parser.add_argument("--train-interval",
        type=int, default=3*60, help="Interval between training calls in seconds. Default value is 180 (3 minutes).")
    parser.add_argument("--serial-rounds",
        type=int, default=40, help="Number of rounds to play in each training call. Default value is 40.")
    parser.add_argument("--threads-num",
        type=int, default=4, help="Number of parallel threads to use for training. Default value is 4.")
    parser.add_argument("--shortest-cutoff",
        type=int, default=0, help="Shortest cutoff for training data. Default value is 0.")
    parser.add_argument("--winner-weight",
        type=float, default=1.0, help="Weight for winner's training data. Default value is 1.0.")
    parser.add_argument("--loser-weight",
        type=float, default=0.0, help="Weight for loser's training data. Default value is 0.0.")
    parser.add_argument("--learning-rate",
        type=float, default=0.0001, help="Learning rate for training. Default value is 0.0001.")
    parser.add_argument("--player-training-variants",
        type=int, default=20, help="Number of player training variants to use for training. Default value is 20.")
    parser.add_argument("--top-tandom-select-size",
        type=int, default=0, help="Size of top random select for training. Default value is 0.")
    parser.add_argument("--weights-scale-coef",
        type=float, default=1.0, help="Coefficient for scaling game turn weights during training. Default value is 1.0.")
    parser.add_argument("--weights-scale-uniform",
        action="store_true", help="Set training data weights to scale game turn weights uniformly based on the length of the game instead of from first turn to last.")
    parser.add_argument("--train-against-best",
        action="store_true", help="Train against version of model using always it's current best selections. Should lead to constant direct improvement. Downside is that only data from the not-best of the players can be used for training.")
    parser.add_argument("--fred-mistake-rate",
        type=float, default=0.2, help="Mistake rate for Fred player during testing. Default value is 0.2.")
    parser.add_argument("--test-runs",
        type=int, default=50, help="Number of test runs to perform during testing against Fred. Default value is 50.")

    parser.add_argument("--start-new",
        action="store_true", help="Start training a new model instead of loading the latest one.")
    parser.add_argument("--input-width",
        type=int, default=4, help="Width of the input data for the model. Default value is 4.")
    parser.add_argument("--conv3x3-layers",
        type=int, default=1, help="Number of 3x3 convolutional layers in the model. Default value is 1.")
    parser.add_argument("--conv3x3-channels",
        type=int, default=4, help="Number of channels in 3x3 convolutional layers. Default value is 5.")
    parser.add_argument("--conv5x5-layers",
        type=int, default=1, help="Number of 5x5 convolutional layers in the model. Default value is 1.")
    parser.add_argument("--conv5x5-channels",
        type=int, default=4, help="Number of channels in 5x5 convolutional layers. Default value is 5.")
    parser.add_argument("--input-to-dense-propagate",
        action="store_true", default=True, help="Propagate input directly to dense layers in the model. Default is True.")
    parser.add_argument("--dense-width-multiplier",
        type=int, default=3, help="Multiplication of internal dense layers width.")
    parser.add_argument("--dense-layers",
        type=int, default=2, help="Number of dense layers in the model. Default value is 4.")

    args = parser.parse_args()

    #p_user = TTTPlayerUser("User")
    #p_fred = TTTPlayerFred(0.2)
    #testAgainstUser("model-t6_16x16_33640", p_user, 4)
    #testAgainstUser("hero_4_16x16_44240", p_fred, 30, 1)
    #exit(0)

    if (args.test_versions):
        name             =     args.test_versions[0]
        versions_num     = int(args.test_versions[1])
        rounds_per_test  = int(args.test_versions[2])
        parallel_threads = int(args.test_versions[3])
        testModelVersions(name, versions_num, rounds_per_test, parallel_threads)
        exit(0)

    #model_name = f"model-t12-0_3x3-1x10_5x5-1x10"
    #storage = ModelStorage()
    #model_files = [fn for fn in storage.getStorageList() if model_name in fn]
    #for m in model_files:
    #    m_hero, fn_hero = storage.loadLatesModelContaining(m)
    #    print(f"Loading model {fn_hero}")
    #    print(m_hero.summary())
    #exit()

    trainCustomModel0(
    name=f"model-t15-1_3x3-{args.conv3x3_layers}x{args.conv3x3_channels}_5x5-{args.conv5x5_layers}x{args.conv5x5_channels}_{str(args.input_to_dense_propagate)}",

    start_new=args.start_new,
    input_width=args.input_width,
    conv3x3_layers=args.conv3x3_layers,
    conv3x3_channels=args.conv3x3_channels,
    conv5x5_layers=args.conv5x5_layers,
    conv5x5_channels=args.conv5x5_channels,
    input_to_dense_propagate=args.input_to_dense_propagate,
    dense_width_multiplier=args.dense_width_multiplier,
    dense_layers=args.dense_layers,

    load_only=False,

    max_training_data_size=1000,
    train_interval=20*60,
    serial_rounds=50,
    threads_num=8,
    shortest_cutoff=0,
    winner_weight=1.0,
    loser_weight=0.0,
    learning_rate=0.000001,
    player_training_variants=50,
    top_random_select_size=8,
    top_select_equal=True,
    weights_scale_coef=3.0,
    weights_scale_uniform=False,
    train_against_best=True,

    fred_mistake_rate=0.2,
    test_runs=50,
    )

    #testModelVersions("hero_4", 3, 4, 8)

    exit(0)
