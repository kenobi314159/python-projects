# Tic Tac Toe neural network

This project implements training of a neural network to play Tic Tac Toe using the Keras library.
The training is done by having the model play against itself, generating training data from the games played.
The model is continuously stored in files to enable resuming training later or comparing multiple models against each other.

## Training flow

The program runs using multiple parallel processes.
There is several "producer" processes which use the latest stored version of the model, run games with it to generate training data, and store the data in a shared memory.
Then there is a single "consumer" process which reads the training data from the shared memory at configured time intervals, trains the model on it, and stores the new version of the model in a file for the producers to use.
There is also an optional "testing" process, which runs the latest stored model against a simple hand coded Tic Tac Toe player called "Fred" to provide continuous feedback on the model's performance in the console.

## Usage

**This project is still work-in-progress, configuration and usage is not very convenient at this moment.**

### Configuration

The project allows for configuration of many different parameters of the trained model and the training process itself.
At this moment, all the parameters need to be modified in the src/main.py file before running it either in the global variables at the top of the file or in the main training function `trainHero4` near the bottom.

- grid_size - Size of the Tic Tac Toe grid the training is done on (3 for 3x3, 4 for 4x4, etc.).
- win_strike_length - Number of symbols in a row needed to win the game.
  Regular Tic Tac Toe has 5, but for begenning of training it is better to use smaller values and first teach the network on simpler rules before going to more difficult ones.
- MAX_TRAINING_DATA_SIZE - Maximum number of training data samples to keep in memory.
  This value helps with overflowing of computer memory capacity.
- model_grid_size - Size of the grid used as input for the model.
  This may be different from the grid size of the game.
  The program contains mechanisms to resize the game grid up or down when passing it to the model.
  This parameter influences the width of the neural network.
- createCnnModel parameters - These are parameters defining the structure of the neural network like width, depth and activation functions.
  See source code for more information.
- training flow parameters
  - max_training_data_size - Maximum size of training data set.
    After reaching this size, the oldest data starts to be overwritten.
  - train_interval - Number of seconds between training is done on generated data.
  - store_interval - Number of seconds between storing the model to an output file for reusage.
  - batch_size - Batch size for training calls.
  - serial_rounds - Number of serial rounds (games) played by a producer process before storing the data and updating the model version.
  - threads_num - Total number of processes used in the program.
  - use_testing_thread - Enablement of the testing process.
  - shortest_cutoff - Parameter of elimination of a portion of the longest Tic Tac Toe games from the training data.
    (Rounds that finished faster may be considered higher-quality training data.)
  - winner_weight - Weight of the training data samples where the model won the game.
  - loser_weight - Weight of the training data samples where the model lost the game.
  - learning_rate - Learning rate for the training process.
    Higher learning rate may lead to faster training, but also to more unstable training and worse final results.
    Lower learning rate may lead to more stable training and better final results, but also to much longer training time.
  - kept_models - Number of latest model iterations that are used for generating training data.
    Using multiple different versions may lead to higher variations of data and avoiding local minimums in the training.
  - player_training_variants - Number of training samples generated from each turn of each game.
    The program contains mechanisms to generate random valid alternatives of every state of the game.
    This allows to greatly increase amount of training data generated from each game.
  - top_random_select_size - Number of highest model's output from which the actual desired turn is selected.
    This can be used to introduce some randomness into the model's decisions thus allowing to avoid local minimums in the training.
    The random selection is weighted by the value of each potential outputs, so higher values have higher chance to be selected.
    For value 1, the model's highest output is always selected, leading to deterministic behavior.
    For value 0, the weighted random selection is done from all outputs.
  - weights_scale_coef - Scaling coefficient for the weights of the training data samples based on the remaining length of the game.
    This can be used to decrease the weight of training samples which are far from the end of the game, creating higher rewards for turns that are closer to the end of the game.
    For weights_scale_coef 0.0, all samples have the same weight coefficient 1.0.
    For weights_scale_coef 1.0, the scaling is linear. So from the last to first turn it goes 1/1, 1/2, 1/3, etc.
    For weights_scale_coef over 1.0, the scaling goes down slower and slower.
    For example, for weights_scale_coef 4.0, it takes 16 turns to get down to coefficient 1/2.
    For weights_scale_coef 5.0, it takes 32 turns.
  - weights_scale_uniform - When set to True, switches the weights scaling to uniform across all turns in one game.
    The weight is then based on the above calculation made only for the first turn.
    This leads to shorted games having overall higher weight than longer games creatng incentive to finish the game faster.
  - fred_mistake_rate - Rate of mistakes made by the testing player Fred.
    This allows to monitor improvements of the model even at stages when it is still very bad.
  - test_runs - Number of runs made by testing process before printing results.
    (Higher number leads to less frequent updates in the console, but more representative results.)
  - train_against_top_random_select_1 - Instead of using the same value of top_random_select_size for both playing sides and then train on both data, use top_random_select_size for one player and 1 for the other. Then only train based on the data of the first one.

### Running

After configuration, the training can be started, aborted, paused or continued using following steps:

1. Start the program using `python3 src/main.py` with the `trainHero4` parameter `start_new` set to True.
2. Abort the program safely by creating file named "abort" in the working directory.
   All parallel processes will finish their job one by one and exit.
3. Pause the program by creating file named "pause" in the working directory.
   All parallel processes will finish their job one by one and wait.
   When the file is removed, the program will resume again.
4. To resume training using the latest version of a model from previous run, start the program using `python3 src/main.py` with the `trainHero4` parameter `start_new` set to False.
   The program will load the latest model from the output directory and continue training it.

### Logging

As training progresses, the program prints out two types of logs in files.

There is the `out.log` file, which contains weights outputed by one of the models for a randomly selected turn combined with input game state for that turn.
The values in this log are color-coded, so by printing them in the console you can easily see which turns the model currently prefers and which it doesn't.
This can give you some insight into what is currently the structure of teh model's output and if, for example, it isn't stuck on a constant output.
The `out.log` can be printed nicely colored using the following command:

```bash
while true; do clear; cat out.log; sleep 4; done
```

The second type of logs are the `out.log_X` files.
These files are also printed randomly and they contain the final states of randomly selected Tic Tac Toe matches.
This allows you to see what kind of matches the model is currently able to play.
It can be monitored using the following command:

```bash
watch -n 1 'cat out.log_*'
```

## Results

I have performed extensive training of models with various parameters, sometimes for several days straight.
Due to limitations of memory size, the largest model I could train has configuration `model_grid_size=16`, `width=4`, `depth=128` leading to a dense network of 1024x128 neurons and 135 million parameters.
The best result I could get was a model which seems like it's trying to play Tic Tac Toe, but still makes many mistakes.
It couldn't consistently beat the hand coded testing player Fred even with relatively high Fred mistake rate set.
This means that it would stand no chance against a human player.
For now, my conclusion is that there is either something wrong in my training approach or there is a need for much larger model to be able to learn the game.
