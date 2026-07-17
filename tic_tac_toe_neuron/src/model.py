import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Input, Dropout, Concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
from random import randint

def createCnnModel(
    input_grids,
    input_grid_size,
    output_grid_size,
    conv3x3_depth=1,
    conv3x3_channels=1,
    conv5x5_depth=1,
    conv5x5_channels=1,
    propagate_input_to_partial_dense=True,
    partial_dense_depth=1,
    partial_dense_width=None,
    full_dense_depth=1,
    full_dense_width=None,
    conv_activation="swish",
    dense_activation="tanh",
    out_activation="softmax",
    ):
    """
    Defines a CNN model with more or less the following structure:

           (I)
            |
       +----+----+
       |    |    |
      (C3) (C5)  |
       |    |    |
      (D)  (D)  (D)
       |    |    |
       +----+----+
            |
           (D)
            |
           (O)

    Legend:
        (I)  - Input layer (Multiple 2D grids of values)
        (C3) - 3x3 convolution layer per grid with multiple channels
        (C5) - 5x5 convolution layer per grid with multiple channels
        (D)  - Dense layer per grid
        (O)  - Output layer (Single dense 2D grid of values)
    """
    # Default setting is same width as one input or output grid, whichever is bigger
    if (partial_dense_width == None):
        partial_dense_width = max(input_grid_size, output_grid_size)**2
    if (full_dense_width == None):
        full_dense_width = max(input_grid_size, output_grid_size)**2

    # Input contains a number of grids
    input_shape  = (input_grids, input_grid_size, input_grid_size, 1)
    # Output contains a single grid
    output_shape = (output_grid_size, output_grid_size, 1)

    input_layer = Input(input_shape)
    input_layer_split = tf.split(input_layer, num_or_size_splits=input_grids, axis=1)
    input_layer_split = [tf.squeeze(t, axis=1) for t in input_layer_split]

    # 3x3 convolution layers
    conv3x3_layer = input_layer_split
    for i in range(conv3x3_depth):
        conv3x3_layer = [Conv2D(conv3x3_channels, (3, 3), padding="same", activation=conv_activation)(l) for l in conv3x3_layer]

    # 5x5 convolution layers
    conv5x5_layer = input_layer_split
    for i in range(conv5x5_depth):
        conv5x5_layer = [Conv2D(conv5x5_channels, (5, 5), padding="same", activation=conv_activation)(l) for l in conv5x5_layer]

    partial_dense_input = []
    if (conv3x3_depth):
        partial_dense_input += conv3x3_layer
    if (conv5x5_depth):
        partial_dense_input += conv5x5_layer
    if (propagate_input_to_partial_dense):
        partial_dense_input += input_layer_split

    assert (len(partial_dense_input) > 0), "No input to dense layers. At least one of conv3x3_depth, conv5x5_depth or propagate_input_to_partial_dense must be non-zero."

    # Flatten individual convolution outputs
    partial_dense_input = [Flatten()(l) for l in partial_dense_input]

    # Partial dense layers
    dense_input = partial_dense_input
    for i in range(partial_dense_depth):
        dense_input = [Dense(partial_dense_width, activation=dense_activation)(l) for l in dense_input]

    # Concatenate and flatten all
    dense_input_flat = tf.stack(dense_input, axis=1)
    dense_input_flat = Flatten()(dense_input_flat)

    # Full dense layers
    full_dense_layer = dense_input_flat
    for i in range(full_dense_depth):
        full_dense_layer = Dense(full_dense_width, activation=dense_activation)(full_dense_layer)

    # Last layer always has the size of the output grid
    full_dense_layer   = Dense(output_grid_size * output_grid_size, activation=out_activation)(full_dense_layer)
    dropout_layer = Dropout(0.2)(full_dense_layer)
    output_layer  = Reshape(output_shape)(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=CategoricalCrossentropy(from_logits=False), weighted_metrics=["categorical_crossentropy"])

    print(model.summary())

    return model

def dataToStr(data):
    """
    Converts a tensor to a string.
    """
    shape = data.shape
    if (len(shape) == 1):
        return ",".join([f"{float(e):.2f} " for e in data])
    if (len(shape) == 2):
        S = ""
        for line in data:
            S += ",".join([f"{float(e):.2f} " for e in line]) + "\n"
        return S
    if (len(shape) == 3):
        S = ""
        for line in data:
            S += ",".join([f"{float(e[0]):.2f} " for e in line]) + "\n"
        return S
    if (len(shape) == 4):
        S = ""
        for grid in data:
            S += dataToStr(grid) + "\n"
        return S

class TrainingData:
    def __init__(self, input_data, weight_data, ref_output_data):
        self.input_data      = input_data
        self.weight_data     = weight_data
        self.ref_output_data = ref_output_data

    def concat(self, other):
        return TrainingData(
            tf.concat([self.input_data, other.input_data], axis=0),
            tf.concat([self.weight_data, other.weight_data], axis=0),
            tf.concat([self.ref_output_data, other.ref_output_data], axis=0)
        )

    def truncate(self, size):
        if (size >= len(self.input_data)):
            return self

        return TrainingData(
            self.input_data[len(self.input_data)-size:],
            self.weight_data[len(self.weight_data)-size:],
            self.ref_output_data[len(self.ref_output_data)-size:]
        )

    def trainModel(self, model, batch_size=None, epochs=1, max_data_size=None):
        if (max_data_size == None):
            max_data_size = len(self.input_data)

        if (batch_size == None):
            batch_size = len(self.input_data)

        if (False):
            S = ""
            t = 80
            #for i in [0, 1, 2, -3, -2, -1]:
            #    S += "----------------\n"
            #    S += f"weights: {2*i}\n"
            #    S += "----------------\n"
            #    S += str(model.weights[2*i][0]) + "\n"

            #for i in [0, 1, 2, -3, -2, -1]:
            #    S += "----------------\n"
            #    S += f"biases: {2*i+1}\n"
            #    S += "----------------\n"
            #    S += str(model.weights[2*i+1]) + "\n"

            turn = randint(0, len(self.input_data)-t-1)
            for i in range(t):
                S += "----------------\n"
                S += f"turn: {turn}\n"
                S += "----------------\n"
                S += f"input:\n"
                S += dataToStr(self.input_data[turn])
                S += f"weight:\n"
                S += dataToStr(self.weight_data[turn])
                S += f"\nref output:\n"
                S += dataToStr(self.ref_output_data[turn])
                S += f"output:\n"
                S += dataToStr(model(self.input_data[turn:turn+1])[0])
                #model.evaluate(self.input_data[turn:turn+1], self.ref_output_data[turn:turn+1], verbose=0)
                #result = model.get_metrics_result()
                #S += f"Training result: loss: {float(result['loss']):.2f}, categorical_accuracy: {float(result['categorical_accuracy']):.2f}\n"
                #for i in [0, 1, 2, -8, -4, -2]:
                #    S += "----------------\n"
                #    S += f"outputs: {i}\n"
                #    S += "----------------\n"
                #    ablation_model = Model(inputs=model.inputs, outputs=model.layers[i].output)
                #    S += str(ablation_model(self.input_data[turn:turn+1])[0]) + "\n"
                turn += 1
            print("Writing tmp file")
            with open("tmp", "w") as F:
                F.write(S)

        for i in range(0, len(self.input_data), max_data_size):
            model.fit(self.input_data[i:i+max_data_size], self.ref_output_data[i:i+max_data_size], sample_weight=self.weight_data[i:i+max_data_size], epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.0, shuffle=True, use_multiprocessing=True, workers=8)
        result = model.get_metrics_result()
        print(f"Training result: loss: {float(result['loss']):.2f}, categorical_accuracy: {float(result['categorical_crossentropy']):.2f}")
        assert (not tf.math.is_nan(result["loss"])), "Loss is NaN! Training failed most probably due to too high learning rate or too big batch size. Try to decrease them and train again."
