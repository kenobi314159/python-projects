import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, MaxPooling2D, Flatten, Dense, Reshape, Input, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from random import randint

def createCnnModel(input_grid_size, output_grid_size, conv_layers=64, dense_sizes=None, conv_activation="linear", mid_activations=["relu", "relu", "relu", "relu"], out_activation="softmax", input_grids=1):
    """
    Defines a simple CNN model.
    """
    if (dense_sizes == None):
        # Default setting is two layers each as big as one input or output, whichever is bigger
        dense_sizes = [max(input_grid_size, output_grid_size)**2] * 2

    # Input contains a number of grids
    input_shape  = (input_grids, input_grid_size, input_grid_size, 1)
    # Input contains a single grid
    output_shape = (output_grid_size, output_grid_size, 1)

    layers = [
        Input(input_shape),
    ]

    if (conv_layers > 0):
        # Convolution is done separately for each input grid
        layers.append(Conv3D(conv_layers, (1, 3, 3), activation=conv_activation))

    layers.append(Flatten())

    for a,s in zip(mid_activations, dense_sizes):
        layers.append(Dense(s, activation=a))

    # Last layer always has the size of the output grid
    layers += [
        Dense(output_grid_size * output_grid_size, activation=out_activation),
        Dropout(0.2),
        Reshape(output_shape)
    ]

    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=CategoricalCrossentropy(from_logits=False), weighted_metrics=["categorical_crossentropy"])

    #print(model.summary())

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

class trainingData:
    def __init__(self, input_data, weight_data, ref_output_data):
        self.input_data      = input_data
        self.weight_data     = weight_data
        self.ref_output_data = ref_output_data

    def concat(self, other):
        return trainingData(
            tf.concat([self.input_data, other.input_data], axis=0),
            tf.concat([self.weight_data, other.weight_data], axis=0),
            tf.concat([self.ref_output_data, other.ref_output_data], axis=0)
        )

    def truncate(self, size):
        if (size >= len(self.input_data)):
            return self

        return trainingData(
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
