from data_processing import DataProcessor
import os
import numpy as np
import sys
import random as rand

HIDDEN_LAYER_SIZE = 10
LEARNING_RATE = 0.1
LINEAR_LAMBDA = 0.2

NUM_EPOCHS = 100

FFNN_WEIGHTS_FILE = 'ffnn_weights.npz'

verbose = False


class Neuron:
    def __init__(self, input_dimension, weights=None, activation='sigmoid'):
        self.input_dimension = input_dimension
        self.activation = activation
        self.b = weights[-1] if weights is not None else None
        self.w = weights[:-1] if weights is not None else None
        self.BIAS = -1

        # Init weights if needed
        if self.w is None:
            self.init_weights()

    def init_weights(self):
        self.w = np.random.rand(self.input_dimension)
        for i, wi in enumerate(self.w):
            self.w[i] /= 2.0
        self.b = rand.random()

    def process_input(self, x):
        if len(x) != self.input_dimension:
            raise ValueError
        tsum = np.dot(self.w, x) + (self.b * self.BIAS)
        for wi in self.w:
            if np.isnan(wi):
                raise ValueError
        ret = self.activate(tsum)
        if np.isnan(ret):
            raise ValueError

        return ret

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1.0/(1.0 + np.exp(-x))
        elif self.activation == 'linear':
            return LINEAR_LAMBDA * x

    def update_weights(self, dWeights, dBias):
        if len(dWeights) != self.input_dimension:
            raise ValueError
        for i in range(self.input_dimension):
            self.w[i] += dWeights[i]
        self.b += dBias

    def get_weights(self):
        t_weights = np.append(self.w, self.b)
        return t_weights


class Layer:
    def __init__(self, input_dimension, output_dimension, weights=None, activation='sigmoid', is_input=False, is_output=False):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.is_input = is_input
        self.is_output = is_output

        # Initialize neurons
        self.neurons = [None] * output_dimension
        if weights is not None:
            w_i = weights
        else:
            w_i = [None] * output_dimension
        for i in range(output_dimension):
            self.neurons[i] = Neuron(input_dimension=input_dimension, activation=activation, weights=w_i[i])

    def predict(self, x):
        if len(x) != self.input_dimension:
            raise ValueError
        return [neuron.process_input(x) for neuron in self.neurons]

    def train(self, x, y_target):
        if len(x) != self.input_dimension:
            raise ValueError
        if self.is_output and len(y_target) != self.output_dimension:
            raise ValueError

        Y = self.predict(x)

        if self.is_output:
            next_error = [LEARNING_RATE * y_target[i] - Y[i] for i in range(self.output_dimension)]
        else:
            next_error = self.next_layer.train(Y, y_target)

        for i, neuron in enumerate(self.neurons):
            # Layer output error
            yi_error = (1.0 - Y[i]) * Y[i]
            # Layer error
            layer_error = next_error * yi_error

            W = np.copy(neuron.w)

            # Generate gradient dw for this neuron
            dW_i = np.zeros(neuron.input_dimension)
            for j, x_j in enumerate(x):
                dW_i[j] = layer_error * x_j
            dBias = layer_error * neuron.BIAS
            neuron.update_weights(dW_i, dBias)

    def process_input(self, x):
        return self.predict(x)


# Predictors used: columns 2, 5, 6, 7, 9
class FFNN:
    def __init__(self, input_dimension, output_dimension=1, hidden_layer_size=HIDDEN_LAYER_SIZE, learning_rate=LEARNING_RATE):
        self.input_dim = input_dimension
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.output_dimension = output_dimension
        file_weights = None
        i_weights = None
        h_weights = None
        o_weights = None
        if os.path.exists(FFNN_WEIGHTS_FILE):
            print('Importing FFNN weights from existing file...')
            file_weights = np.load(FFNN_WEIGHTS_FILE)
            h_weights = file_weights['arr_0']
            o_weights = file_weights['arr_1']
            i_weights = file_weights['arr_2']
        else:
            print("No FFNN weight file found, creating new FFNN.")

        # Create output layer
        self.output_layer = Layer(input_dimension=hidden_layer_size, output_dimension=output_dimension,
                                  weights=o_weights, is_output=True)

        # Create hidden layer
        self.hidden_layer = [None] * hidden_layer_size
        for i in range(hidden_layer_size):
            t_weights = None
            if file_weights is not None:
                t_weights = h_weights[i]
            self.hidden_layer[i] = Neuron(input_dimension=self.input_dim, activation='linear', weights=t_weights)

        # Create input layer
        self.input_layer = [None] * input_dimension
        for i in range(input_dimension):
            t_weights = None
            if file_weights is not None:
                t_weights = i_weights[i]
            self.input_layer[i] = Neuron(input_dimension=self.input_dim, activation='sigmoid', weights=t_weights)

    def predict(self, x):
        if len(x) != self.input_dim:
            raise ValueError
        X = [neuron.process_input(x) for neuron in self.input_layer]
        Y = [neuron.process_input(X) for neuron in self.hidden_layer]
        O_k = self.output_layer.predict(Y)
        return O_k

    def train(self, x, target):
        if len(x) != self.input_dim:
            raise ValueError
        if len(target) != self.output_dimension:
            raise ValueError

        # Output of input layer
        X = [neuron.process_input(x) for neuron in self.input_layer]
        # Output of hidden layer
        Y = [neuron.process_input(x) for neuron in self.hidden_layer]
        # Output of output layer
        Oh = self.output_layer.predict(Y)

        # For each output value from output layer
        for i, output_neuron in enumerate(self.output_layer.neurons):
            o_i = Oh[i]
            # Error for this output value
            next_error = self.learning_rate * (target[i] - o_i)

            # Layer output error
            yi_error = (1.0 - o_i) * o_i
            # Layer error
            layer_error = next_error * yi_error

            # Store copy of neuron weights before updating
            output_neuron = self.output_layer.neurons[i]
            W_O = np.copy(output_neuron.w)

            # Generate gradient dw for output layer
            dW_O = np.zeros(self.hidden_layer_size)
            for j, hidden_output in enumerate(Y):
                dW_O[j] = layer_error * hidden_output
            dBias_O = layer_error * output_neuron.BIAS
            output_neuron.update_weights(dW_O, dBias_O)

            # Update hidden layer weights
            for j, hidden_neuron in enumerate(self.hidden_layer):
                # Layer output error
                yj_error = (1.0 - Y[j]) * Y[j]
                # Layer error
                hidden_error = layer_error * W_O[j] * yj_error

                # Store neuron weights before updating
                W_H = np.copy(hidden_neuron.w)

                # Generate gradient dw for hidden layer
                dW_H = np.zeros(hidden_neuron.input_dimension)
                for k, input_output in enumerate(X):
                    dW_H[k] = hidden_error * input_output
                dBias_Y = hidden_error * hidden_neuron.BIAS
                hidden_neuron.update_weights(dW_H, dBias_Y)

                # Update input layer weights
                for k, input_neuron in enumerate(self.input_layer):
                    # Layer output error
                    xk_error = (1.0 - X[k]) * X[k]
                    # Layer error
                    input_error = hidden_error * W_H[k]

                    # Generate gradient dw for input layer
                    dw_I = np.zeros(input_neuron.input_dimension)
                    for i_z, input_value in enumerate(x):
                        dw_I[i_z] = input_error * input_value
                    dBias_X = input_error * input_neuron.BIAS
                    input_neuron.update_weights(dw_I, dBias_X)

    def print_weights(self):
        for i, h_neuron in enumerate(self.hidden_layer):
            print(f"H{i}. {str(h_neuron.w)}, {str(h_neuron.b)}")

    def export_weights(self):
        h_weights = np.array([n.get_weights() for n in self.hidden_layer])
        o_weights = np.array([n.get_weights() for n in self.output_layer.neurons])
        i_weights = np.array([n.get_weights() for n in self.input_layer])
        np.savez(FFNN_WEIGHTS_FILE, h_weights, o_weights, i_weights)


def do_learn():
    global learner, train_x, train_y
    np.random.shuffle(train_x)
    for i, train_xi in enumerate(train_x):
        learner.train(train_xi, train_y[i])
    if verbose:
        learner.print_weights()
    learner.export_weights()


def do_evaluate():
    global learner, valid_x, valid_y

    tmse = 0
    accuracy_ctr = 0
    zero_ctr = 0
    one_ctr = 0
    for i, valid_xi in enumerate(train_x):
        y_hat = learner.predict(valid_xi)
        y_diff = abs(y_hat[0] - train_y[i][0])
        mse_i = y_diff**2
        tmse += mse_i
        # If p_zero >= p_one and target is zero, or p_zero < p_one and target is one
        if (y_hat[0] >= y_hat[1] and train_y[i][0] == 1) or (y_hat[0] < y_hat[1] and train_y[i][1] == 1):
            accuracy_ctr += 1
            if train_y[i][1] == 0:
                zero_ctr += 1
            else:
                one_ctr += 1
    tmse /= len(train_x)
    return tmse, accuracy_ctr, one_ctr, zero_ctr


if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    data_folder = "data\\titanic"
    processed_data_folder = os.path.join(cur_path, data_folder)
    mode = 'train' if 'train' in sys.argv else 'predict'
    # Note: Not using test.csv as it does not provide whether or not the passenger survived; therefore we cannot assess
    #       how well the model performed.
    data_file_path = os.path.join(processed_data_folder, "train.csv")
    data_processor = DataProcessor(data_file_path, processed_data_folder, "ffnn_processed.npz")

    # Load data
    try:
        #Try to load data
        data_processor.load_processed_data()

    except FileNotFoundError:
        #No data found, so process it
        data_processor.process_data((0.2, 0.2, 0.6), filter_nan=True)  # 10% test, 10% validation, 80% training samples from data

    # Extract training data, initialize neural network
    (train_x, train_y) = (data_processor.training_X, data_processor.training_y)
    train_y = np.array([[0, 1] if train_y[i] == 1 else [1, 0] for i in range(len(train_y))])
    # train_x = np.array([[i/1000,2*i/1000] for i in range(100)])
    # train_y = np.array([(train_x[i][0] + train_x[i][1])/1000 for i in range(100)])
    (valid_x, valid_y) = (data_processor.validation_X, data_processor.validation_y)
    valid_y = np.array([[0, 1] if valid_y[i] == 1 else [1, 0] for i in range(len(valid_y))])
    print('Loading neural network...')
    learner = FFNN(input_dimension=len(train_x[0]), output_dimension=len(valid_y[0]))

    if mode == 'train':
        mse_accum = [0] * NUM_EPOCHS
        for epoch in range(NUM_EPOCHS):
            if verbose:
                print(f"Beginning epoch {epoch+1}/{NUM_EPOCHS}")
            do_learn()
            mse_accum[epoch], accuracy, ones, zeros = do_evaluate()
            print(f"Epoch {epoch}/{NUM_EPOCHS}: {accuracy}/{len(train_x)} ({zeros}|{ones}); MSE={mse_accum[epoch]}")

        mse_x = np.arange(NUM_EPOCHS)
        mse_out = np.array(mse_accum) # np.array([[mse_x[i], mse_accum[i]] for i in range(NUM_EPOCHS)])
        fname = f'results_h{HIDDEN_LAYER_SIZE}_e{NUM_EPOCHS}.txt'
        np.savetxt(fname, mse_out)
