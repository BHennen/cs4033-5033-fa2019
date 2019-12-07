from data_processing import DataProcessor
import os
import numpy as np
import sys

HIDDEN_LAYER_SIZE = 50
LEARNING_RATE = 0.001
LINEAR_LAMBDA = 0.8

NUM_EPOCHS = 1000

FFNN_WEIGHTS_FILE = 'ffnn_weights.npz'

verbose = False


class Neuron:
    def __init__(self, input_dimension, weights=None, activation='sigmoid'):
        self.input_dimension = input_dimension
        self.activation = activation
        self.b = 0.001
        self.w = weights
        self.BIAS = -1

        # Init weights if needed
        if self.w is None:
            self.init_weights()

    def init_weights(self):
        self.w = np.random.rand(self.input_dimension)
        for i, wi in enumerate(self.w):
            self.w[i] /= 5.0

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
            if np.isnan(dWeights[i]):
                raise ValueError
            self.w[i] += dWeights[i]

        self.b += dBias


# Predictors used: columns 2, 5, 6, 7, 9
class FFNN:
    def __init__(self, input_dimension, hidden_layer_size=HIDDEN_LAYER_SIZE, learning_rate=LEARNING_RATE):
        self.input_dim = input_dimension
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        file_weights = None
        h_weights = None
        o_weights = None
        if os.path.exists(FFNN_WEIGHTS_FILE):
            print('Importing FFNN weights from existing file...')
            file_weights = np.load(FFNN_WEIGHTS_FILE)
            h_weights = file_weights['arr_0']
            o_weights = file_weights['arr_1']
        else:
            print("No FFNN weight file found, creating new FFNN.")

        self.output_neuron = Neuron(input_dimension=hidden_layer_size, activation='sigmoid', weights=o_weights)
        self.hidden_layer = [None] * hidden_layer_size
        for i in range(hidden_layer_size):
            t_weights = None
            if file_weights is not None:
                t_weights = h_weights[i]
            self.hidden_layer[i] = Neuron(input_dimension=self.input_dim, activation='linear', weights=t_weights)

    def predict(self, x):
        if len(x) != self.input_dim:
            raise ValueError
        Y = [neuron.process_input(x) for neuron in self.hidden_layer]
        return self.output_neuron.process_input(Y)

    def train(self, x, target):
        if len(x) != self.input_dim:
            raise ValueError
        # Get output of FFNN
        Y = [neuron.process_input(x) for neuron in self.hidden_layer]
        o_k = self.output_neuron.process_input(Y)

        # Find error
        output_error = target - o_k

        # Store copy of output neuron weights before updating
        W = np.copy(self.output_neuron.w)

        # Update output neuron's weights
        dWs = np.zeros(self.hidden_layer_size)
        global_error = self.learning_rate * output_error * (1.0 - o_k) * o_k
        for j, hidden_output in enumerate(Y):
            dWs[j] = global_error * Y[j]
        dBias = global_error*-1.0
        self.output_neuron.update_weights(dWs, dBias)

        # Update hidden layer weights
        for j, h_neuron in enumerate(self.hidden_layer):
            yj_error = (1.0 - Y[j]) * Y[j]
            dVs = np.zeros(self.input_dim)
            y_error = global_error * W[j] * yj_error
            for i in range(self.input_dim):
                dVs[i] = y_error * x[i]
            dBias = y_error * -1.0
            h_neuron.update_weights(dVs, dBias)

    def print_weights(self):
        for i, h_neuron in enumerate(self.hidden_layer):
            print(f"H{i}. {str(h_neuron.w)}")

    def export_weights(self):
        h_weights = np.array([n.w for n in self.hidden_layer])
        o_weights = self.output_neuron.w
        np.savez(FFNN_WEIGHTS_FILE, h_weights, o_weights)


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
        y_diff = abs(y_hat - train_y[i])
        mse_i = y_diff**2
        tmse += mse_i
        if (y_hat >= 0.5 and train_y[i] == 1) or (y_hat < 0.5 and train_y[i] == 0):
            accuracy_ctr += 1
            if train_y[i] == 0:
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
        data_processor.process_data((0.2, 0.2, 0.6))  # 10% test, 10% validation, 80% training samples from data

    # Extract training data, initialize neural network
    (train_x, train_y) = (data_processor.training_X, data_processor.training_y)
    # train_x = np.array([[i/1000,2*i/1000] for i in range(100)])
    # train_y = np.array([(train_x[i][0] + train_x[i][1])/1000 for i in range(100)])
    (valid_x, valid_y) = (data_processor.validation_X, data_processor.validation_y)
    print('Loading neural network...')
    learner = FFNN(input_dimension=len(train_x[0]))

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
