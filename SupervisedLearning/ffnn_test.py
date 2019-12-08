from data_processing import DataProcessor
import os
import numpy as np
import sys
import random as rand

HIDDEN_LAYER_SIZE = 10
LEARNING_RATE = 0.5

NUM_EPOCHS = 1000

FFNN_WEIGHTS_FILE = f'ffnn_weights_{HIDDEN_LAYER_SIZE}.npz'

verbose = False


class Neuron:
    def __init__(self, input_dimension, weights=None, activation='sigmoid'):
        self.input_dimension = input_dimension
        self.activation = activation
        self.b = weights[-1] if weights is not None else None
        self.w = weights[:-1] if weights is not None else None
        self.output = -1

        # Init weights if needed
        if self.w is None:
            self.init_weights()

    def init_weights(self):
        self.w = np.random.rand(self.input_dimension)
        self.b = rand.random()

    def process_input(self, x):
        if len(x) != self.input_dimension:
            raise ValueError
        tsum = np.dot(self.w, x) + self.b
        ret = self.activate(tsum)
        self.output = ret
        return ret

    def activate(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def get_weights(self):
        t_weights = np.append(self.w, self.b)
        return t_weights

    # def get_error_total(self, target):
    #     return -(target - self.output) * self.output * (1 - self.output)


class Layer:
    def __init__(self, input_dimension, output_dimension, weights=None, activation='sigmoid', is_input=False,
                 is_output=False):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.is_input = is_input
        self.is_output = is_output
        self.activation = activation

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
        if not isinstance(y_target, (str, list, tuple, np.ndarray)):
            y_target = [y_target]
        if len(x) != self.input_dimension:
            raise ValueError
        if self.is_output and len(y_target) != self.output_dimension:
            raise ValueError

        Y = self.predict(x)

        if self.is_output:
            next_gradient = [0] * self.output_dimension
            loss = self.get_loss(y_target, Y)
            for i in range(self.output_dimension):
                next_gradient[i] = LEARNING_RATE * -loss
        else:
            next_gradient = self.next_layer.train(Y, y_target)

        this_gradient = np.zeros(self.input_dimension)
        for i_neuron, neuron in enumerate(self.neurons):
            # Layer output error - derivative of MSE
            y_gradient = (1.0 - Y[i_neuron]) * Y[i_neuron]
            error_i = next_gradient[i_neuron] * y_gradient if self.loss == 'absolute' else Y[i_neuron]

            # Store neuron weights before updating
            W = np.copy(neuron.w)

            # dW_i = e_i * x_i
            dW = error_i * np.array(x)
            dBias_Y = error_i * neuron.BIAS
            neuron.update_weights(dW, dBias_Y)

            # Update gradient
            for i_input in range(self.input_dimension):
                this_gradient[i_input] += error_i * W[i_input]
        return this_gradient

    def process_input(self, x):
        return self.predict(x)


# Predictors used: columns 2, 5, 6, 7, 9
class FFNN:
    def __init__(self, input_dimension, output_dimension,
                 hidden_layer_size=HIDDEN_LAYER_SIZE, learning_rate=LEARNING_RATE):
        self.input_dim = input_dimension
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.output_dimension = output_dimension
        file_weights = None
        # i_weights = None
        h_weights = None
        o_weights = None
        if os.path.exists(FFNN_WEIGHTS_FILE):
            print('Importing FFNN weights from existing file...')
            file_weights = np.load(FFNN_WEIGHTS_FILE)
            h_weights = file_weights['arr_0']
            o_weights = file_weights['arr_1']
            # i_weights = file_weights['arr_2']
        else:
            print("No FFNN weight file found, creating new FFNN.")



        # Create output layer
        self.output_layer = Layer(input_dimension=hidden_layer_size, output_dimension=output_dimension,
                                  weights=o_weights, is_output=True, activation='sigmoid')

        self.hidden_layer = Layer(input_dimension=input_dimension, output_dimension=hidden_layer_size,
                                  weights=h_weights)

    def predict(self, x):
        if len(x) != self.input_dim:
            raise ValueError
        return self.output_layer.predict(self.hidden_layer.predict(x))

    def train(self, x, target):
        if len(x) != self.input_dim:
            raise ValueError
        if isinstance(target, (str, list, tuple, np.ndarray)) and len(target) != self.output_dimension:
            raise ValueError

        # Output of hidden layer
        o_input_h_output = [neuron.process_input(x) for neuron in self.hidden_layer.neurons]
        # Output of output layer
        o_output = self.output_layer.predict(o_input_h_output)

        # Output Gradient
        err_output_total = [0] * len(self.output_layer.neurons)
        for o, o_neuron in enumerate(self.output_layer.neurons):
            output_err = -(target[o] - o_output[o])
            input_err = o_output[o] * (1 - o_output[o])
            err_output_total[o] = output_err * input_err
            # err_output_total[o] = o_neuron.get_error_total(target[o])

        # Hidden Gradient
        err_hidden_total = [0] * len(self.hidden_layer.neurons)
        for h, h_neuron in enumerate(self.hidden_layer.neurons):
            dH = 0
            for o, o_neuron in enumerate(self.output_layer.neurons):
                dH += err_output_total[o] * o_neuron.w[h]

            err_hidden_total[h] = dH * o_input_h_output[h] * (1 - o_input_h_output[h])

        # Update output neurons
        for o, o_neuron in enumerate(self.output_layer.neurons):
            for w_ho in range(len(self.output_layer.neurons[o].w)):
                err_weight = err_output_total[o] * o_input_h_output[w_ho]
                dw = LEARNING_RATE * err_weight
                o_neuron.w[w_ho] -= dw

        # Update hidden neurons
        for h, h_neuron in enumerate(self.hidden_layer.neurons):
            for w_ih in range(len(self.hidden_layer.neurons[h].w)):
                err_weight = err_hidden_total[h] * x[w_ih]
                dw = self.learning_rate * err_weight
                h_neuron.w[w_ih] -= dw

        return

    def print_weights(self):
        for i, h_neuron in enumerate(self.hidden_layer.neurons):
            print(f"H{i}. {str(h_neuron.w)}, {str(h_neuron.b)}")

    def export_weights(self):
        h_weights = np.array([n.get_weights() for n in self.hidden_layer.neurons])
        o_weights = np.array([n.get_weights() for n in self.output_layer.neurons])
        np.savez(FFNN_WEIGHTS_FILE, h_weights, o_weights)


def do_learn():
    global learner, train_x, train_y
    for i, train_xi in enumerate(train_x):
        learner.train(train_xi, train_y[i])
    if verbose:
        learner.print_weights()
    learner.export_weights()


def do_evaluate():
    global learner, valid_x, valid_y

    err_accum = 0
    accuracy_ctr = 0
    zero_ctr = 0
    one_ctr = 0
    for i, valid_xi in enumerate(train_x):
        y = train_y[i]
        y_hat = learner.predict(valid_xi)
        # MSE
        err_i = 0
        for j in range(len(y_hat)):
            err_ij = 0.5 * (y[j] - y_hat[j])**2
            err_i += err_ij
        # binary cross-entropy
        # if train_y[i][1] == 1:
        #     err_i = sum([-log(y_hat[j]) for j in range(len(y_hat))])
        # else:
        #     err_i = log_loss(train_y[i], y_hat)
        err_accum += err_i
        # If p_zero >= p_one and target is zero,         or p_one > p_zero and target is one
        if (y_hat[0] >= y_hat[1] and train_y[i][0] == 1) or (y_hat[1] > y_hat[0] and train_y[i][1] == 1):
            accuracy_ctr += 1
            if train_y[i][1] == 1:
                one_ctr += 1
            else:
                zero_ctr += 1
    err_accum /= len(train_x)
    return err_accum, accuracy_ctr, one_ctr, zero_ctr


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
         # 20% test, 20% validation, 60% training samples from data
        splits = (0.2, 0.2, 0.6)
        # Only use certain columns
        use_cols = (  # 0, #PassengerID
                    1,  # Survived
                    2,  # Pclass
                    # 3, #Name
                    4,  # Sex
                    5,  # Age
                    6,  # SibSp
                    7,  # Parch
                    # 8, #Ticket
                    9,  # Fare
                    # 10, #Cabin
                    # 11,  # Embarked
        )
        # Mark features as categorical (so we can one-hot-encode them later)
        # categorical_cols = ()
        categorical_cols = (2,  # Pclass
                            4,  # Sex
                            11  # Embarked
        )
        # Convert certain columns to float values (so we can use numpy arrays)
        converters = {4: lambda sex: {'male':0.0, 'female':1.0}[sex],
                      11: lambda embarked: {'S': 0.0, 'C': 1.0, 'Q': 2.0}[embarked]}
        filter_missing = True
        data_processor.process_data(splits=splits, use_cols=use_cols, categorical_cols=categorical_cols,
                                    converters=converters, filter_missing=filter_missing)

    # Extract training data, initialize neural network
    (train_x, train_y) = (data_processor.training_X, data_processor.training_y)
    train_y = np.array([[0, 1] if train_y[i] == 1 else [1, 0] for i in range(len(train_y))])
    # train_x = np.array([[i / 1000, 2 * i / 1000] for i in range(100)])
    # train_y = np.array([[1, 0] if sum(train_x[i]) > 100 / 1000 else [0, 1] for i in range(100)])
    (valid_x, valid_y) = (data_processor.validation_X, data_processor.validation_y)
    valid_y = np.array([[0, 1] if valid_y[i] == 1 else [1, 0] for i in range(len(valid_y))])
    print('Loading neural network...')
    learner = FFNN(input_dimension=len(train_x[0]), output_dimension=len(train_y[0]))

    if mode == 'train':
        err_accum = [0] * NUM_EPOCHS
        for epoch in range(NUM_EPOCHS):
            if verbose:
                print(f"Beginning epoch {epoch+1}/{NUM_EPOCHS}")
            do_learn()
            err_accum[epoch], accuracy, ones, zeros = do_evaluate()
            print(f"Epoch {epoch}/{NUM_EPOCHS}: {accuracy}/{len(train_x)} ({zeros}|{ones}); MSE={err_accum[epoch]}")

        err_out = np.array(err_accum)
        fname = f'results_h{HIDDEN_LAYER_SIZE}_e{NUM_EPOCHS}.txt'
        np.savetxt(fname, err_out)
