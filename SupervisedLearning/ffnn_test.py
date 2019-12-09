from data_processing import DataProcessor
import os
import numpy as np
import sys
import random as rand

# Defines sequence of hidden layer sizes to run with
HIDDEN_LAYER_SIZES = [10, 20, 50, 70, 100]
MAX_LEARNING_RATE = 0.05
LEARNING_RATE = MAX_LEARNING_RATE
MAX_MOMENTUM_FACTOR = 0.75
MOMENTUM_FACTOR = MAX_MOMENTUM_FACTOR
MAX_MOMENTUM = 5

NUM_EPOCHS = 250

verbose = False


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


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
        ret = sigmoid(tsum)
        self.output = ret
        return ret

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
                 hidden_layer_size, weights_file):
        self.input_dim = input_dimension
        self.hidden_layer_size = hidden_layer_size
        self.output_dimension = output_dimension
        file_weights = np.load(FFNN_WEIGHTS_FILE) if os.path.exists(weights_file) else None
        h1_weights = file_weights['arr_0'] if file_weights is not None else None
        h2_weights = file_weights['arr_1'] if file_weights is not None else None
        o_weights = file_weights['arr_2'] if file_weights is not None else None

        # Create output layer
        self.output_layer = Layer(input_dimension=hidden_layer_size, output_dimension=output_dimension,
                                  weights=o_weights, is_output=True, activation='sigmoid')

        self.hidden_layer2 = Layer(input_dimension=hidden_layer_size, output_dimension=hidden_layer_size,
                                   weights=h2_weights)

        self.hidden_layer1 = Layer(input_dimension=input_dimension, output_dimension=hidden_layer_size,
                                   weights=h1_weights)

        # Momentum gradient storage
        self.dw_o = np.zeros(shape=(self.output_layer.output_dimension, self.output_layer.input_dimension))
        self.dw_h2 = np.zeros(shape=(self.hidden_layer2.output_dimension, self.hidden_layer2.input_dimension))
        self.dw_h1 = np.zeros(shape=(self.hidden_layer1.output_dimension, self.hidden_layer1.input_dimension))
        self.momentum = 0

    def predict(self, x):
        if len(x) != self.input_dim:
            raise ValueError
        return self.output_layer.predict(self.hidden_layer2.predict(self.hidden_layer1.predict(x)))

    def train(self, x, target):
        if len(x) != self.input_dim:
            raise ValueError
        if isinstance(target, (str, list, tuple, np.ndarray)) and len(target) != self.output_dimension:
            raise ValueError

        # Output of first hidden layer
        h2_input_h1_output = [neuron.process_input(x) for neuron in self.hidden_layer1.neurons]
        # Output of second hidden layer
        o_input_h2_output = [neuron.process_input(h2_input_h1_output) for neuron in self.hidden_layer2.neurons]
        # Output of output layer
        o_output = self.output_layer.predict(o_input_h2_output)

        # Output Gradient
        err_output_total = [0] * len(self.output_layer.neurons)
        for o, o_neuron in enumerate(self.output_layer.neurons):
            output_err = -(target[o] - o_output[o])
            input_err = o_output[o] * (1 - o_output[o])
            err_output_total[o] = output_err * input_err

        # Update momentum if needed
        avg_error = sum(err_output_total) / len(err_output_total)
        self.update_momentum(avg_error)

        # Hidden2 Gradient
        err_hidden2_total = [0] * len(self.hidden_layer2.neurons)
        for h2, h_neuron in enumerate(self.hidden_layer2.neurons):
            dH = 0
            for o, o_neuron in enumerate(self.output_layer.neurons):
                dH += err_output_total[o] * o_neuron.w[h2]

            err_hidden2_total[h2] = dH * o_input_h2_output[h2] * (1 - o_input_h2_output[h2])

        # Hidden1 Gradient
        err_hidden1_total = [0] * len(self.hidden_layer1.neurons)
        for h2, h_neuron in enumerate(self.hidden_layer1.neurons):
            dH = 0
            for o, h2_neuron in enumerate(self.hidden_layer2.neurons):
                dH += err_hidden2_total[o] * h2_neuron.w[h2]

            err_hidden1_total[h2] = dH * h2_input_h1_output[h2] * (1 - h2_input_h1_output[h2])

        # Update output neurons
        for o, o_neuron in enumerate(self.output_layer.neurons):
            for w_ho in range(len(self.output_layer.neurons[o].w)):
                # Compute dw
                err_weight = err_output_total[o] * o_input_h2_output[w_ho]
                tdw = LEARNING_RATE * err_weight

                # Add momentum to dw
                dw = tdw
                if self.dw_o is not None:
                    dw += MOMENTUM_FACTOR*self.dw_o[o][w_ho]

                # Update momentum matrix
                self.dw_o[o][w_ho] += tdw

                o_neuron.w[w_ho] -= dw

        # Update hidden neurons
        for h2, h_neuron in enumerate(self.hidden_layer2.neurons):
            for w_ih in range(len(self.hidden_layer2.neurons[h2].w)):
                # Find dw
                err_weight = err_hidden2_total[h2] * h2_input_h1_output[w_ih]
                tdw = LEARNING_RATE * err_weight

                # Add momentum to dw
                dw = tdw
                if self.dw_h2 is not None:
                    dw += MOMENTUM_FACTOR*self.dw_h2[h2][w_ih]

                # Update momentum matrix
                self.dw_h2[h2][w_ih] += tdw

                h_neuron.w[w_ih] -= dw

        for h1, h_neuron in enumerate(self.hidden_layer1.neurons):
            for w_ih in range(len(self.hidden_layer1.neurons[h1].w)):
                err_weight = err_hidden1_total[h1] * x[w_ih]
                tdw = LEARNING_RATE * err_weight
                # Add momentum to dw
                dw = tdw
                if self.dw_h1 is not None:
                    dw += MOMENTUM_FACTOR*self.dw_h1[h1][w_ih]
                # Update momentum matrix
                self.dw_h1[h1][w_ih] += tdw

                h_neuron.w[w_ih] -= dw

        return

    def update_momentum(self, avg_error):
        # Trending positive
        if avg_error >= 0 and self.momentum >= 0:
            self.momentum += 1
        # Trending negative
        elif avg_error < 0 and self.momentum <= 0:
            self.momentum -= 1
        # Error is opposite of momentum
        else:
            # Too strong momentum -> reset momentum values
            if abs(self.momentum) > MAX_MOMENTUM:
                self.momentum = 0
                self.dw_h1 = np.zeros(shape=(self.hidden_layer1.output_dimension, self.hidden_layer1.input_dimension))
                self.dw_h2 = np.zeros(shape=(self.hidden_layer2.output_dimension, self.hidden_layer2.input_dimension))
                self.dw_o = np.zeros(shape=(self.output_layer.output_dimension, self.output_layer.input_dimension))
            # Reasonable momentum -> continue
            else:
                self.momentum += 1 if avg_error >= 0 else -1

    def export_weights(self):
        h1_weights = np.array([n.get_weights() for n in self.hidden_layer1.neurons])
        h2_weights = np.array([n.get_weights() for n in self.hidden_layer2.neurons])
        o_weights = np.array([n.get_weights() for n in self.output_layer.neurons])
        np.savez(FFNN_WEIGHTS_FILE, h1_weights, h2_weights, o_weights)


def do_learn(tindices):
    global learner, train_x, train_y
    for idx in tindices:
        learner.train(train_x[idx], train_y[idx])
    if verbose:
        learner.print_weights()
    learner.export_weights()


def do_evaluate():
    global learner, valid_x, valid_y

    err_accum = 0
    accuracy_ctr = 0
    zero_ctr = 0
    one_ctr = 0
    for i, valid_xi in enumerate(valid_x):
        y = valid_y[i]
        y_hat = learner.predict(valid_xi)
        # MSE
        err_i = 0
        for j in range(len(y_hat)):
            err_ij = 0.5 * (y[j] - y_hat[j])**2
            err_i += err_ij
        # binary cross-entropy
        # if valid_y[i][1] == 1:
        #     err_i = sum([-log(y_hat[j]) for j in range(len(y_hat))])
        # else:
        #     err_i = log_loss(valid_y[i], y_hat)
        err_accum += err_i
        # If p_zero >= p_one and target is zero,         or p_one > p_zero and target is one
        if (y_hat[0] >= y_hat[1] and valid_y[i][0] == 1) or (y_hat[1] > y_hat[0] and valid_y[i][1] == 1):
            accuracy_ctr += 1
            if valid_y[i][1] == 1:
                one_ctr += 1
            else:
                zero_ctr += 1
    err_accum /= len(valid_x)
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
        # Try to load data
        data_processor.load_processed_data()

    except FileNotFoundError:
        # No data found, so process it
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
        converters = {4: lambda sex: {'male': 0.0, 'female': 1.0}[sex],
                      11: lambda embarked: {'S': 0.0, 'C': 1.0, 'Q': 2.0}[embarked]}
        filter_missing = True
        data_processor.process_data(splits=splits, use_cols=use_cols, categorical_cols=categorical_cols,
                                    converters=converters, filter_missing=filter_missing)

    # Extract training data
    (train_x, train_y) = (data_processor.training_X, data_processor.training_y)
    train_y = np.array([[0, 1] if train_y[i] == 1 else [1, 0] for i in range(len(train_y))])
    (valid_x, valid_y) = (data_processor.validation_X, data_processor.validation_y)
    valid_y = np.array([[0, 1] if valid_y[i] == 1 else [1, 0] for i in range(len(valid_y))])

    # Easy data for performance comparison
    if 'easydata' in sys.argv:
        train_x = np.array([[i/1000, 2*i/1000] for i in range(100)])
        train_y = np.array([(train_x[i][0] + train_x[i][1])/1000 for i in range(100)])

    if mode == 'train':
        for HIDDEN_LAYER_SIZE in HIDDEN_LAYER_SIZES:
            print('Loading neural network...')
            FFNN_WEIGHTS_FILE = f'ffnn_weights_{HIDDEN_LAYER_SIZE}.npz'
            learner = FFNN(input_dimension=len(train_x[0]), output_dimension=len(train_y[0]),
                           hidden_layer_size=HIDDEN_LAYER_SIZE, weights_file=FFNN_WEIGHTS_FILE)
            err_accum = [0] * NUM_EPOCHS
            indices = [i for i in range(len(train_x))]
            for epoch in range(NUM_EPOCHS):
                # Shuffle indices
                rand.shuffle(indices)
                # Setup hyperparameters
                epoch_grad_coef = (NUM_EPOCHS - epoch)/NUM_EPOCHS
                MOMENTUM_FACTOR = MAX_MOMENTUM_FACTOR * epoch_grad_coef
                LEARNING_RATE = MAX_LEARNING_RATE * epoch_grad_coef
                if verbose:
                    print(f"Beginning epoch {epoch+1}/{NUM_EPOCHS}")
                do_learn(indices)
                err_accum[epoch], accuracy, ones, zeros = do_evaluate()
                print(f"H={HIDDEN_LAYER_SIZE}, Epoch {epoch}/{NUM_EPOCHS}: {accuracy}/{len(valid_x)} ({zeros}|{ones}); "
                      f"MSE={err_accum[epoch]}")
            # Deprecated: (epoch, mse) format
            # mse_x = np.arange(NUM_EPOCHS)
            # err_out = np.array([[mse_x[i], err_accum[i]] for i in range(NUM_EPOCHS)])
            err_out = err_accum
            fname = f'results\\results_h{HIDDEN_LAYER_SIZE}_e{NUM_EPOCHS}.txt'
            np.savetxt(fname, err_out)
