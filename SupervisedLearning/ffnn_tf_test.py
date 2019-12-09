from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense
import os
import numpy as np
import sys
from data_processing import DataProcessor
from matplotlib import pyplot

cur_path = os.path.dirname(__file__)
data_folder = "data\\titanic"
processed_data_folder = os.path.join(cur_path, data_folder)
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
if 'train' in sys.argv:
    # Extract training data, initialize neural network
    (train_x, train_y) = (data_processor.training_X, data_processor.training_y)
    train_y = np.array([[0, 1] if train_y[i] == 1 else [1, 0] for i in range(len(train_y))])
    (valid_x, valid_y) = (data_processor.validation_X, data_processor.validation_y)
    valid_y = np.array([[0, 1] if valid_y[i] == 1 else [1, 0] for i in range(len(valid_y))])
    if 'easydata' in sys.argv:
        train_x = np.array([[i/1000, 2*i/1000] for i in range(100)])
        train_y = np.array([(train_x[i][0] + train_x[i][1])/1000 for i in range(100)])
        valid_x, valid_y = train_x, train_y  # Note: validation is on training set for easy data
    print('Loading neural network...')

    # Set NN params
    input_dimension = len(train_x[0])
    output_dimension = len(train_y[0]) if isinstance(train_y[0], (str, list, tuple, np.ndarray)) else 1
    NUM_EPOCHS = 250
    hidden_layer_sizes = [10, 20, 50, 70, 100]
    handles = [None] * len(hidden_layer_sizes)
    fig = pyplot.figure("CS 4033/5033 - Hennen/Bost")
    ax = fig.add_subplot(111)
    ax.set_ylabel('mse')
    ax.set_xlabel('epoch')
    ax.set_title('Keras model (easy data, 1-hidden)')

    for idx_size, hidden_layer_size in enumerate(hidden_layer_sizes):

        # init NN
        model = Sequential()
        model.add(Dense(hidden_layer_size, input_dim=input_dimension, activation='sigmoid'))
        model.add(Dense(output_dimension, activation='sigmoid'))

        model.compile(loss='mean_squared_error', metrics=['accuracy', 'mse'],
                      optimizer=Adam(learning_rate=0.001))

        history = model.fit(train_x, train_y, epochs=NUM_EPOCHS, validation_split=0,
                            validation_data=(valid_x, valid_y), verbose=2)

        mse_data = history.history['val_mse']
        handles[idx_size] = pyplot.plot(mse_data, label=f"H={hidden_layer_size}")

        _, accuracy, mse = model.evaluate(train_x, train_y)

    pyplot.legend()
    pyplot.show()
