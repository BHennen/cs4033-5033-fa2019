from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense
import os
import numpy as np
import sys
from data_processing import DataProcessor

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
    data_processor.process_data((0.2, 0.2, 0.6), filter_missing=True)  # 10% test, 10% validation, 80% training samples from data

# Extract training data, initialize neural network
(train_x, train_y) = (data_processor.training_X, data_processor.training_y)
train_y = np.array([[0, 1] if train_y[i] == 1 else [1, 0] for i in range(len(train_y))])
# train_x = np.array([[i/1000,2*i/1000] for i in range(100)])
# train_y = np.array([(train_x[i][0] + train_x[i][1])/1000 for i in range(100)])
(valid_x, valid_y) = (data_processor.validation_X, data_processor.validation_y)
valid_y = np.array([[0, 1] if valid_y[i] == 1 else [1, 0] for i in range(len(valid_y))])
print('Loading neural network...')
train_x = np.array([[i/1000, 2*i/1000] for i in range(100)])
train_y = np.array([[1, 0] if sum(train_x[i]) > 100/1000 else [0, 1] for i in range(100)])

# Set NN params
NUM_EPOCHS = 1000
hidden_layer_size = 10
input_dimension = len(train_x[0])
output_dimension = len(train_y[0])

# init NN
model = Sequential()
model.add(Dense(hidden_layer_size, input_dim=input_dimension, activation='relu'))
model.add(Dense(output_dimension, activation='sigmoid'))

model.compile(loss='mean_squared_error', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=NUM_EPOCHS)

_, accuracy = model.evaluate(train_x, train_y)
print('Accuracy: %.2f' % (accuracy*100))