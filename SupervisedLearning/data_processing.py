import os
import numpy as np

class DataProcessor():
    def __init__(self, data_file_path, processed_data_folder, processed_data_filename="processed_data.npz"):
        self._data_file_path = data_file_path
        self._processed_data_folder = processed_data_folder
        self._processed_data_filename = processed_data_filename
        self._processed_data_path = os.path.join(self._processed_data_folder, self._processed_data_filename)

    def check_if_processed(self):
        # Checks for already processed data
        if os.path.exists(self.processed_data_path):
            return True
        else:
            return False
    
    def load_processed_data(self):
        #Loads already processed data
        if self.check_if_processed():
            with np.load(self._processed_data_path) as data:
                self.test_X = data['test_X']
                self.test_y = data['test_y']
                self.training_X = data['training_X']
                self.training_y = data['training_y']
                self.validation_X = data['validation_X']
                self.validation_y = data['validation_y']
        else:
            raise FileNotFoundError(f"Data not processed yet; no file found at:{self.processed_data_path}")
                
    def process_data(self, splits):
        '''
        splits: 3-tuple of floats which adds to 1, which is the proportion of data that will be allocated to
            the testing set, validation set, and training set, in that order. (0.1, 0.1, 0.8) means 10% allocated
            each to training and validation, and 80% to training.
        '''
        if sum(splits) != 1:
            raise ArithmeticError("process_data: splits does not add to 1")

        #Process data for the first time
        if os.path.exists(self.data_file_path):
            #Process data
            ##TODO: update function to match format of data file
            data = np.genfromtxt(self._data_file_path)
            #TODO: We might have to make our own predictor columns

            #Shuffle data
            np.random.shuffle(data)

            #Split into test and validation combined with training data
            len_test_data = int(len(data) * splits[0])
            test_data = data[0:len_test_data]
            validation_train_data = data[len_test_data:]
            ## Split remaing into validation and training
            len_validation_data = int(len(data) * splits[1])
            validation_data = validation_train_data[0:len_validation_data]
            training_data = validation_train_data[len_validation_data:]

            # TODO: Remove target columns from dataset and save them for later
            # *_X = only predictors for the data
            # *_y = only target columns for the data
            test_X = np.delete(test_data, 1, 1)  # delete target column
            test_y = test_data[:, 1]  # Only use target column
            training_X = np.delete(training_data, 1, 1)  # delete target column
            training_y = training_data[:, 1]  # Only use target column
            validation_X = np.delete(validation_data, 1, 1)  # delete target column
            validation_y = validation_data[:, 1]  # Only use target column
            
            # Save *_X and *_y data
            self.test_X = test_X
            self.test_y = test_y
            self.training_X = training_X
            self.training_y = training_y
            self.validation_X = validation_X
            self.validation_y = validation_y

            np.savez(self._data_file_path, test_X = test_X, test_y = test_y, training_X = training_X, 
                     training_y = training_y, validation_X = validation_X, validation_y = validation_y)

        else:
            raise FileNotFoundError(f"No data at path specified:{self.data_file_path}")
       

