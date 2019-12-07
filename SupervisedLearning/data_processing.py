import os
import numpy as np
from csv import reader


class DataProcessor():
    def __init__(self, data_file_path, processed_data_folder, processed_data_filename="processed_data.npz"):
        self._data_file_path = data_file_path
        self._processed_data_folder = processed_data_folder
        self._processed_data_filename = processed_data_filename
        self._processed_data_path = os.path.join(self._processed_data_folder, self._processed_data_filename)

    def check_if_processed(self):
        # Checks for already processed data
        if os.path.exists(self._processed_data_path):
            return True
        else:
            return False

    def load_processed_data(self):
        # Loads already processed data
        if self.check_if_processed():
            print(f"Loading preprocessed data, if you want to process again, delete: {self._processed_data_path}")
            with np.load(self._processed_data_path) as data:
                self.test_X = data['test_X']
                self.test_y = data['test_y']
                self.training_X = data['training_X']
                self.training_y = data['training_y']
                self.validation_X = data['validation_X']
                self.validation_y = data['validation_y']
        else:
            raise FileNotFoundError(f"Data not processed yet; no file found at: {self._processed_data_path}")

    def process_data(self, splits, filter_nan=False):
        '''
        splits: 3-tuple of floats which adds to 1, which is the proportion of data that will be allocated to
            the testing set, validation set, and training set, in that order. (0.1, 0.1, 0.8) means 10% allocated
            each to training and validation, and 80% to training.

        filter_nan: Boolean indicating if we encounter no value for a column, should we skip that row.
        '''
        if sum(splits) != 1:
            raise ArithmeticError("process_data: splits does not add to 1")

        # Process data for the first time
        if os.path.exists(self._data_file_path):
            # Process data
            print(f"Processing data: {self._data_file_path}")
            
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
                        11,  # Embarked
            )
            # Mark features as categorical (so we can one-hot-encode them later)
            categorical_cols = (2,  # Pclass
                                4,  # Sex
                                11  # Embarked
            )
            # Convert certain columns to float values (so we can use numpy arrays)
            converters = {4: lambda sex: {'male':0.0, 'female':1.0}[sex],
                          11: lambda embarked: {'S': 0.0, 'C': 1.0, 'Q': 2.0}[embarked]}
            # convert column index to actual index in np array (since we may have skipped another col)
            col_mapping = {col: index for index, col in enumerate(sorted(use_cols))}
            data = []
            with open(self._data_file_path) as data_file:
                for line_no, line in enumerate(reader(data_file)):
                    if line_no < 1:
                        continue
                    # Iterate through columns, skipping those we dont want, and converting others
                    skip_line = False
                    cols = []
                    for index, col in enumerate(line):
                        if index not in use_cols:
                            continue
                        if col == '':
                            # Default value for no data
                            if filter_nan:
                                skip_line = True
                                break
                            cols.append(None)
                        elif index in converters:
                            # Try checking for converter
                            cols.append(converters[index](col))
                        else:
                            #Try converting to float
                            try:
                                cols.append(float(col))
                            except ValueError as e:
                                print("Default conversion to float did not work.")
                                raise e
                    if skip_line:
                        continue
                    data.append(cols)
            
            data = np.array(data, dtype=np.float_)

            # Convert categorical columns into n new columns, where n is the number of possible classes in that
            # category. Ex: A column called color with possible values of red, blue, green would be converted
            # into 3 columns: red, blue and green, each row having value of 1 or 0 for each class
            #TODO: Does not handle nan correctly (each nan counts as separate category)
            categorical_idx = [col_mapping[col] for col in categorical_cols]
            offset = 0
            for col_idx in categorical_idx:
                col_idx = col_idx + offset  # Adjust current column index if we have already added new columns to data
                # Determine how many possibilities there are for this category
                categories, indices = np.unique(data[:, col_idx], return_inverse=True)
                num_new_cols = len(categories)
                offset += num_new_cols - 1 # -1 since we delete the old column
                one_hot_matrix = np.zeros((len(data), num_new_cols))
                # loop through indices and set the specified column to 1
                for row, col in enumerate(indices):
                    one_hot_matrix[row][col] = 1
                # Add one_hot_matrix to data
                data = np.insert(data, col_idx + 1, one_hot_matrix.T, axis=1)
                # Delete old col
                data = np.delete(data, col_idx, axis=1)

            # Shuffle data
            np.random.shuffle(data)

            # Split into test and validation combined with training data
            len_test_data = int(len(data) * splits[0])
            test_data = data[0:len_test_data]
            validation_train_data = data[len_test_data:]
            validation_train_X = np.delete(validation_train_data, 0, 1) #Delete target column
            # Split remaing into validation and training
            len_validation_data = int(len(data) * splits[1])
            validation_data = validation_train_data[0:len_validation_data]
            training_data = validation_train_data[len_validation_data:]

            # Remove target columns from dataset and save them for later
            # *_X = only predictors for the data
            # *_y = only target columns for the data
            test_X = np.delete(test_data, 0, 1)  # delete target column
            test_y = test_data[:, 0]  # Only use target column
            training_X = np.delete(training_data, 0, 1)  # delete target column
            training_y = training_data[:, 0]  # Only use target column
            validation_X = np.delete(validation_data, 0, 1)  # delete target column
            validation_y = validation_data[:, 0]  # Only use target column

            ## Normalize predictor data
            #Get mean and standard deviation of each column (except the last)
            train_valid_mean = np.nanmean(validation_train_X, axis=0)
            train_valid_sd = np.nanstd(validation_train_X, axis=0)
            #calculate z-score for each datum
            for row_index, row in enumerate(test_X):
                for col_index, val in enumerate(row):
                    normalized_val = (val - train_valid_mean[col_index]) / train_valid_sd[col_index]
                    test_X[row_index, col_index] = normalized_val
            for row_index, row in enumerate(training_X):
                for col_index, val in enumerate(row):
                    normalized_val = (val - train_valid_mean[col_index]) / train_valid_sd[col_index]
                    training_X[row_index, col_index] = normalized_val
            for row_index, row in enumerate(validation_X):
                for col_index, val in enumerate(row):
                    normalized_val = (val - train_valid_mean[col_index]) / train_valid_sd[col_index]
                    validation_X[row_index, col_index] = normalized_val

            # Save *_X and *_y data
            self.test_X = test_X
            self.test_y = test_y
            self.training_X = training_X
            self.training_y = training_y
            self.validation_X = validation_X
            self.validation_y = validation_y

            print(f"Saving processed data to: {self._processed_data_path}")
            np.savez(self._processed_data_path, test_X=test_X, test_y=test_y, training_X=training_X,
                     training_y=training_y, validation_X=validation_X, validation_y=validation_y)

        else:
            raise FileNotFoundError(f"No data at path specified:{self._data_file_path}")
