import os
import numpy as np
from csv import reader
import matplotlib.pyplot as plt


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

    def process_data(self, splits, use_cols, categorical_cols, converters, filter_missing):
        '''
        Parameters
        ----------
        splits: tuple
            Tuple of floats which adds to 1, which is the proportion of data that will be allocated to
            the testing set, validation set, and training set, in that order. (0.1, 0.1, 0.8) means 10% allocated
            each to training and validation, and 80% to training.

        use_cols : tuple
            Specifies which columns to use from the raw data.

        categorical_cols : tuple
            From the raw data, specifies which columns are to be treated as categorical.

        converters : dict
            Mapping from raw column to a function which returns the processed value for that datum.
            Ex: {4: lambda x: x + 1} # For column 4, take the raw value and add 1 to it

        filter_missing: Boolean 
            Indicates if we encounter no value for a column, should we skip that row.
        '''
        if sum(splits) != 1:
            raise ArithmeticError("process_data: splits does not add to 1")

        # Process data for the first time
        if os.path.exists(self._data_file_path):
            # Process data
            print(f"Processing data: {self._data_file_path}")
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
                            cols.append(None)
                            if filter_missing:
                                skip_line = True
                                break
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
            #TODO: Does not handle nan correctly (each nan counts as separate category, not a big deal if filtering missing)
            # convert column index to actual index in np array (since we may have skipped another col)
            col_mapping = {col: index for index, col in enumerate(sorted(use_cols))}
            categorical_idx = [col_mapping[col] for col in categorical_cols if col in col_mapping.keys()]
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
            self.test_X = np.delete(test_data, 0, 1)  # delete target column
            self.test_y = test_data[:, 0]  # Only use target column
            self.training_X = np.delete(training_data, 0, 1)  # delete target column
            self.training_y = training_data[:, 0]  # Only use target column
            self.validation_X = np.delete(validation_data, 0, 1)  # delete target column
            self.validation_y = validation_data[:, 0]  # Only use target column

            ## Normalize predictor data
            #Get mean and standard deviation of each column (except the last)
            train_valid_mean = np.nanmean(validation_train_X, axis=0)
            train_valid_sd = np.nanstd(validation_train_X, axis=0)
            #calculate z-score for each datum
            for row_index, row in enumerate(self.test_X):
                for col_index, val in enumerate(row):
                    normalized_val = (val - train_valid_mean[col_index]) / train_valid_sd[col_index]
                    self.test_X[row_index, col_index] = normalized_val
            for row_index, row in enumerate(self.training_X):
                for col_index, val in enumerate(row):
                    normalized_val = (val - train_valid_mean[col_index]) / train_valid_sd[col_index]
                    self.training_X[row_index, col_index] = normalized_val
            for row_index, row in enumerate(self.validation_X):
                for col_index, val in enumerate(row):
                    normalized_val = (val - train_valid_mean[col_index]) / train_valid_sd[col_index]
                    self.validation_X[row_index, col_index] = normalized_val
            np.savez(self._processed_data_path, test_X=self.test_X, test_y=self.test_y, training_X=self.training_X,
                     training_y=self.training_y, validation_X=self.validation_X, validation_y=self.validation_y)

        else:
            raise FileNotFoundError(f"No data at path specified:{self._data_file_path}")

class Metrics():
    roc_thresholds = np.arange(0.000, 1.001, 0.001)

    @staticmethod
    def get_ROC_data(calc_proba, true_proba):
        '''
        Given the calculated probabilities and true probabilities, find the Probability of Detection (POD) and 
        Probability of False Detection (POFD) values for each threshold in Metrics.roc_thresholds. Can be used
        to plot the ROC curve or find the AUC for ROC plot.

        This function only works for probabilities for one class (Ex: Is it green or not?).

        Parameters
        ----------
        calc_proba : 1d numpy array of floats from [0.0-1.0]
            Calculated probabilities where each row in calc_proba is the probability that the datum is part of a class.

        true_proba : 1d numpy array of floats from [0.0-1.0]
            True probabilities where each row in true_proba is the true probability that the datum is part of a class.    
        
        Returns
        -------
        (POFD_arr, POD_arr) : tuple of arrays
            POFD and POD are arrays where each element corresponds to one threshold in the determinization thresholds.
            Returned in order sorted by POFD_arr
        
        '''

        if len(calc_proba) != len(true_proba):
            raise IndexError("Unequal probability list length.")

        dtrmzd_proba = [] # Determinized (ie 1 or 0) probability for the calculated probability

        # Determinize probabilities
        for threshold in Metrics.roc_thresholds:
            def determinizer(p): return 1 if p >= threshold else 0
            vdeterminizer = np.vectorize(determinizer)
            dtrmzd_proba.append(vdeterminizer(calc_proba))

        # Compute probability of detection and probability of false detection
        POD_arr = []
        POFD_arr = []

        for threshold_idx, threshold in enumerate(Metrics.roc_thresholds):
            # a - Number of: Event observed, forecasted
            # c - Number of: Event observed, not forecasted
            # b - Number of: Event not observed, forecasted
            # d - Number of: Event not observed, not forecasted
            # Calculate a b c d
            abcd = {"a": 0, "b": 0, "c": 0, "d": 0}
            calc_proba = dtrmzd_proba[threshold_idx]
            for obs_idx, observation in enumerate(true_proba):
                if observation == 1:  # Event observed
                    if calc_proba[obs_idx] == 1:  # forecasted
                        abcd["a"] += 1
                    else:  # not forecasted
                        abcd["c"] += 1
                else:  # Event not observed
                    if calc_proba[obs_idx] == 1:  # forecasted
                        abcd["b"] += 1
                    else:  # not forecasted
                        abcd["d"] += 1

            #Calculate POD = a/(a+c) and POFD = b/(b+d)
            POD = abcd["a"] / (abcd["a"] + abcd["c"])
            POFD = abcd["b"] / (abcd["b"] + abcd["d"])
            POD_arr.append(POD)
            POFD_arr.append(POFD)
        
        #Make sure we have (0,0) and (1,1)
        POD_arr.append(0)
        POFD_arr.append(0)
        POD_arr.append(1)
        POFD_arr.append(1)

        # # Sort by POFD_arr in ascending order
        # data = list(zip(POFD_arr, POD_arr))
        # data.sort()  
        return POFD_arr, POD_arr
    
    @staticmethod
    def add_ROC_curve(POFD_arr, POD_arr, label, color, include_AUC=True):
        ''' Plot ROC curve
        
            Does not display curve until show_roc_curve is called.

            Parameters
            ----------
            label : String

            color : Color

            include_AUC : Boolean
                Whether or not to include the area under the curve as part of the label.

        '''

        # Sort by POFD values in ascending order
        POFD_arr, POD_arr = zip(*sorted(zip(POFD_arr, POD_arr)))

        # Calculate area under curve
        AUC = None
        if include_AUC:
            AUC = Metrics.get_AUC(POFD_arr, POD_arr, is_sorted=True)
            
        #grey line
        x = np.linspace(0, 1, 100)
        y = x
        greycolor = (.6, .6, .6)
        linestyle = "--"
        plt.plot(x, y, color=greycolor, linestyle=linestyle)

        # Labels
        plt.xlabel('Probability of False Detection (POFD)')
        plt.ylabel('Probability of Detection (POD)')        
        plt.plot(POFD_arr, POD_arr, color=color, label=label + f" AUC: {AUC:.4f}")

        return AUC
        

    @staticmethod
    def show_ROC_curve(title):
        plt.legend()
        plt.title(title + " ROC Curve")
        plt.show()

    @staticmethod
    def get_AUC(x_vals, y_vals, is_sorted=False):
        '''Get area under the curve of a plot. Uses integration approximation using midpoint rule.

        Parameters
        ----------
        x_vals : Array
            Values for the x-axis. For ROC curve, this would be POFD_arr.

        y_vals : Array 
            Values for the y-axis, corresponding with the x_vals. For ROC curve, this would be POD_arr.
        
        is_sorted: Boolean
            Whether or not the data is sorted already by the x_vals.

        '''

        if len(x_vals) != len(y_vals):
            raise IndexError("Unequal list length.")

        data = list(zip(x_vals, y_vals))
        if not is_sorted: data.sort()  # Sort by x_vals ascending order
        dx = 0
        prev_y = 0
        prev_x = data[0][0]
        tot_area = 0
        
        for idx, (x_val, y_val) in enumerate(data):
            #Calculate using trapezoidal rule
            y_avg = (y_val + prev_y) / 2
            dx = x_val - prev_x
            area = y_avg * dx
            tot_area += area
            prev_x = x_val
            prev_y = y_val

        return tot_area
