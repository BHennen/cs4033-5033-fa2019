from data_processing import DataProcessor
from TREES import DecisionTreeClassifier
import os
import numpy as np

if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    data_folder = "data\\titanic"
    processed_data_folder = os.path.join(cur_path, data_folder)
    # Note: Not using test.csv as it does not provide whether or not the passenger survived; therefore we cannot assess
    #       how well the model performed.
    data_file_path = os.path.join(processed_data_folder, "train.csv") 
    data = DataProcessor(data_file_path, processed_data_folder)

    try:
        #Try to load data
        data.load_processed_data()
    except FileNotFoundError:
        #No data found, so process it
        data.process_data((0.1, 0.1, 0.8))  # 10% test, 10% validation, 80% training samples from data

    decision_tree = DecisionTreeClassifier()
    X = np.array([[1, 2],
                  [2, 7],
                  [4, 3]])
    y = np.array([1, 2, 3])
    decision_tree.fit(X, y)

    decision_tree