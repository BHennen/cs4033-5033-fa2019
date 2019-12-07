from data_processing import DataProcessor
from TREES import DecisionTreeClassifier, RandomForestClassifier
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

    X = np.array([[1, 2],
                  [2, 7],
                  [2, 3]])
    y = np.array([1, 3, 2])

    random_forest = RandomForestClassifier(n_estimators=100, min_node_size=1, max_features='auto', random_seed=None)
    random_forest.fit(X,y)


    predictions = random_forest.predict(X)
    probas = random_forest.predict_proba(X)
    print(predictions)
    print(probas)
