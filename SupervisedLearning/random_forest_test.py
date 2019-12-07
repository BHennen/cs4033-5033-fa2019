from data_processing import DataProcessor
from TREES import DecisionTreeClassifier, RandomForestClassifier
import os
import numpy as np
from sklearn import ensemble, metrics


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
        data.process_data((0.1, 0.1, 0.8), filter_missing=True)  # 10% test, 10% validation, 80% training samples from data

    

    random_forest = RandomForestClassifier(n_estimators=100, min_node_size=1, max_features='auto', random_seed=None)
    random_forest.fit(data.training_X, data.training_y)

    train_predictions = random_forest.predict(data.training_X)
    train_proba = random_forest.predict_proba(data.training_X)
    train_xentropy = metrics.log_loss(data.training_y, train_proba)
    train_num_correct = np.sum(np.equal(train_predictions, data.training_y))
    train_percent_correct = train_num_correct / len(train_predictions)
    print(f"{train_num_correct} / {len(train_predictions)}; Train Accuracy:{train_percent_correct:.4f}, Xentropy:{train_xentropy}")
    validation_predictions = random_forest.predict(data.validation_X)
    validation_num_correct = np.sum(np.equal(validation_predictions, data.validation_y))
    validation_percent_correct = validation_num_correct / len(validation_predictions)

    print(f"{validation_num_correct} / {len(validation_predictions)}; Validation Accuracy:{validation_percent_correct}")
    
    skrf = ensemble.RandomForestClassifier(n_estimators=100)
    skrf.fit(data.training_X, data.training_y)
    skrf_train_predictions = skrf.predict(data.training_X)
    skrf_train_proba = skrf.predict_proba(data.training_X)
    skrf_train_xentropy = metrics.log_loss(data.training_y, skrf_train_proba)
    skrf_train_num_correct = np.sum(np.equal(skrf_train_predictions, data.training_y))
    skrf_train_percent_correct = skrf_train_num_correct / len(skrf_train_predictions)
    print(f"{skrf_train_num_correct} / {len(skrf_train_predictions)}; SKRF Train Accuracy:{skrf_train_percent_correct:.4f}, Xentropy:{skrf_train_xentropy}")

    

    # predictions = random_forest.predict(X)
    # probas = random_forest.predict_proba(X)
    # print(predictions)
    # print(probas)
