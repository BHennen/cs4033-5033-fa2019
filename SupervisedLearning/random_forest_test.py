from data_processing import DataProcessor
from TREES import DecisionTreeClassifier, RandomForestClassifier
import os
import numpy as np
from sklearn import ensemble, metrics, tree


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

    # X = np.array([[1,1],[1,2]])
    # y = np.array([1,2])

    # my_t = DecisionTreeClassifier()
    # my_t.fit(X,y)
    # my_t_p = my_t.predict(X)
    # my_t_pp = my_t.predict_proba(X)

    # sk_t = tree.DecisionTreeClassifier(max_features="auto")
    
    # sk_t.fit(X,y)
    # skt_t_p = sk_t.predict(X)
    # skt_t_pp = sk_t.predict_proba(X)

    X = np.concatenate((data.training_X, data.validation_X), axis=0)
    y = np.concatenate((data.training_y, data.validation_y), axis = 0)

    random_forest = RandomForestClassifier(n_estimators=100, min_node_size=1, max_features='auto', random_seed=None)
    random_forest.fit(X, y)

    my_f_predictions = random_forest.predict(X)
    my_f_proba = random_forest.predict_proba(X)
    my_f_xentropy = metrics.log_loss(y, my_f_proba)
    my_f_num_correct = np.sum(np.equal(my_f_predictions, y))
    my_f_percent_correct = my_f_num_correct / len(my_f_predictions)
    print(f"{my_f_num_correct} / {len(my_f_predictions)}; Train Accuracy:{my_f_percent_correct:.4f}, Xentropy:{my_f_xentropy}")
    
    skrf = ensemble.RandomForestClassifier(n_estimators=100, max_features='auto')
    skrf.fit(X, y)
    skrf_train_predictions = skrf.predict(X)
    skrf_train_proba = skrf.predict_proba(X)
    skrf_train_xentropy = metrics.log_loss(y, skrf_train_proba)
    skrf_train_num_correct = np.sum(np.equal(skrf_train_predictions, y))
    skrf_train_percent_correct = skrf_train_num_correct / len(skrf_train_predictions)
    print(f"{skrf_train_num_correct} / {len(skrf_train_predictions)}; SKRF Train Accuracy:{skrf_train_percent_correct:.4f}, Xentropy:{skrf_train_xentropy}")

    

    # predictions = random_forest.predict(X)
    # probas = random_forest.predict_proba(X)
    # print(predictions)
    # print(probas)
