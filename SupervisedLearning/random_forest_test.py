from data_processing import DataProcessor, Metrics
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
        # 10% test, 10% validation, 80% training samples from data
        splits = (0.1, 0.1, 0.8)
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
        # categorical_cols = ()
        categorical_cols = (2,  # Pclass
                            4,  # Sex
                            11  # Embarked
        )
        # Convert certain columns to float values (so we can use numpy arrays)
        converters = {4: lambda sex: {'male':0.0, 'female':1.0}[sex],
                        11: lambda embarked: {'S': 0.0, 'C': 1.0, 'Q': 2.0}[embarked]}
        data.process_data(splits=splits, use_cols=use_cols, categorical_cols=categorical_cols, converters=converters, 
                          filter_missing=True)

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

    # x_vals = [0,0,0,1,2,2,2]
    # y_vals = [0, 0.5, 1, 1, 1, 1, 1]
    # print(DataProcessor.get_AUC(x_vals, y_vals))

    X = np.concatenate((data.training_X, data.validation_X), axis=0)
    y = np.concatenate((data.training_y, data.validation_y), axis = 0)

    #Get probability for being the lowest unique value in y (basically 1 if it is lowest unique value, 0 if it isnt)
    unique_y = np.unique(y)
    y_true_proba = np.array([1 if y_elem == unique_y[0] else 0 for y_elem in y])

    random_forest = RandomForestClassifier(n_estimators=30, min_node_size=1, max_features=4, random_seed=None)
    random_forest.fit(X, y)

    my_f_predictions = random_forest.predict(X)
    my_f_proba = random_forest.predict_proba(X)
    my_f_xentropy = metrics.log_loss(y, my_f_proba)
    my_f_num_correct = np.sum(np.equal(my_f_predictions, y))
    my_f_percent_correct = my_f_num_correct / len(my_f_predictions)
    print(f"{my_f_num_correct} / {len(my_f_predictions)}; Train Accuracy:{my_f_percent_correct:.4f}, Xentropy:{my_f_xentropy}")
    
    skrf = ensemble.RandomForestClassifier(n_estimators=30, max_features=4)
    skrf.fit(X, y)
    skrf_train_predictions = skrf.predict(X)
    skrf_train_proba = skrf.predict_proba(X)
    skrf_train_xentropy = metrics.log_loss(y, skrf_train_proba)
    skrf_train_num_correct = np.sum(np.equal(skrf_train_predictions, y))
    skrf_train_percent_correct = skrf_train_num_correct / len(skrf_train_predictions)
    print(f"{skrf_train_num_correct} / {len(skrf_train_predictions)}; SKRF Train Accuracy:{skrf_train_percent_correct:.4f}, Xentropy:{skrf_train_xentropy}")

    skrf_POFD_arr, skrf_POD_arr = Metrics.get_ROC_data(skrf_train_proba[:, 0], y_true_proba)
    skrf_AUC = Metrics.add_ROC_curve(skrf_POFD_arr, skrf_POD_arr, f"sklearn rf -", color="r")
    my_f_POFD_arr, my_f_POD_arr = Metrics.get_ROC_data(my_f_proba[:, 0], y_true_proba)
    my_f_AUC = Metrics.add_ROC_curve(my_f_POFD_arr, my_f_POD_arr, f"my rf -", color="g")
    Metrics.show_ROC_curve("aiosjdfoais")


    # predictions = random_forest.predict(X)
    # probas = random_forest.predict_proba(X)
    # print(predictions)
    # print(probas)
