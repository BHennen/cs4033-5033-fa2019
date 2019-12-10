from data_processing import DataProcessor, Metrics, ModelProcessor
from TREES import DecisionTreeClassifier, RandomForestClassifier
import os
import numpy as np
from sklearn import ensemble, metrics, tree


def load_rf_data(cur_path):
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
        converters = {4: lambda sex: {'male': 0.0, 'female': 1.0}[sex],
                      11: lambda embarked: {'S': 0.0, 'C': 1.0, 'Q': 2.0}[embarked]}
        data.process_data(splits=splits, use_cols=use_cols, categorical_cols=categorical_cols, converters=converters,
                          filter_missing=True)
    return data

def load_rf_model(cur_path):
    model_folder_name = "trained_models\\random_forest"
    model_name = "rf_test_3"
    model_processor = ModelProcessor(cur_path, model_folder_name, model_name)
    return model_processor

def gen_best_random_forest(train_X, train_y, valid_X, valid_y, verbose=False, save_data=False):
    ''' Given training and validation data, find the best hyperparameters that fit the data for a random forest.
    '''

    # Calculate valid y true probabilites
    unique_y = np.unique(valid_y)
    valid_true_proba = np.array([1 if y_elem == unique_y[0] else 0 for y_elem in valid_y])

    # Make sure seeds are the same for each test so we can compare
    seeds = range(20) # Run 20 tests at each combination so we can get the average

    # How many trees in the forest
    trees_list = range(25, 225, 25)
    
    # Max number of features that are analyzed 
    max_features_list = range(1, train_X.shape[1] + 1) # from 1 feature to len(features)

    # Find best number of max_features that produces the lowest cross-entropy for 30 trees
    # Assuming best max_feature for this tree count applies to all tree counts
    min_cross_entropy = 999
    best_max_features = None
    features_vs_xentropy = []
    for max_features in max_features_list:
        cross_entropies = []
        for seed in seeds:
            # Make fit tree to training data
            rf = RandomForestClassifier(n_estimators=30, max_features=max_features, random_seed=seed).fit(train_X, train_y)
            # Get predictions for validation data
            valid_proba = rf.predict_proba(valid_X)
            # Get cross entropy
            cross_entropy = Metrics.cross_entropy(valid_proba[:, 0], valid_true_proba)
            cross_entropies.append(cross_entropy)
            if save_data:
                features_vs_xentropy.append([max_features, seed, cross_entropy])
        avg_cross_entropy = np.mean(cross_entropies)
        if verbose:
            print(f"Max Features:{max_features}, Avg Cross Entropy:{avg_cross_entropy:.4f}")
        # Save minimums
        if avg_cross_entropy < min_cross_entropy:
            min_cross_entropy = avg_cross_entropy
            best_max_features = max_features
    # Now that we have best number of features, find the best number of trees that produces lowest cross-entropy
    # and that will be our best tree
    min_cross_entropy = 999
    best_forest = None
    best_trees = None
    trees_vs_xentropy = []
    for trees in trees_list:
        cross_entropies = []
        for seed in seeds:
            # Make fit tree to training data
            rf = RandomForestClassifier(n_estimators=trees, max_features=best_max_features, random_seed=seed).fit(train_X, train_y)
            # Get predictions for validation data
            valid_proba = rf.predict_proba(valid_X)
            # Get cross entropy
            cross_entropy = Metrics.cross_entropy(valid_proba[:, 0], valid_true_proba)
            cross_entropies.append(cross_entropy)
            if save_data:
                trees_vs_xentropy.append([trees, seed, cross_entropy])
        avg_cross_entropy = np.mean(cross_entropies)
        if verbose:
            print(f"Trees:{trees}, Avg Cross Entropy:{avg_cross_entropy:.4f}")
        # Save minimums
        if avg_cross_entropy < min_cross_entropy:
            min_cross_entropy = avg_cross_entropy
            best_trees = trees
            best_forest = rf
            
    if verbose:
        print(f"Best Random forest- Trees:{best_trees}, Max Features:{best_max_features}, Avg Cross Entropy:{min_cross_entropy:.4f}")
    
    results = None
    if save_data:
        results_data = {"features_vs_xentropy": features_vs_xentropy, "trees_vs_xentropy": trees_vs_xentropy}
        results_header = {"features_vs_xentropy": "max_features,seed,cross_entropy", "trees_vs_xentropy": "trees,seed,cross_entropy"}
        results = (results_data, results_header)
    return (best_forest, best_trees, best_max_features, results)
    

if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    
    data = load_rf_data(cur_path)
    model_processor = load_rf_model(cur_path)

    train_X = data.training_X
    train_y = data.training_y
    valid_X = data.validation_X
    valid_y = data.validation_y
    test_X = data.test_X
    test_y = data.test_y

    best_forest = None
    try:
        # best_forest = model_processor.load_model()
        best_trees, best_max_features = (75, 1)
        # FIXME: best_trees, best_max_features = model_processor.load_parameters(["best_trees", "best_max_features"])
    except FileNotFoundError:
        best_forest, best_trees, best_max_features, results = gen_best_random_forest(train_X, train_y, valid_X, valid_y, verbose=True, save_data=True)
        model_processor.save_model(best_forest)
        model_processor.save_parameters(best_trees=best_trees, best_max_features=best_max_features)
        if results:
            results_data, results_header = results
            model_processor.save_results(results_data, results_header)
    
    # best_forest = RandomForestClassifier(n_estimators=20, max_features=1).fit(train_X, train_y)
    # # Now we have the best forest
    # # Evaluate the model
    # Metrics.evaluate_model(model, test_X, test_y)

    # # Compare against sklearn model
    # Metrics.compare_models(model1, model2, test_X, test_Y)
        
    #Get probability for being the lowest unique value in y (basically 1 if it is lowest unique value, 0 if it isnt)
    unique_y = np.unique(test_y)
    y_true_proba = np.array([1 if y_elem == unique_y[0] else 0 for y_elem in test_y])

    bf_percents = []
    sk_percents = []
    # Fit random forest 30 times and record accuracy
    for _ in range(30):
        bf = RandomForestClassifier(n_estimators=best_trees, max_features=best_max_features)
        bf.fit(train_X, train_y)
        bf_pred = bf.predict(test_X)
        bf_correct = np.sum(np.equal(bf_pred, test_y))
        bf_percent = bf_correct / len(bf_pred)
        bf_percents.append(bf_percent)

        sk = ensemble.RandomForestClassifier(n_estimators=best_trees, max_features=best_max_features)
        sk.fit(train_X, train_y)
        sk_pred = sk.predict(test_X)
        sk_correct = np.sum(np.equal(sk_pred, test_y))
        sk_percent = sk_correct / len(sk_pred)
        sk_percents.append(sk_percent)
    
    bf_mean_acc = np.mean(bf_percents)
    bf_sd_acc = np.std(bf_percents)
    sk_mean_acc = np.mean(sk_percents)
    sk_sd_acc = np.std(sk_percents)

    print(f"bf_mean_acc:{bf_mean_acc}, bf_sd_acc:{bf_sd_acc}, sk_mean_acc:{sk_mean_acc}, sk_sd_acc:{sk_sd_acc}")

    # Metrics.add_ROC_curve(my_f_proba[:, 0], y_true_proba, f"Our Random Forest -", color="g")
    # Metrics.add_ROC_curve(skrf_train_proba[:, 0], y_true_proba, f"Scikit-learn Random Forest -", color="r")
    # Metrics.show_ROC_curve("")

