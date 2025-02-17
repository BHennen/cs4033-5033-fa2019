import math
import numpy as np
from collections import defaultdict

class RandomForestClassifier():
    ''' A random forest classifier using GINI impurity for the decision tree split criterion.

    Skeleton and parameters based on sklearn.ensemble.RandomForestClassifier.

    Parameters
    ----------
    n_estimators : integer, optional (default=50)
        The number of trees in the forest.

    min_node_size : integer, optional (default=1)
        The minimum number of samples in a leaf node. Nodes will not be split at this size.
        For classification, the default value for minimum node size is one.

    max_features : integer, optional (default=⌊√p⌋)`
        The number of features to consider when looking for the best split.
        For classification, the default value for m is ⌊√p⌋.
    
    random_seed : integer, optional (default=None)
        The random seed used to determine random elements of the algorithm. If integer provided, it is
        used as seed. If tuple provided, it is expected to be the tuple that is returned by a call to an
        instance of RandomState.get_state()
    '''

    def __init__(self, n_estimators=50, min_node_size=1, max_features="auto", random_seed=None):
        self.n_estimators = n_estimators
        self.min_node_size = min_node_size
        self.max_features = max_features
        if isinstance(random_seed, tuple):
            self.rand_state = np.random.RandomState()
            self.rand_state.set_state(random_seed)
        else:
            self.rand_state = np.random.RandomState(seed=random_seed)
        self.forest = None

    def fit(self, X, y):
        '''
        Fit data X to target values in y using random forest.

        Parameters
        ----------
        X : numpy array
        y : numpy array
        '''
        # update max_features to be ⌊√p⌋ if it is == "auto"
        if self.max_features == "auto":
            self.max_features = int(X.shape[1]**0.5)  # Num features is how many columns there are in dataset
        
        sample_size = X.shape[0] # num rows in predictor data
        X_sorted_idx = np.argsort(X, axis=0) # Presort X based on each column

        #Determine unique classes in y
        self.y_classes = np.unique(y)

        # Create a n_estimators trees
        self.forest = []
        for _ in range(self.n_estimators):
            tree = self._grow_tree(X, y, sample_size, X_sorted_idx)
            self.forest.append(tree)
        return self

    def predict(self, X):
        '''
        Predict classes for X values
        '''
        # Get predictions from each tree
        predictions = []
        predictions_list = np.column_stack([tree.predict(X) for tree in self.forest])
        for x in predictions_list:
            values, counts = np.unique(x, return_counts=True)
            x_pred = values[counts.argmax()]
            predictions.append(x_pred)
        # # Get counts of each prediction
        # values, counts = np.unique(predictions, return_counts=True)
        # # Prediction is the value with the most counts
        # prediction = values[counts.argmax()]
        return np.array(predictions)


    def predict_proba(self, X):
        '''
        Predict probabilities for classes in y for given X values
        '''
        #Loop through all trees, getting their individual prediction
        probas = [tree.predict_proba(X) for tree in self.forest]
        #Get mean of predictions
        predict_proba = np.mean(probas, axis=0)
        return predict_proba
            
    
    def _grow_tree(self, X, y, sample_size, X_sorted_idx):
        '''
        Make a new DecisionTreeClassifier.
        '''
        ## From training data, take sample of size N with replacement
        # Choose indices randomly based the size of the sample
        indices = self.rand_state.randint(0, sample_size, sample_size)

        # Make new tree from data using only specified indices.
        tree = DecisionTreeClassifier(min_node_size=self.min_node_size,
                                      max_features=self.max_features,
                                      random_seed=self.rand_state.get_state(),
                                      y_classes=self.y_classes)
        
        # Fit the given data to this tree
        tree.fit(X, y, sample_indices=indices, X_sorted_idx=X_sorted_idx)
        return tree
        


class DecisionTreeClassifier():
    '''
    Build a decision tree using the GINI impurity as its criterion for splitting.

    Parameters
    ----------
    min_node_size : integer, optional (default=1)
        The minimum number of samples in a leaf node. Nodes will not be split at this size.
        For classification, the default value for minimum node size is one.

    max_features : integer, optional (default=⌊√p⌋)`
        The number of features to consider when looking for the best split.
        For classification, the default value for m is ⌊√p⌋.
    
    random_seed : integer, tuple, optional (default=None)
        The random seed used to determine random elements of the algorithm. If integer provided, it is
        used as seed. If tuple provided, it is expected to be the tuple that is returned by a call to an
        instance of RandomState.get_state()

    '''

    def __init__(self, min_node_size=1, max_features="auto", random_seed=None, y_classes=None):
        self.min_node_size = min_node_size
        self.max_features = max_features
        self.root_node = None
        if isinstance(random_seed, tuple):
            self.rand_state = np.random.RandomState()
            self.rand_state.set_state(random_seed)
        else:
            self.rand_state = np.random.RandomState(seed=random_seed)
        self.y_classes = y_classes
    
    def fit(self, X, y, sample_indices=None, X_sorted_idx=None):
        '''
        Fit data X to target values in y using decision tree classifier.

        Parameters
        ----------
        X : numpy array

        y : 1d numpy array (does not work for multiple feature target (yet?))

        sample_indices: Only use these indices when building the tree (repeats allowed)

        X_sorted_idx: The indices of X where each column is sorted. 
            idx = X_sorted_idx[1,0] will give the index in X which is the second lowest value for feature(column) 0
            X[idx] will give the second lowest value for feature(column) 0
        '''

        # Sort each column of X and save sorted indices in 2d array
        if X_sorted_idx is None:
            X_sorted_idx = np.argsort(X, axis=0)
        
        # update max_features to be ⌊√p⌋ if it is == "auto"
        if self.max_features == "auto":
            self.max_features = int(X.shape[1]**0.5)  # Num features is how many columns there are in dataset
        if self.max_features > X.shape[1]:
            raise IndexError("Max features is more than number of features in the data.")

        #Determine unique classes in y
        if self.y_classes is None:
            self.y_classes = np.unique(y)

        # init root node
        root_GINI_impurity = DecisionTreeNode.calc_GINI_impurity(y)
        root_parent = None
        self.root_node = DecisionTreeNode(root_parent, root_GINI_impurity, sample_indices, self.min_node_size)
        nodes = [self.root_node]
        # Loop through nodes, splitting them until we've turned all nodes into nodes that can't be split anymore
        while nodes:
            node = nodes.pop()
            new_nodes = node.split(X, y, self.max_features, self.rand_state, X_sorted_idx, self.y_classes)
            for new_node in new_nodes:
                nodes.append(new_node)
        return self

    def predict(self, X):
        '''
        Given values X, predict what class in y they fall into.
        '''
        if self.root_node is None:
            raise RuntimeError("Not able to predict without calling fit first.")

        predictions = []
        for x in X:
            # Starting at root node, follow decision tree split feature and split value until we hit 
            # a leaf node; the leaf_nodes predicted value will be the prediction for x
            node = self.root_node
            while not node.is_leaf():
                if x[node.split_feature] < node.split_value:
                    node = node.l_child
                else:
                    node = node.r_child
            # Found root node. Now take find index in nodes
            y_class_idx = np.argmax(node.prediction_counts)
            y_class_pred = self.y_classes[y_class_idx]
            predictions.append(y_class_pred)
        return np.array(predictions)
    
    def predict_proba(self, X):
        '''
        Given values X, predict what probabilities of class in y they fall into.
        '''
        predictions = []
        for x in X:
            # Starting at root node, follow decision tree split feature and split value until we hit
            # a leaf node; the leaf_nodes predicted value will be the prediction for x
            node = self.root_node
            while not node.is_leaf():
                if x[node.split_feature] < node.split_value:
                    node = node.l_child
                else:
                    node = node.r_child
            probabilities = node.prediction_counts / sum(node.prediction_counts)
            predictions.append(probabilities)
        if len(predictions) == 1:
            return predictions[0]
        else:
            return np.array(predictions)


class DecisionTreeNode():
    def __init__(self, parent, GINI_impurity, sample_indices, min_node_size, valid_features = None):
        self.parent = parent
        self.GINI_impurity = GINI_impurity
        self.sample_indices = sample_indices
        self.min_node_size = min_node_size
        self.l_child = None
        self.r_child = None
        self.split_feature = None
        self.split_value = None
        self.prediction_counts = None
        self.valid_features = valid_features
        

    def split(self, X, y, max_features, rand_state, X_sorted_idx, y_classes):
        ## Split this node into two others (if it improves overall GINI impurity)        

        # If sample size not set, set sample indices to be the range from 0 to length of X (use the whole dataset)
        if self.sample_indices is None:
            self.sample_indices = range(len(X))

        # Don't split if we're at min size, and save the predicted y value
        if len(self.sample_indices) <= self.min_node_size:
            # calculate probability of falling into a certain y class
            self._calc_counts(y, y_classes)
            return []
        
        # Count how many samples we have at each index
        index_counts = defaultdict(int)
        for index in self.sample_indices:
            index_counts[index] += 1

        # Keep track of min GINI impurity value
        min_GINI_impurity = self.GINI_impurity

        ## Loop through random features that will produce a split point or until we've hit max_features
        # Randomly determine which features(columns) we will use as split points 
        if self.valid_features is None:
            self.valid_features = [i for i in range(X.shape[1])]
        # Get permutation of valid features to put into random order
        rand_features = rand_state.permutation(self.valid_features)

        # Iterate through features and get the best split point for the feature
        num_features_tried = 0
        for feature in rand_features:
            # Check if we've tried maximum number of features
            if num_features_tried >= max_features:
                break
            
            # Sort data for this feature
            X_sorted = []
            y_sorted = []
            sample_idx_sorted = []
            # Iterate through the already sorted data for this feature
            for sorted_idx in X_sorted_idx[:, feature]:
                # Add however many samples we had at this index we counted earlier
                for _ in range(index_counts[sorted_idx]):
                    X_sorted.append(X[sorted_idx, feature])
                    y_sorted.append(y[sorted_idx])
                    sample_idx_sorted.append(sorted_idx)
            
            # Check if beginning of X sorted and end of X sorted are the same; if they are this is not good split point
            # and remove it from the list of valid features
            if X_sorted[0] == X_sorted[-1]:
                self.valid_features.remove(feature)
                continue

            # Now that we've sorted the data for the feature and found that the feature does change, loop through rows until we 
            # find consecutive ones that are unequal, and try that as the split point
            split_idx = 1
            while split_idx < len(X_sorted):
                if X_sorted[split_idx - 1] != X_sorted[split_idx]:
                    #Found potential split point, calculate GINI impurity for left and right potential nodes
                    l_y = y_sorted[:split_idx]
                    r_y = y_sorted[split_idx:]
                    l_GINI_impurity = DecisionTreeNode.calc_GINI_impurity(l_y)
                    r_GINI_impurity = DecisionTreeNode.calc_GINI_impurity(r_y)
                    l_weight = len(l_y) / len(y_sorted)
                    r_weight = len(r_y) / len(y_sorted)
                    weighted_avg_GINI_impurity = l_weight * l_GINI_impurity + r_weight * r_GINI_impurity
                    # Check if split point produced a lower GINI impurity than the min so far
                    if weighted_avg_GINI_impurity < min_GINI_impurity:
                        #Found new split point; create new child nodes and then continue searching
                        l_sample_indices = sample_idx_sorted[:split_idx]
                        r_sample_indices = sample_idx_sorted[split_idx:]
                        self.l_child = DecisionTreeNode(self, l_GINI_impurity, l_sample_indices, self.min_node_size, list(self.valid_features))
                        self.r_child = DecisionTreeNode(self, r_GINI_impurity, r_sample_indices, self.min_node_size, list(self.valid_features))
                        self.split_feature = feature
                        self.split_value = (X_sorted[split_idx - 1] + X_sorted[split_idx]) / 2 # Value is the mean of the two consecutive data points
                split_idx += 1
            
            num_features_tried += 1
        
        #Done splitting; Return list of child nodes (or empty list if this is better off as leaf node)
        if self.l_child is not None:
            return [self.l_child, self.r_child]
        else:
            self._calc_counts(y, y_classes)
            return []

    def is_leaf(self):
        '''
        A node is a leaf node if it has no children (only valid after running split)
        '''
        return self.l_child is None

    def _calc_counts(self, y, y_classes):
        # calculate the counts in this node of all the y's for each class
        # Get all target values for the specified sample indices
        predictions = y[self.sample_indices]
        vals, counts = np.unique(predictions, return_counts=True)
        prediction_counts = np.zeros_like(y_classes, dtype=np.float_)
        predict_class_idx = 0
        # Loop through possible classes
        for y_class_idx, y_class in enumerate(y_classes):
            # If current y class matches the current y class in our predictions, set that probability and counts
            if y_class == vals[predict_class_idx]:
                prediction_counts[y_class_idx] = counts[predict_class_idx]
                predict_class_idx += 1
                if predict_class_idx >= len(vals):
                    break
        self.prediction_counts = prediction_counts

    @staticmethod    
    def calc_GINI_impurity(y):
        # Calculate the GINI impurity with the given y values
        vals, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        GINI_impurity = 1 - sum(ps**2)
        return GINI_impurity
