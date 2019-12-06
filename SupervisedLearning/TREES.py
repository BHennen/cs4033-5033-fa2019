import math
import numpy as np
from collections import defaultdict

class RandomForestClassifier():
    ''' A random forest classifier using GINI impurity for the decision tree split criterion.

    ***
    Pseudocode: (based on Elements of Statistical Learning)
    For each tree b:
        From training data, take sample of size N with replacement
        Until min node size is reached:
            Select a subset of m features(columns) at random from the p features available
            Pick the best split-point (the one that produces the most separation between observations
                in left node vs right node) among the m features.
            Split the node into two daughter nodes
    Output the ensemble of trees

    For a classification prediction at a point x:
        Let Cb(x) be the class prediction of the bth random-forest tree.
        Then Crf(x) = majority-vote for all Cb(x) in B
    ***

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

        for tree in range(self.n_estimators):
            sapling = self._grow_tree(X, y, sample_size)

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
    
    def _grow_tree(self, X, y, sample_size):
        '''
        Make a new DecisionTreeClassifier.
        '''
        ## From training data, take sample of size N with replacement
        # Choose indices randomly based the size of the sample
        indices = self.rand_state.randint(0, sample_size, sample_size)
        # TODO: Make new tree from data using only specified indices.
        pass


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

    def __init__(self, min_node_size=1, max_features="auto", random_seed=None):
        self.min_node_size = min_node_size
        self.max_features = max_features
        if isinstance(random_seed, tuple):
            self.rand_state = np.random.RandomState()
            self.rand_state.set_state(random_seed)
        else:
            self.rand_state = np.random.RandomState(seed=random_seed)
    
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

        # init node
        root_GINI_impurity = DecisionTreeNode.calc_GINI_impurity(y)
        root_parent = None
        self.root_node = DecisionTreeNode(root_parent, root_GINI_impurity, sample_indices, self.min_node_size)
        nodes = [self.root_node]
        # Loop through nodes, splitting them until we've turned all nodes into nodes that can't be split anymore
        while nodes:
            node = nodes.pop()
            new_nodes = node.split(X, y, self.max_features, self.rand_state, X_sorted_idx)
            for new_node in new_nodes:
                nodes.append(new_node)

    def predict(X):
        '''
        Given values X, determine what class y they fall into.
        '''
        pass

class DecisionTreeNode():
    def __init__(self, parent, GINI_impurity, sample_indices, min_node_size):
        self.parent = parent
        self.GINI_impurity = GINI_impurity
        self.sample_indices = sample_indices
        self.min_node_size = min_node_size
        self.l_child = None
        self.r_child = None
        self.split_feature = None
        self.split_value = None
        self.predicted_value = None

    def split(self, X, y, max_features, rand_state, X_sorted_idx):
        ## Split this node into two others (if it improves overall GINI impurity)
        # Don't split if we're at min size, and save the predicted y value
        if len(X) <= self.min_node_size:
            self.predicted_value = np.mean(y[self.sample_indices])
            return []

        # Randomly determine which features(columns) we will use as split points 
        features = rand_state.choice(X.shape[1], max_features, replace=False)

        # If sample size not set, set sample indices to be the range from 0 to length of X (use the whole dataset)
        if self.sample_indices is None:
            self.sample_indices = range(len(X))
        # Count how many samples we have at each index
        index_counts = defaultdict(int)
        for index in self.sample_indices:
            index_counts[index] += 1

        # Keep track of min GINI impurity value
        min_GINI_impurity = self.GINI_impurity

        # Iterate through features and get the best split point for the feature
        for feature in features:
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
            
            # Now that we've sorted the data for the feature, loop through rows until we find consecutive
            # ones that are unequal, and try that as the split point
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
                        self.l_child = DecisionTreeNode(self, l_GINI_impurity, l_sample_indices, self.min_node_size)
                        self.r_child = DecisionTreeNode(self, r_GINI_impurity, r_sample_indices, self.min_node_size)
                        self.split_feature = feature
                        self.split_value = (X_sorted[split_idx - 1] + X_sorted[split_idx]) / 2 # Value is the mean of the two consecutive data points
                split_idx += 1
        
        #Done splitting; Return list of child nodes (or empty list if this is better off as leaf node)
        if self.l_child is not None:
            return [self.l_child, self.r_child]
        else:
            self.predicted_value = np.mean(y[self.sample_indices])
            return []

    @staticmethod    
    def calc_GINI_impurity(y):
        # Calculate the GINI impurity with the given y values
        vals, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        GINI_impurity = 1 - sum(ps**2)
        return GINI_impurity
