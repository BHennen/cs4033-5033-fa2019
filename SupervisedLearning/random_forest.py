import math
import numpy as np

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
            self.max_features = X.shape[1]**0.5 # Num features is how many columns there are in dataset
        
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

    Pseudocode:
    for node in internal_nodes:
        vals, counts = np.unique(y, return_counts=True)
        ps = counts/len(y)
        gini = 1-sum(ps**2)
        for feature in features:

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
    
    def fit(self, X, y, X_sorted_idx=None):
        '''
        Fit data X to target values in y using decision tree classifier.

        Parameters
        ----------
        X : numpy array
        y : numpy array
        '''

        # Sort each column of X and save sorted indices in 2d array
        if X_sorted_idx is None:
            X_sorted_idx = np.argsort(X, axis=0)
        
        

class DecisionTreeNode():
    def __init__(self):
        pass