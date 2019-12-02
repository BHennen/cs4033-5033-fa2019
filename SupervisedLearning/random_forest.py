import math

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

    n_estimators : integer, optional (default=50)
        The number of trees in the forest.

    min_node_size : integer, optional (default=1)
        The minimum number of samples in a leaf node. Nodes will not be split at this size.
        For classification, the default value for minimum node size is one.

    max_features : integer, optional (default=⌊√p⌋)`
        The number of features to consider when looking for the best split.
        For classification, the default value for m is ⌊√p⌋.
    '''

    def __init__(self, n_estimators=50, min_node_size=1, max_features="auto"):
        self.n_estimators = n_estimators
        self.min_node_size = min_node_size
        self.max_features = max_features

    def fit(self, X, y):
        # TODO: update max_features to be ⌊√p⌋ if it is == "auto"
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


