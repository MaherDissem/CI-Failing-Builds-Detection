import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class modDecisionTree:
    """
    Represents the classification model
    based on sklearn implementation with added methods for modifying single nodes
    nodes are indexed depth first
    """


    def __init__(self, max_depth=3, random_state=42):
        # need to add init of other hyper-param
        self.logger = logging.getLogger('Tree')
        self.logger.debug("constructing tree with max depth {}".format(max_depth))
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=max_depth, splitter='random', random_state=random_state)
    

    def fit(self, X_train, y_train, columns_names):
        self.logger.debug("fitting tree")
        self.model.fit(X_train, y_train)
        self.tree = self.model.tree_
        self.n_nodes = self.tree.node_count       # nbr of nodes
        self.features = self.tree.feature         # list of each node's feature
        self.thresholds = self.tree.threshold     # list of each node's threshold
        self.columns_names = columns_names
        self.features_names = [list(self.columns_names)[i] for i in self.features]
        self.nodes_type = self.get_nodes_type()  # a node is either a split node or a terminal node/leaf


    def evaluate(self, X_val, y_val, reward_only=True, print_confusion=False):
        self.logger.debug("evaluating tree")
        y_pred = self.model.predict(X_val)
        metrics = {}
        metrics['F1'] = f1_score(y_val, y_pred)
        if not reward_only:
            metrics['recall'] = recall_score(y_val, y_pred)
            metrics['precision'] = precision_score(y_val, y_pred)
            metrics['accuracy'] = accuracy_score(y_val, y_pred)
            metrics['AUC'] = roc_auc_score(y_val, y_pred)
        if print_confusion:
            print(confusion_matrix(y_val, y_pred))
        return metrics


    def feature_importance(self):
        feat_imp = []
        for name, importance in zip(self.features_names , self.model.feature_importances_):
            feat_imp.append((name, importance))
        feat_imp.sort(key=lambda t:t[1], reverse=True)
        return feat_imp


    def plot_tree(self):
        self.logger.debug("plotting tree")
        plt.figure(figsize=(15,10))  # set plot size (denoted in inches)
        tree.plot_tree(self.model, fontsize=10, class_names=['pass','fail'])
        plt.show()


    def get_nodes_type(self):
        children_left = self.tree.children_left
        children_right = self.tree.children_right
        node_depth = np.zeros(shape=self.n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=self.n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # 'pop' ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            # If the left and right child of a node is not the same we have a split node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        return is_leaves


    def node_is_leaf(self, node):
        """Returns whether a node is a leaf (terminal node) or a split node"""
        return self.nodes_type[node]==1


    def set_node_threshold(self, node, value):
        if node>=self.tree.node_count:
            print("Error: selected node id is not in the tree.")
            return
        if self.node_is_leaf(node):
            print("Error: can't change a leaf node's threshold.")
            return
        self.thresholds[node] = value 


    def set_node_feature(self, node, feat_index=None, feat_name=None):
        if node>=self.tree.node_count:
            print("Error: selected node id is not in the tree.")
            return
        if self.node_is_leaf(node):
            print("Error: can't change a terminal node's feature.")
            return
        # convert feature name to index if supplied with name
        if feat_index==None:
            feat_index = self.features_names.index(feat_name)
        self.features[node] = feat_index