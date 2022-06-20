import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from rltree.agent import flatten



use_meth_1 = True
env_logger = logging.getLogger('Environment')
env_logger.info("using convolution method {}".format(1 if use_meth_1 else 2))

def generate_state(model, features, thresholds, nbr_of_conv):
    """ Choose which convolution method to use"""
    env_logger.debug("generating state")
    if use_meth_1:
        return generate_state_1(model, features, thresholds, nbr_of_conv=0)
    else:
        return generate_state_2(model, nbr_of_conv)



#==================================================================================================================================#




def tree_convolution_1(model, features, thresholds, node=0):
    """
        convolution of each 3 nodes, with overlapping => child of 1 subtree is parent of the next
        => after each conv, slit nodes' featuress and thresholdss are updated and terminal nodes are removed
        returns new features,thresholds values as lists
    """
    visited = []
    queue = []
    visited.append(node)
    queue.append(node)
    new_features = features
    new_thresholds = thresholds

    # BFS traversal
    while queue:
        node = queue.pop(0) 
        left_node = model.tree.children_left[node]
        right_node = model.tree.children_right[node]
        #print (node, left_node, right_node, end = "\n")
        if left_node not in visited:
            visited.append(left_node)
            queue.append(left_node)
        if right_node not in visited:
            visited.append(right_node)
            queue.append(right_node)

        # define kernel here
        if features[left_node]==-2 and features[right_node]==-2: # both child nodes are leaves
            new_node_features = -2
            new_node_thresholds = 0
        elif features[left_node]==-2: # left node is leaf
            new_node_features = (features[node]+features[right_node])/2
            new_node_thresholds = (thresholds[node]+thresholds[right_node])/2
        elif features[right_node]==-2: # right node is leaf
            new_node_features = (features[node]+features[left_node])/2
            new_node_thresholds = (thresholds[node]+thresholds[left_node])/2
        else:
            new_node_features = (features[node]+features[left_node]+features[right_node])/3
            new_node_thresholds = (thresholds[node]+thresholds[left_node]+thresholds[right_node])/3
        # save new values
        new_features[node] = new_node_features # will auto convert to int 
        new_thresholds[node] = new_node_thresholds

    return new_features, new_thresholds



def generate_state_1(model, features, thresholds, nbr_of_conv):
    features, thresholds = features.copy(), thresholds.copy()
    for _ in range(nbr_of_conv):
        new_features, new_thresholds = tree_convolution_1(model, features, thresholds) # model must be fitted => add condition?
        features, thresholds = new_features, new_thresholds
    return flatten(features, thresholds)



#==================================================================================================================================#



def tree_convolution_2(model, encoded_nodes, node=0):
    """
        convolution of each 3 nodes, with overlapping => child of 1 subtree is parent of the next
        => after each conv, slit nodes' featuress and thresholdss are updated and terminal nodes are removed
        takes a 1 hot representation of a tree and returns it after one convolution (same shape)
    """
    new_encoded_nodes = encoded_nodes.copy()
    
    visited = []
    queue = []
    visited.append(node)
    queue.append(node)
    # BFS traversal
    while queue:
        node = queue.pop(0) 
        left_node = model.tree.children_left[node]
        right_node = model.tree.children_right[node]
        #print (node, left_node, right_node, end = "\n")
        if left_node not in visited:
            visited.append(left_node)
            queue.append(left_node)
        if right_node not in visited:
            visited.append(right_node)
            queue.append(right_node)

        # define kernel here
        if encoded_nodes[left_node] is None and encoded_nodes[right_node] is None: # both child nodes are leaves
            new_enc_node = None
        elif encoded_nodes[left_node] is None: # left node is leaf
            vect1 = encoded_nodes[node]
            vect2 = encoded_nodes[right_node]
            new_enc_node = node_aggregate(vect1, vect2)
        elif encoded_nodes[right_node] is None: # right node is leaf
            vect1 = encoded_nodes[node]
            vect2 = encoded_nodes[left_node]
            new_enc_node = node_aggregate(vect1, vect2)
        else:
            vect1 = encoded_nodes[node]
            vect2 = encoded_nodes[left_node]
            vect3 = encoded_nodes[right_node]
            new_enc_node = node_aggregate(vect1, vect2, vect3)
        # save new values
        new_encoded_nodes[node] = new_enc_node

    return new_encoded_nodes


def node_aggregate(vect1, vect2, vect3=None):
    """
        given 2 or 3 vectors,
        for all attributes
        if all vectors contain the same attribute, this will average them
        else it will keep the non null value
    """
    out = vect1
    if vect3 is None:
        for i in range(number_of_attributes):
            if vect1[i]!=0 and vect2[i]==0:
                out[i] = vect1[i]
            if vect1[i]!=0 and vect2[i]!=0:
                out[i] = (vect1[i]+vect2[i])/2
    else:
        for i in range(number_of_attributes):
            if vect1[i]==0:
                if vect2[i]!=0 and vect3[i]!=0:
                    out[i] = (vect2[i]+vect3[i])/2
                if vect2[i]==0 and vect3[i]!=0:
                    out[i] = vect3[i]
                if vect2[i]!=0 and vect3[i]==0:
                    out[i] = vect2[i]

            else:
            # if vect1[i]!=0:
                if vect2[i]!=0 and vect3[i]!=0:
                    out[i] = (vect1[i]+vect2[i]+vect3[i])/2
                if vect2[i]==0 and vect3[i]!=0:
                    out[i] = (vect1[i]+vect3[i])/2
                if vect2[i]!=0 and vect3[i]==0:
                    out[i] = (vect1[i]+vect2[i])/2
    return out


def encode_tree(model):
    """
        Creates a vector representation to a decision tree
        each node is represented by a 1 hot vector containing the threshold at the attribute's index
        output is 1D Tensor of node representations
    """
    # fit 1 hot encoder
    enc = OneHotEncoder()
    arr = np.array(range(number_of_attributes)).reshape(-1, 1)
    enc.fit(arr)

    # build init vect
    n = len(model.features)
    encoded_nodes = []
    for i in range(n):
        feat, thres = model.features[i], model.thresholds[i]
        if feat==-2:
            encoded_nodes.append(None)
            continue
        enc_node = enc.transform([[feat]])*thres
        encoded_nodes.append(enc_node.toarray()[0])

    return encoded_nodes


def shorten_state(encoded_nodes):
    # finding the new state size
    n = len(encoded_nodes)
    out_size = n*number_of_attributes
    for node in encoded_nodes:
        if node is None:
            out_size -= number_of_attributes
    
    out = torch.zeros(out_size).to(device)
    
    # removing None values / terminal nodes
    k = -1
    for i in range(n):
        node = encoded_nodes[i]
        if node is None:
            continue
        k += 1
        for j in range(number_of_attributes):
            out[k * number_of_attributes + j] = node[j]
    return out


def generate_state_2(model, nbr_of_conv):
    encoded_nodes = encode_tree(model)
    for _ in range(nbr_of_conv):
        new_encoded_nodes = tree_convolution_2(model, encoded_nodes) # model must be fitted => add condition?
        encoded_nodes = new_encoded_nodes
    return shorten_state(encoded_nodes)