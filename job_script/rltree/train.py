#import random
import time
import os
import logging
import csv
#from collections import deque

import numpy as np
import pandas as pd
import torch
#from tqdm import tqdm
from sklearn.model_selection import train_test_split

from rltree.decisionTree import modDecisionTree
from rltree.agent import Agent
from rltree.environment import generate_state

import threading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RLdecisionTreeTrain:


    def __init__(self, hidden_size, buffer_size, batch_size, lr_actor, lr_critic, gamma, epsilon, max_depth, use_method1, nbr_of_conv, n_episodes, curdir, seed, columns, cols_to_keep, save_every):
        self.logger = logging.getLogger('Training')
        self.hidden_size = hidden_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.use_method1 = use_method1
        self.nbr_of_conv = nbr_of_conv
        self.n_episodes = n_episodes
        self.curdir = curdir
        self.seed = seed
        self.prev_metric = 0
        self.path = os.path.join(curdir, "dataset")
        self.columns = columns
        self.cols_to_keep = cols_to_keep
        self.save_every = save_every
    

    def train(self, X_train, y_train, X_val, y_val, df, eval_meth):

        t0 = time.time()

        self.logger.info("starting training with evaluation method {}".format(eval_meth))

        #writer = SummaryWriter("runs/") # for TensorBoard
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        model = modDecisionTree(max_depth=self.max_depth)
        model.fit(X_train, y_train, df.columns)

        number_of_attributes = X_train.shape[1]
        threshold_vector_size = X_train.shape[1]

        state = generate_state(model, model.features, model.thresholds, self.nbr_of_conv, self.use_method1, number_of_attributes)
        state_size = len(state)+1

        # manually setting hidden size = state_size//2
        agent = Agent(state_size, threshold_vector_size, number_of_attributes, self.seed, state_size//2, self.lr_actor, self.lr_critic, self.buffer_size, self.batch_size, self.gamma, self.curdir)

        self.logger.info(f'tree depth={self.max_depth}, state size={state_size}, number of attribute={number_of_attributes}')

        for p in [os.path.join(self.curdir, "checkpoints"), os.path.join(self.curdir, "results")]:
            if not os.path.exists(p):
                os.mkdir(p)
        #os.system('mkdir -p checkpoints results')
        #os.system('rm -f checkpoints/*')

        with open(os.path.join(self.curdir,"results",f"{eval_meth}-{threading.current_thread().ident}.csv"),'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([eval_meth])
            csv_writer.writerow([f"max_depth={self.max_depth} lr={self.lr_actor} epsilon={self.epsilon} gamma={self.gamma} batch_size={self.batch_size} n_episodes={self.n_episodes} seed={self.seed}"])

        start_ep = 1 # do not change

        # load checkpoint (comment/uncomment)
        # 
        # start_ep = 106
        # agent.load_checkpoint(start_ep-1)
        # 

        for i_episode in range(start_ep, self.n_episodes+1):
            self.logger.debug("starting episode {}".format(i_episode))
            # state init / reset => reset the DT model to initial values
            model = modDecisionTree(max_depth=self.max_depth)
            model.fit(X_train, y_train, df.columns)
            state = generate_state(model, model.features, model.thresholds, self.nbr_of_conv, self.use_method1, number_of_attributes)
            state = torch.cat((torch.Tensor([0]).to(device), state))
            self.prev_metric = 0

            #for t in tqdm(range(model.n_nodes)):
            for t in range(model.n_nodes):
                if model.node_is_leaf(t):
                    continue
                action = agent.act(state, eps=self.epsilon)
                # print(f"node={t}/{model.n_nodes}: {model.features[t]}<={model.thresholds[t]} => {action[0]}<={action[1]}")
                next_state, reward, done, info = self.env_step(model, t, action, X_train, y_train, number_of_attributes)
                if reward+0.1<0:
                    break
                next_state = torch.cat((torch.Tensor([t]).to(device), next_state))
                agent.step(state, action, reward, next_state, done)
                state = next_state
                res = model.evaluate(X_val, y_val, False, False)
                f1score, AUC = res['F1'], res['AUC']
                if done: 
                    break

            # save checkpoint to resume training
            if i_episode % self.save_every==0:
                agent.save_checkpoint(i_episode)

            # save results to a csv file
            with open(os.path.join(self.curdir,"results",f"{eval_meth}-{threading.current_thread().ident}.csv"),'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([i_episode, f1score, AUC])
            
            # save results for Tensorboard
            #tb_writer.add_scalar("Average F1 score", f1score, i_episode)

            # display classification metrics
            self.logger.info("Episode: {}, F1 score: {} AUC: {}".format(i_episode, f1score, AUC))


        self.logger.info("final score: {}".format(model.evaluate(X_val, y_val, False, False)))

        t1 = time.time()
        self.logger.info("training took {} min".format((t1-t0)/60))

        return model.evaluate(X_val, y_val, False, False)['F1']



    def env_step(self, model, node, action, X_train, y_train, number_of_attributes):
        """
            Environment step: updates the DT model given a node and an action
            and return the reward the new state of the tree.
        """
        # update tree
        model.set_node_feature(node, feat_index=action[0])
        model.set_node_threshold(node, value=action[1])
        next_state = generate_state(model, model.features, model.thresholds, self.nbr_of_conv, self.use_method1, number_of_attributes)
        # calc reward
        metrics = model.evaluate(X_train, y_train)
        current_metric = metrics['F1']
        reward = current_metric-self.prev_metric 
        self.prev_metric = current_metric
        done = 0
        if model.node_is_leaf(node):
            done = 1

        info = 0

        return next_state, reward, done, info 


    def within_eval(self, valid_proj):
        df = pd.read_csv(os.path.join(self.path, valid_proj))
        X = df.iloc[:,1:self.cols_to_keep]
        y = df.iloc[:,0].astype(int)

        X_train, X_val , y_train, y_val = train_test_split(np.array(X), np.array(y), test_size=0.2, shuffle=True, stratify=y, random_state=self.seed) # keep ratio of classes in split

        eval_meth = f'within_proj_{valid_proj}'[:-4]

        return self.train(X_train, y_train, X_val, y_val, df, eval_meth)

    
    def cross_eval(self, valid_proj):

        df_train = pd.DataFrame(columns=self.columns, dtype='object')
        for dirname, _, filenames in os.walk(self.path):
            for filename in filenames:
                if filename[-4:]==".csv" and filename!=valid_proj:
                    df_train = pd.concat([df_train, pd.read_csv(os.path.join(dirname, filename))])

        X_train = np.array(df_train.iloc[:,1:self.cols_to_keep])
        y_train = np.array(df_train.iloc[:,0].astype(int))

        df_val = pd.read_csv(os.path.join(self.path, valid_proj))
        df = df_val 

        X_val = np.array(df_val.iloc[:,1:self.cols_to_keep])
        y_val = np.array(df_val.iloc[:,0].astype(int))

        eval_meth = f'cross_proj_{valid_proj}'[:-4]

        return self.train(X_train, y_train, X_val, y_val, df, eval_meth)

