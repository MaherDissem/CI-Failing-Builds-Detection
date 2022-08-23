# %% [markdown]
# ga

# %%
"""Class that represents the solution to be evolved."""
import random
class Solution():
    def __init__(self, all_possible_params):
        self.entry = {}
        self.score = 0.
        self.all_possible_params = all_possible_params
        self.params = {}  #  represents model parameters to be picked by creat_random method
        self.model = None
        
    """Create the model random params."""
    def create_random(self):
        for key in self.all_possible_params:
            self.params[key] = random.choice(self.all_possible_params[key])

    def set_params(self, params):
        self.params = params
      
    """
        Train the model and record the score.
    """
    def train_model(self, fn_train,params_fn):
        
        if self.score == 0.:
                res = fn_train(self.params,params_fn)
                self.score =  res["entry"]["F1"] #1-float(res["validation_loss"])
                self.model = res["model"]
                self.entry = res['entry']
            
    """Print out a network."""
    def print_solution(self):
        print("for params ", self.params , "the score in the train = ",self.score)

# %%
"""
Class that holds a genetic algorithm for evolving a population of params.
"""
from functools import reduce
from operator import add
import random
"""Class that implements genetic algorithm for Hyper-parameter tuning"""
class Optimizer():
    
    def __init__(self, GA_params, all_possible_params):
        """Create an optimizer."""
        self.random_select = GA_params["random_select"]
        self.mutate_chance = GA_params["mutate_chance"]
        self.retain = GA_params["retain"]
        self.all_possible_params = all_possible_params
    
    def create_population(self, count):
        """Create a population of random solutions."""
        pop = []
        for _ in range(0, count):
            # Create a random solution.
            solution = Solution(self.all_possible_params)
            solution.create_random()
            # Add the solution to our population.
            pop.append(solution)
        return pop

    @staticmethod
    def fitness(solution):
        """Return the score, which is our fitness function."""
        return solution.score

    def grade(self, pop):
        """Find average fitness for a population. """
        summed = reduce(add, (self.fitness(solution) for solution in pop))
        return summed / float((len(pop)))

    def crossover(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): parameters
            father (dict): parameters
        Returns:
            (list): combined params
        """
        children = []
        for _ in range(2):
            child = {}
            # Loop through the parameters and pick params for the kid.
            for param in self.all_possible_params:
                child[param] = random.choice([mother.params[param], father.params[param]] )

            solution = Solution(self.all_possible_params)
            solution.set_params(child)
            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                solution = self.mutate(solution)
            children.append(solution)
        return children
    
    
    def mutate(self, solution):
        """Randomly mutate one part of the solution."""
        # Choose a random key.
        mutation = random.choice(list(self.all_possible_params.keys()))
        # Mutate one of the params.
        solution.params[mutation] = random.choice(self.all_possible_params[mutation])
        return solution
    
    """Evolve a population of solutions."""
    def evolve(self, pop):
        #Get scores for each solution.
        graded = [(self.fitness(solution), solution) for solution in pop]
        #"Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        #Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)
        # define what we want to keep.
        parents = graded[:retain_length]
        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        
        # Add children, which are bred from two remaining solutions.
        if parents_length > 1 and desired_length> 0:
            children = []
            while len(children) < desired_length:
                if parents_length==2:
                    male_index = 1
                    female_index = 0
                else:
                    male_index = random.randint(0, parents_length-1)
                    female_index = random.randint(0, parents_length-1)
                
                # Assuming they aren't the same solutions...
                if male_index != female_index:
                    male = parents[male_index]
                    female = parents[female_index]
                    # crossover them.
                    babies = self.crossover(male, female)
                    # Add the children one at a time.
                    for baby in babies:
                        # Don't grow larger than desired length.
                        if len(children) < desired_length:
                            children.append(baby)
            parents.extend(children)
        return parents

# %%
from tqdm import tqdm
import threading
def train_sol_thread(solution,fn_train,params_fn,i):
    solution.train_model(fn_train,params_fn)
    
def train_population(pop, fn_train,params_fn):
    pbar = tqdm(total=len(pop))
    threads = list()
    i=1
    for solution in pop:
        x = threading.Thread(target=train_sol_thread, args=(solution,fn_train,params_fn,i))
        i=i+1
        threads.append(x)
        x.start()
        pbar.update(1)
        
    for index, thread in enumerate(threads):
        thread.join()
    pbar.close()


def get_average_score(pop):
    """Get the average score for a group of solutions."""
    total_scores = 0
    for solution in pop:
        total_scores += solution.score
    return total_scores / len(pop)


def generate(all_possible_params, fn_train , params_fn):
    """Generate the optimal params with the genetic algorithm."""
    """ Args:
            GA_params: Params for GA
            all_possible_params (dict): Parameter choices for the model
            train_set : training dataset
            fn_train : a function used to compute the prediction accuracy
    """
   
    GA_params = {
            "population_size": nbr_sol,
            "max_generations": nbr_gen,
            "retain": 0.7,
            "random_select":0.1,
            "mutate_chance":0.1
            }
    
    optimizer = Optimizer(GA_params ,all_possible_params)
    pop = optimizer.create_population(GA_params['population_size'])
    # Evolve the generation.
    for i in range(GA_params['max_generations']):
        # Train and get accuracy for solutions.
        train_population(pop,fn_train,params_fn)
        # Get the average accuracy for this generation.
        average_accuracy = get_average_score(pop)
        # Print out the average accuracy each generation.
        # Evolve, except on the last iteration.
        if i != (GA_params['max_generations']):
            evolved = optimizer.evolve(pop)
            if(len(evolved)!=0):
                pop=evolved
        else:
            pop = sorted(pop, key=lambda x: x.score, reverse=True)
    # Print out the top 2 solutions.
    size = len(pop)
    if size < 3:
        print_pop(pop[:size])
    else:
        print_pop(pop[:3])
    return pop[0].params ,pop[0].model,pop[0].entry

def print_pop(pop):
    for solution in pop:
        solution.print_solution()    


# %% [markdown]
# utils

# %%

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings # `do not disturbe` mode
warnings.filterwarnings('ignore')
sc = StandardScaler()
from numpy import arange
from numpy import argmax
import numpy as np

nbr_rep = 2 # repreats?
nbr_gen = 10
nbr_sol = 6
max_eval = nbr_gen*nbr_sol

with_smote = False 
hybrid_option = False # means smote and threshold moving

if hybrid_option:
    with_smote =True

import os

def getDataset(file_name):
    dataset = pd.read_csv("ordered-data/"+file_name, 
                          #parse_dates=['date'], 
                          index_col="date")
    dataset.sort_values(by=['date'], inplace=True)
    return dataset

def getDataset_2(valid_proj, type):
    columns = ['ci_skipped', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev',
        'age', 'nuc', 'exp', 'rexp', 'sexp', 'TFC', 'is_doc', 'is_build',
        'is_meta', 'is_media', 'is_src', 'is_merge', 'FRM', 'COM', 'CFT',
        'classif', 'prev_com_res', 'proj_recent_skip', 'comm_recent_skip',
        'same_committer', 'is_fix', 'day_week', 'CM', 'commit_hash']

    if type=="train":
        df_train = pd.DataFrame(columns=columns, dtype=np.float)
        for dirname, _, filenames in os.walk("ordered-data"):
            for filename in filenames:
                if filename[-4:]==".csv" and filename!=valid_proj:
                    new_data = pd.read_csv(os.path.join(dirname, filename),index_col="date")
                    new_data.sort_values(by=['date'], inplace=True)
                    df_train = pd.concat([df_train, new_data])
        # print(df_train.shape) # (15001, 34) => (422, 33)
        # print(df_train)
        return df_train

    if type=="test":
        df_test = pd.read_csv(os.path.join("ordered-data", valid_proj),index_col="date")
        df_test.sort_values(by=['date'], inplace=True)
        return df_test

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def getBestThreshold(probs, y_train):
    # keep probabilities for the positive outcome only
    #probs = predicted_builds[:, 1]
    thresholds = arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [roc_auc_score(y_train, to_labels(probs, t)) for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    #print('\nThreshold=%.2f, AUC=%.2f' % (thresholds[ix], scores[ix]))
    return  thresholds[ix]


def failureInfo(dataset):
    condition =  dataset['ci_skipped'] > 0
    rate = (dataset[condition].shape[0]) /dataset.shape[0]
    size=dataset.shape[0]
    return rate,size

def getEntry(y, predicted_builds):
    entry = {}
    entry["AUC"] =  roc_auc_score(y, predicted_builds)
    entry["accuracy"] =  accuracy_score(y, predicted_builds)
    entry["F1"] =  f1_score(y,predicted_builds)
    return entry

def predict_lstm(classifier,X,y):
    predicted_builds = classifier.predict(X)
    
    if with_smote and not hybrid_option:
        decision_threshold = 0.5
    else:
        decision_threshold = getBestThreshold(predicted_builds, y)
        
    predicted_builds = (predicted_builds >= decision_threshold)
    return getEntry(y, predicted_builds)

def isInt(n):
    try:
        n=int(n)
        return True
    except:
        return False
def online_validation_folds(dataset):
    train_sets=[]
    test_sets =[]
    fold_size = int(len(dataset) * 0.1)
    for i in range(6,11):
        train_sets.append(dataset.iloc[0:(fold_size*(i-1))])
        test_sets.append(dataset.iloc[fold_size*(i-1):(fold_size*i)])
    return  train_sets, test_sets
def frange(start, stop=None, step=None):

    if stop == None:
        stop = start + 0.0
        start = 0.0

    if step == None:
        step = 1.0

    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start) # return float number
        start = start + step
        
def frange_int(start, stop=None, step=None):

    if stop == None:
        stop = start 
        start = 0

    if step == None:
        step = 1

    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield (start) # return int number
        start = start + step
 

# %% [markdown]
# lstm tuner

# %%
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from timeit import default_timer as timer


def train_preprocess(dataset_train,time_step):
    training_set = dataset_train.iloc[:,0:32].values
    if with_smote:
        X= training_set
        y= dataset_train.iloc[:,0].values
        X, y = SMOTE().fit_resample(X, y)
        training_set = X
    
    X_train = []
    y_train = []
    for i in range(time_step, len(training_set)):
        X_train.append(training_set[i-time_step:i, 0])#0 : we have only one column in training_set
        y_train.append(training_set[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_train.shape[0] : nbr of lines or observations; X_train.shape[1]:nbr of columns or timestep; 1: nbr of indicators
    return X_train,y_train

def test_preprocess(dataset_train,dataset_test,time_step):
    #Test preprocessing
    y_test = dataset_test.iloc[:,0:1].values
    dataset_total = pd.concat((dataset_train['ci_skipped'], dataset_test['ci_skipped']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values
    inputs = inputs.reshape(-1,1)
    X_test = []
    for j in range(time_step, len(inputs)):
        X_test.append(inputs[j-time_step:j, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test,y_test

def get_threshold_list(dataset):
    cdt =  dataset['ci_skipped'] > 0
    failure_rate = (dataset[cdt].shape[0] /dataset.shape[0])
    return list(frange(0.01,max(1,failure_rate), 0.1))

class LSTMWorker(Worker):
    def __init__(self,  train_set, **kwargs):
        super().__init__(**kwargs)
        self.train_set= train_set

    def compute(self, config, *args, **kwargs):
        res = construct_lstm_model(config,self.train_set)
        return({
                    'loss': float(res["validation_loss"]),  # this is the a mandatory field to run hyperband,   
                    #remember: HpBandSter always minimizes!
                    'info': res["entry"] # can be used for any user-defined information - also mandatory
                })

def construct_lstm_model (network_params,train_set):
    X_train,y_train = train_preprocess(train_set,network_params["time_step"])# need to preprocess each time to tune the time_step
    drop = round(network_params["drop_proba"])
    # Initialising the RNN
    classifier = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = network_params["nb_units"], return_sequences = True, input_shape = (X_train.shape[1], 1)))
    classifier.add(Dropout(drop))
    # Adding LSTM layer and some Dropout regularisation
    for nbLayesr in range (0,network_params["nb_layers"]):
        classifier.add(LSTM(units = network_params["nb_units"], return_sequences = True))
        classifier.add(Dropout(drop))
    # Adding another LSTM layer without return_sequences
    classifier.add(LSTM(units = network_params["nb_units"]))
    classifier.add(Dropout(drop))
    # Adding the output layer
    classifier.add(Dense(units = 1,activation='sigmoid'))
    # Compiling the RNN
    classifier.compile(optimizer = network_params["optimizer"],
                       loss = 'binary_crossentropy',metrics=["accuracy"])
    
    es = EarlyStopping(monitor='loss',mode='min', verbose=1,patience=10)
    
     # Fitting the RNN to the Training set
    result =  classifier.fit(X_train, y_train, epochs = network_params["nb_epochs"]
                   , batch_size = network_params["nb_batch"],
                   verbose=0, callbacks=[es])
    
    # Get the lowest validation loss of the training epochs
    validation_loss = np.amin(result.history['loss']) 
    # Get prediction probs
    entry = predict_lstm(classifier,X_train,y_train)
    entry['validation_loss']=validation_loss
    return      {
                'validation_loss'  : validation_loss, #required by TPE,GA
                'model'   : classifier#required by GA
                ,"entry"  : entry #required by GA
                }
global data
global global_params
global global_model
global global_entry


def evaluate_tuner(tuner_option, train_set):
    global data
    data = train_set

    nb_units =  list(frange_int(32,64, 32))#[64]#,128,256
    nb_epochs = [4,5,6]#list(frange_int(5,10, 1))#list(frange_int(5,25, 5))#15,20,25,,10
    nb_batch =[4,8,16,32, 64]#,, . power of 2
    nb_layers = [1,2,3,4]
    optimizers = [ 'adam','rmsprop']#,
    time_steps = list(frange_int(30,61, 1))
    drops = list(frange_int(0.01,0.21, 0.01))

    start = timer()
    
    rnn_param_choices = {
        'nb_units':   nb_units,
        'nb_layers':  nb_layers,
        'optimizer':  optimizers,
        'time_step':  time_steps,
        'nb_epochs':  nb_epochs,
        'nb_batch':   nb_batch,
        'drop_proba': drops
        # 'decision_threshold'       :  threshold_list
    }
    best_params ,best_model , entry_train = generate(rnn_param_choices, construct_lstm_model, data)


    end = timer()
    period = (end - start)
    entry_train["time"] = period
    entry_train["params"] = best_params
    entry_train["model"]  = best_model
    return entry_train
  

# %% [markdown]
# main

# %%
import pandas as pd
import os
import csv

global columns_res,columns_comp
columns_res = ["proj"]+["algo"]+["iter"]+["AUC"]+["accuracy"]+["F1"]+["exp"]
tuner = "ga"
# bellwether="steve.csv"

def train(bellwether):
    trainset = getDataset_2(bellwether,"train")
    for iteration in range (1,nbr_rep):
        entry_train  = evaluate_tuner(tuner,trainset)
        best_params = entry_train["params"]
        best_model = entry_train["model"]
        entry_train["proj"] = bellwether
        for file_name in os.listdir("ordered-data"):
            if file_name!=bellwether:
                testset = getDataset_2(file_name,"test")
                X,y = test_preprocess(trainset,testset,best_params["time_step"])
                entry= predict_lstm(best_model,X,y)
                entry["iter"] = iteration
                entry["proj"] = file_name
                entry["exp"] =  1
                entry["algo"] = "LSTM"
                # save results to a csv file
    with open(f"LSTM-HPO-{bellwether}.csv", 'a') as f:
        csv_writer = csv.DictWriter(f)
        csv_writer.writerow(entry_train)
        csv_writer.writerow(entry)

import sys

if __name__ == "__main__":

    # Parsing input
    if len(sys.argv)!=3:
        sys.exit("Incorrect arguments recieved. Aborting!")
    else:
        job_num = int(sys.argv[1])
        job_id = sys.argv[2]


    validation_projects = ['candybar-library.csv','GI.csv', 'mtsar.csv', 'ransack.csv', 'SemanticMediaWiki.csv', 'contextlogger.csv', 'grammarviz2_src.csv', 'parallec.csv', 'SAX.csv', 'solr-iso639-filter.csv', 'future.csv', 'groupdate.csv', 'pghero.csv', 'searchkick.csv', 'steve.csv']
    valid_proj = validation_projects[job_num]
    train(valid_proj)