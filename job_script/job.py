import os
import sys
import logging.config
import logging
import random

from rltree.train import RLdecisionTreeTrain
from multiprocess import Process, Manager 



def hyper_param_opt(eval_method, valid_proj, job_num, job_id, nbr_tries, scores):      
    
    # Setting up logging
    curdir = os.curdir
    if not os.path.exists(os.path.join(curdir, "logging", str(job_id), str(os.getpid()))):
        os.makedirs(os.path.join(curdir, "logging", str(job_id), str(os.getpid())))

    logging.config.fileConfig(
        os.path.join(curdir,"logging",'logging.conf'),
        # defaults={"logfilename":os.path.join(curdir, "logging", str(job_id), str(os.getpid()), f"logs_{eval_method}_{valid_proj[:-4]}.log")}
        defaults={"logfilename":os.path.join(curdir,"logging",f"logs_{eval_method}_{valid_proj[:-4]}.log")}
    )
    logger = logging.getLogger('Main')
    # Uncomment to disable debug level logging
    #logging.disable(logging.DEBUG)

    logger.info("defining hyperparameters")
    # DL hyper-parameters
    HIDDEN_SIZE = 128
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 256
    LR_ACTOR = 1e-2   # change these!
    LR_CRITIC = 1e-2
    # RL agent hyper-parameters
    gamma = 0.99      # reward calc
    epsilon = 0.3     # greedy-eps param
    n_episodes = 5
    save_every = 1000
    # DT hyper-parameters 
    max_depth = 7
    use_meth_1 = True
    nbr_of_conv = 2
    # random seed
    seed = 42
    # disable warnings, sometimes there's a warning about metrics calculation when the classifier only predicts one class
    import warnings
    warnings.filterwarnings("ignore")

    cols_to_keep = 32
    columns = ['ci_skipped', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev',
           'age', 'nuc', 'exp', 'rexp', 'sexp', 'TFC', 'is_doc', 'is_build',
           'is_meta', 'is_media', 'is_src', 'is_merge', 'FRM', 'COM', 'CFT',
           'classif', 'prev_com_res', 'proj_recent_skip', 'comm_recent_skip',
           'same_committer', 'is_fix', 'day_week', 'CM', 'commit_hash']


    # random search hyper params tuning
    
    for _ in range(nbr_tries): 

        # Possible hyper-params
        max_depth = random.choice([3,5,7,10,12])
        max_depth=3
        lr = random.choice([1e0, 1e-1, 1e-2, 1e-3])
        epsilon = random.choice([0.1, 0.2, 0.3, 0.4])
        gamma = random.choice([1, 0.9, 0.8, 0.7, 0.5])
        BATCH_SIZE = random.choice([32, 64, 128, 256, 512, 1024])
        n_episodes = random.choice([200, 300, 500, 700, 1000])
        n_episodes=3

        # Initializing training instance
        training = RLdecisionTreeTrain(HIDDEN_SIZE, BUFFER_SIZE, BATCH_SIZE, lr, lr, gamma, epsilon ,max_depth, use_meth_1, nbr_of_conv, n_episodes, curdir, seed, columns, cols_to_keep, save_every)

        # Starting training
        logger.info("starting process for job {}".format(job_num))
        logger.info(f"hyperparameters choice: max_depth={max_depth} lr={lr} epsilon={epsilon} gamma={gamma} batch_size={BATCH_SIZE} n_episodes={n_episodes} seed={seed}")
        
        if eval_method=="within":
            final_eval_score = training.within_eval(valid_proj)
            scores[final_eval_score] = f"max_depth={max_depth} lr={lr} epsilon={epsilon} gamma={gamma} batch_size={BATCH_SIZE} n_episodes={n_episodes} seed={seed} output={os.getpid()}"
        else:
            final_eval_score = training.cross_eval(valid_proj)
            scores[final_eval_score] = f"max_depth={max_depth} lr={lr} epsilon={epsilon} gamma={gamma} batch_size={BATCH_SIZE} n_episodes={n_episodes} seed={seed} output={os.getpid()}"


if __name__ == "__main__":

    # Parsing input
    if len(sys.argv)!=4:
        sys.exit("Incorrect arguments recieved. Aborting!")
    else:
        eval_method = sys.argv[1]
        job_num = int(sys.argv[2])
        job_id = sys.argv[3]
        if eval_method not in ["within","cross"]:
            sys.exit("Unknowm argument '{}' for evaluation method. Aborting!".format(eval_method))

    validation_projects = ['candybar-library.csv','GI.csv', 'mtsar.csv', 'ransack.csv', 'SemanticMediaWiki.csv', 'contextlogger.csv', 'grammarviz2_src.csv', 'parallec.csv', 'SAX.csv', 'solr-iso639-filter.csv', 'future.csv', 'groupdate.csv', 'pghero.csv', 'searchkick.csv', 'steve.csv']
    valid_proj = validation_projects[job_num]

    nbr_tries = 5
    manager = Manager()
    scores = manager.dict()
    processes = []

    for n in range(nbr_tries):
        p = Process(target=hyper_param_opt, args=(eval_method, valid_proj, job_num, job_id, nbr_tries, scores))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Setting up logging
    curdir = os.curdir
    if not os.path.exists(os.path.join(curdir, "logging", str(job_id))):
        os.makedirs(os.path.join(curdir, "logging", str(job_id)))


    logging.config.fileConfig(
        os.path.join(curdir,"logging",'logging.conf'),
        # defaults={"logfilename":os.path.join(curdir, "logging", str(job_id), f"logs_{eval_method}_{valid_proj[:-4]}.log")}
        defaults={"logfilename":os.path.join(curdir, "logging", f"Hyper_param_logs_{eval_method}_{valid_proj[:-4]}.log")}
    )

    logger = logging.getLogger('Main')

    logger.info(f"scores of tried hyper-params: {scores}")
        
    max_score = max([score for score in scores.keys()])
    logger.info(f"Best score for {eval_method}-{valid_proj[:-4]}: f1={max_score}, obtained using hyper-params: {scores[max_score]}\n")