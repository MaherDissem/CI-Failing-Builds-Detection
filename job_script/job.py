import os
import sys
import logging.config
import logging

from rltree.train import RLdecisionTreeTrain


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
    
    # Setting up logging
    curdir = os.curdir
    if not os.path.exists(os.path.join(curdir,"logging",str(job_id))):
        os.mkdir(os.path.join(curdir,"logging",str(job_id)))
    logging.config.fileConfig(os.path.join(curdir,"logging",'logging.conf'),
                              defaults={"logfilename":os.path.join(curdir,"logging",str(job_id),'logs_{}_{}.log'.format(
                              eval_method, valid_proj[:-4]))})
    logger = logging.getLogger('Main')
    
    # Uncomment to disable debug level logging
    #logging.disable(logging.DEBUG)

    logger.info("defining hyperparameters")
    # hyper-parameters
    HIDDEN_SIZE = 32
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 256
    # learning rates
    LR_ACTOR = 1e-2  # change these!
    LR_CRITIC = 1e-2
    GAMMA = 0.99    # reward calc
    # size
    max_depth = 7
    use_meth_1 = True
    nbr_of_conv = 2
    n_episodes = 1000
    save_every = 1000
    seed = 42
    cols_to_keep = 32
    columns = ['ci_skipped', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev',
           'age', 'nuc', 'exp', 'rexp', 'sexp', 'TFC', 'is_doc', 'is_build',
           'is_meta', 'is_media', 'is_src', 'is_merge', 'FRM', 'COM', 'CFT',
           'classif', 'prev_com_res', 'proj_recent_skip', 'comm_recent_skip',
           'same_committer', 'is_fix', 'day_week', 'CM', 'commit_hash']


    # Initializing training instance
    training = RLdecisionTreeTrain(HIDDEN_SIZE, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR, LR_CRITIC, GAMMA, max_depth, use_meth_1, nbr_of_conv, n_episodes, curdir, seed, columns, cols_to_keep, save_every)

    # Starting training
    logger.info("starting process for job {}".format(job_num))
    if eval_method=="within":
        training.within_eval(valid_proj)
    else:
        training.cross_eval(valid_proj)