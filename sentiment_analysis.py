#!/usr/bin/env python
"""
Reproduce the sentiment analysis experiment in [Zellinger2017a]

The central moment discrepancy (CMD) is used for domain adaptation
as first described in the preliminary conference paper [Zellinger2017b].
It is implemented as keras regularizer that can be used by shared layers.
This implementation is tested under keras 1.1.0.

[Zellinger2017a] W. Zellinger, B.A. Moser, T. Grubinger, E. Lughofer,
T. Natschlaeger, and S. Saminger-Platz, "Robust unsupervised domain adaptation
for neural networks via moment alignment," arXiv preprint arXiv:1711.06114, 2017
[Zellinger2017b] W.Zellinger, T. Grubinger, E. Lughofer, T. Ntschlaeger,
and Susanne Saminger-Platz, "Central moment discrepancy (cmd) for
domain-invariant representation learning," International Conference on Learning
Representations (ICLR), 2017

__author__ = "Werner Zellinger"
__copyright__ = "Copyright 2017, Werner Zellinger"
__credits__ = ["Thomas Grubinger, Robert Pollak"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Werner Zellinger"
__email__ = "werner.zellinger@jku.at"
"""

from __future__ import print_function

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist

# moment alignment neural network algorithm of paper based on CMD
from models.mann_sentiment_analysis import MANN
from models.central_moment_discrepancy import CMDRegularizer
from models.correlation_alignment import CORALRegularizer
from models.maximum_mean_discrepancy import MMDRegularizer

N_FEATURES = 5000
N_TR_SAMPLES = 2000
TEMP_FOLDER = 'temp/sentiment_analysis/'
AMAZON_DATA_FILE = 'data/amazon_reviews_dataset/amazon.mat'
INIT_WEIGHTS_FILE = 'init_weights'
OUTPUT_FOLDER = 'output/object_recognition/'
# network architecture
N_HIDDEN_UNITS = 50
MAX_N_EPOCH = 1500
BATCH_SIZE = 300
N_REPETITIONS = 5

def shuffle(x, y):
    """
    shuffle data (used by split)
    """
    index_shuf = np.arange(x.shape[0])
    np.random.shuffle(index_shuf)
    x=x[index_shuf,:]
    y=y[index_shuf]
    return x,y

def split_data(d_s_ind,d_t_ind,x,y,offset,n_tr_samples,r_seed=0):
    """
    split data (train/validation/test, source/target)
    """
    np.random.seed(r_seed)
    x_s_tr = x[offset[d_s_ind,0]:offset[d_s_ind,0]+n_tr_samples,:]
    x_t_tr = x[offset[d_t_ind,0]:offset[d_t_ind,0]+n_tr_samples,:]
    x_s_tst = x[offset[d_s_ind,0]+n_tr_samples:offset[d_s_ind+1,0],:]
    x_t_tst = x[offset[d_t_ind,0]+n_tr_samples:offset[d_t_ind+1,0],:]
    y_s_tr = y[offset[d_s_ind,0]:offset[d_s_ind,0]+n_tr_samples]
    y_t_tr = y[offset[d_t_ind,0]:offset[d_t_ind,0]+n_tr_samples]
    y_s_tst = y[offset[d_s_ind,0]+n_tr_samples:offset[d_s_ind+1,0]]
    y_t_tst = y[offset[d_t_ind,0]+n_tr_samples:offset[d_t_ind+1,0]]
    x_s_tr,y_s_tr=shuffle(x_s_tr,y_s_tr)
    x_t_tr,y_t_tr=shuffle(x_t_tr,y_t_tr)
    x_s_tst,y_s_tst=shuffle(x_s_tst,y_s_tst)
    x_t_tst,y_t_tst=shuffle(x_t_tst,y_t_tst)
    y_s_tr[y_s_tr==-1]=0
    y_t_tr[y_t_tr==-1]=0
    y_s_tst[y_s_tst==-1]=0
    y_t_tst[y_t_tst==-1]=0
    return x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst
    
def load_dataset(n_features, filename):
    """
    Load amazon reviews
    """
    mat = loadmat(filename)
    xx=mat['xx']
    yy=mat['yy']
    offset=mat['offset']
    x=xx[:n_features,:].toarray().T#n_samples X n_features
    y=yy.ravel()
    return x, y, offset

def accuracy(y, y_true):
    """
    amount of right classified reviews
    """
    return 1-np.sum(np.abs(np.round(y).ravel()-y_true.ravel()))/y.shape[0]

def reverse_validation(model, init_weights, S, T, name='nn'):
    """
    reverse validation
    
    - Zhong, Erheng, et al. "Cross validation framework to choose amongst
    models and datasets for transfer learning.", Joint European Conference on
    Machine Learning and Knowledge Discovery in Databases. Springer Berlin
    Heidelberg, 2010.
    """
    train_perc=0.8
    x_tr_s=S[0][:int(S[0].shape[0]*train_perc),:]
    y_tr_s=S[1][:int(S[1].shape[0]*train_perc)]
    x_val_s=S[0][int(S[0].shape[0]*train_perc):,:]
    y_val_s=S[1][int(S[1].shape[0]*train_perc):]
    x_tr_t=T[:int(T.shape[0]*train_perc),:]
    x_val_t=T[int(T.shape[0]*train_perc):,:]
    # Train model \nu
    model.fit(x_tr_s, y_tr_s, x_tr_t, val_set=(x_val_s, y_val_s), init_weights=init_weights)
    # Save the weights as init for next turn
    model.save('tmp_weights_rv')
    # Predict target labels
    y_pred_t = model.predict(x_tr_t)
    y_pred_val_t = model.predict(x_val_t)
    # Learn reverse classifier \nu_r (load weights)
    # Init with first fitted weights, procedere taken from
    # Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks.",
    # arXiv preprint arXiv:1505.07818 (2015).
    model.fit(x_tr_t, y_pred_t, x_tr_s, val_set=(x_val_t,y_pred_val_t),
              init_weights='tmp_weights_rv')
    # Evaluate reverse classifier
    y_pred_s=model.predict(x_val_s)
    # Calculate accuracy
    acc = accuracy(y_pred_s,y_val_s)
    # Return reverse validation risk
    return acc


print("\nLoading amazon review data...")
x, y, offset = load_dataset(N_FEATURES,AMAZON_DATA_FILE)
domains=['books','dvd','electronics','kitchen']


print("\nTraining four test models (NN, NNcoral, NNmmd, MANN)...")
# Split data for domains books->kitchen
x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                             = split_data(1, 2, x, y, offset, N_TR_SAMPLES,0)
# NN
nn = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS, folder=TEMP_FOLDER,
          n_epochs=MAX_N_EPOCH, bsize=BATCH_SIZE, save_weights='nn')
nn.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
print('NN: '+str(accuracy(y_t_tst,nn.predict(x_t_tst))))
# NN with Coral domain adaptation
# use moment alignment neural network (MANN) algorithm with coral function
# - S. Baochen, and K. Saenko. "Deep coral: Correlation alignment for deep
#   domain adaptation," Computer Vision--ECCV 2016 Workshops. Springer
#   International Publishing, 2016.
coral = CORALRegularizer(1.)
nn_coral = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS, folder=TEMP_FOLDER,
                n_epochs=MAX_N_EPOCH, bsize=BATCH_SIZE, save_weights='coral',
                activity_regularizer=coral)
nn_coral.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0, init_weights='nn')
print('CORAL: '+str(accuracy(y_t_tst,nn_coral.predict(x_t_tst))))
# NN with MMD domain adaptation
# - M. Long, et al. "Learning transferable features with deep adaptation
#   networks." International Conference on Machine Learning. 2015.
# compute Gaussian kernel beta based on heuristic
a_s,a_t = nn.get_activations(x_s_tr,x_t_tr)
a = np.concatenate((a_s,a_t),axis=0)
b = 1.0/np.median(pdist(a))
# train with mmd
mmd = MMDRegularizer(1.,beta=b)
nn_mmd = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS, folder=TEMP_FOLDER,
              n_epochs=MAX_N_EPOCH, bsize=BATCH_SIZE, save_weights='mmd',
              activity_regularizer=mmd)
nn_mmd.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0, init_weights='nn')
print('MMD: '+str(accuracy(y_t_tst,nn_mmd.predict(x_t_tst))))
# our approach
# moment alignment neural network: NN with CMD domain adaptation
cmd = CMDRegularizer(1.)
mann = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS, folder=TEMP_FOLDER,
            n_epochs=MAX_N_EPOCH, bsize=BATCH_SIZE, save_weights='cmd',
            activity_regularizer=cmd)
mann.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0, init_weights='nn')
print('CMD (ours): '+str(accuracy(y_t_tst,mann.predict(x_t_tst))))


print("\nRunning evaluation...")
accs_big_eval = np.zeros((12,4,N_REPETITIONS))
for random_seed in range(N_REPETITIONS):
    # Create different initial weights by setting RANDOM_SEED for split_data
    # function before the weights creation.
    # This is important for every model to start at same initial situation.
    x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                             = split_data(0, 1, x, y, offset, N_TR_SAMPLES,
                                          random_seed)
    nn = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS,
              folder=TEMP_FOLDER, n_epochs=MAX_N_EPOCH, bsize=BATCH_SIZE)
    weights = INIT_WEIGHTS_FILE+'_'+str(random_seed)
    nn.create_initial_weights(x_s_tr, y_s_tr, x_t_tr, weights)
    
    # init
    accuracies_nn = np.zeros((4,4))
    accuracies_coral = np.zeros((4,4))
    accuracies_mmd = np.zeros((4,4))
    accuracies_cmd = np.zeros((4,4))
    settings_mmd = np.chararray((4,4),itemsize=100)
    settings_mmd[:] = 'empty'
    
    # for all possible tasks in dataset
    for d_s_ind,dom_s in enumerate(domains):
        for d_t_ind,dom_t in enumerate(domains):
            if dom_s==dom_t:
                continue            
            
            print('\nsource domain: '+str(dom_s))
            print('target domain: '+str(dom_t))
            
            # split data
            x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst = \
            split_data(d_s_ind,d_t_ind,x,y,offset,N_TR_SAMPLES)
            data = [x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst]
            
            # NN without domain adaptation objective
            nn = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS,
                      folder=TEMP_FOLDER, n_epochs=MAX_N_EPOCH,
                      bsize=BATCH_SIZE, save_weights='nn')
            nn.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0,
                   init_weights=weights)
            accuracies_nn[d_s_ind,d_t_ind] = accuracy(y_t_tst,
                                                      nn.predict(x_t_tst))
            print('NN: '+str(accuracies_nn[d_s_ind,d_t_ind]))
            
            # NN with CORAL for domain adaptation
            coral = CORALRegularizer(l=1.)# argumentation for l=1 in paper
            nn_coral = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS,
                            folder=TEMP_FOLDER, n_epochs=MAX_N_EPOCH,
                            bsize=BATCH_SIZE, activity_regularizer=coral,
                            save_weights='coral')
            nn_coral.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0,
                         init_weights=weights)
            accuracies_coral[d_s_ind,d_t_ind] = accuracy(y_t_tst,
                                                         nn_coral.predict(x_t_tst))
            print('CORAL: '+str(accuracies_coral[d_s_ind,d_t_ind]))
            
            # moment alignment neural networks (ours)
            coral = CMDRegularizer(l=1.)# argumentation for l=1 in paper
            nn_cmd = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS,
                            folder=TEMP_FOLDER, n_epochs=MAX_N_EPOCH,
                            bsize=BATCH_SIZE, activity_regularizer=coral,
                            save_weights='cmd')
            nn_cmd.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0, init_weights=weights)
            accuracies_cmd[d_s_ind,d_t_ind] = accuracy(y_t_tst,
                                                       nn_cmd.predict(x_t_tst))
            print('CMD: '+str(accuracies_cmd[d_s_ind,d_t_ind]))
            
            # NN with MMD for domain adaptation
            # parameter selection for mmd via reverse validation (unsupervised
            # hyperparameter selection for transfer learning algorithms)
            drws = np.logspace(np.log10(0.1),np.log10(500),
                               num=10).tolist()# domain regularizer weights
            betas= np.logspace(np.log10(0.01),np.log10(10),
                               num=10).tolist()# gaussian kernel betas
            accs_mmd = np.zeros((len(drws),len(betas)))            
            for i,weight in enumerate(drws):
                for j,beta in enumerate(betas):
                    # info
                    print('mmd '+dom_s+'->'+dom_t+' step: '+str(len(betas)*i+j+1)
                    +'/'+str(len(drws)*len(betas))+' dr-weight: '+str(weight)
                    +' ('+str(i+1)+'/'+str(len(drws))+') beta: '+str(beta)
                    +' ('+str(j+1)+'/'+str(len(betas))+')')
                    mmd = MMDRegularizer(l=weight, beta=beta)
                    nn_mmd = MANN(n_features=N_FEATURES,
                                  n_hiddens=N_HIDDEN_UNITS,
                                  folder=TEMP_FOLDER,
                                  n_epochs=MAX_N_EPOCH,
                                  bsize=BATCH_SIZE,
                                  activity_regularizer=mmd,
                                  save_weights='mmd')
                    S=(x_s_tr,y_s_tr)
                    T=x_t_tr
                    # grid search for mmd via reverse validation (see paper)
                    accs_mmd[i,j] = reverse_validation(nn_mmd, weights, S, T)
            # Find best mmd setting
            [i,j]=np.unravel_index(accs_mmd.argmax(),accs_mmd.shape)
            # Train best model
            mmd = MMDRegularizer(l=drws[i], beta=betas[j])
            nn_mmd = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS,
                        folder=TEMP_FOLDER, n_epochs=MAX_N_EPOCH,
                        bsize=BATCH_SIZE, activity_regularizer=mmd,
                        save_weights='mmd_final')
            nn_mmd.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0, init_weights=weights)
            accuracies_mmd[d_s_ind,d_t_ind] = accuracy(y_t_tst,
                                                       nn_mmd.predict(x_t_tst))
            settings_mmd[d_s_ind,d_t_ind] = 'domadapt-weight: ' + str(drws[i])\
                                            + 'beta: ' + str(betas[j])
            print('MMD: '+str(accuracies_mmd[d_s_ind,d_t_ind]))


    # report results
    table = np.zeros((16,4))
    table[:,0] = accuracies_nn.ravel()
    table[:,1] = accuracies_coral.ravel()
    table[:,2] = accuracies_mmd.ravel()
    table[:,3] = accuracies_cmd.ravel()
    table = np.delete(table,[0,5,10,15],0)# delete entries for b->b, k->k, etc.
    accs_big_eval[:,:,random_seed] = table
    np.save(OUTPUT_FOLDER+'big_eval_accs.npy',accs_big_eval)
    

print("\nresults...")
means = accs_big_eval.mean(axis=2)
stds = accs_big_eval.std(axis=2)
print('b->d')
print('NN: acc-tst=    '+str(means[0,0])+'+-'+str(stds[0,0]))
print('CORAL: acc-tst= '+str(means[0,1])+'+-'+str(stds[0,1]))
print('MMD: acc-tst=   '+str(means[0,2])+'+-'+str(stds[0,2]))
print('CMD: acc-tst=   '+str(means[0,3])+'+-'+str(stds[0,3]))
print('b->e')
print('NN: acc-tst=    '+str(means[1,0])+'+-'+str(stds[1,0]))
print('CORAL: acc-tst= '+str(means[1,1])+'+-'+str(stds[1,1]))
print('MMD: acc-tst=   '+str(means[1,2])+'+-'+str(stds[1,2]))
print('CMD: acc-tst=   '+str(means[1,3])+'+-'+str(stds[1,3]))
print('b->k')
print('NN: acc-tst=    '+str(means[2,0])+'+-'+str(stds[2,0]))
print('CORAL: acc-tst= '+str(means[2,1])+'+-'+str(stds[2,1]))
print('MMD: acc-tst=   '+str(means[2,2])+'+-'+str(stds[2,2]))
print('CMD: acc-tst=   '+str(means[2,3])+'+-'+str(stds[2,3]))
print('d->b')
print('NN: acc-tst=    '+str(means[3,0])+'+-'+str(stds[3,0]))
print('CORAL: acc-tst= '+str(means[3,1])+'+-'+str(stds[3,1]))
print('MMD: acc-tst=   '+str(means[3,2])+'+-'+str(stds[3,2]))
print('CMD: acc-tst=   '+str(means[3,3])+'+-'+str(stds[3,3]))
print('d->e')
print('NN: acc-tst=    '+str(means[4,0])+'+-'+str(stds[4,0]))
print('CORAL: acc-tst= '+str(means[4,1])+'+-'+str(stds[4,1]))
print('MMD: acc-tst=   '+str(means[4,2])+'+-'+str(stds[4,2]))
print('CMD: acc-tst=   '+str(means[4,3])+'+-'+str(stds[4,3]))
print('d->k')
print('NN: acc-tst=    '+str(means[5,0])+'+-'+str(stds[5,0]))
print('CORAL: acc-tst= '+str(means[5,1])+'+-'+str(stds[5,1]))
print('MMD: acc-tst=   '+str(means[5,2])+'+-'+str(stds[5,2]))
print('CMD: acc-tst=   '+str(means[5,3])+'+-'+str(stds[5,3]))
print('e->b')
print('NN: acc-tst=    '+str(means[6,0])+'+-'+str(stds[6,0]))
print('CORAL: acc-tst= '+str(means[6,1])+'+-'+str(stds[6,1]))
print('MMD: acc-tst=   '+str(means[6,2])+'+-'+str(stds[6,2]))
print('CMD: acc-tst=   '+str(means[6,3])+'+-'+str(stds[6,3]))
print('e->d')
print('NN: acc-tst=    '+str(means[7,0])+'+-'+str(stds[7,0]))
print('CORAL: acc-tst= '+str(means[7,1])+'+-'+str(stds[7,1]))
print('MMD: acc-tst=   '+str(means[7,2])+'+-'+str(stds[7,2]))
print('CMD: acc-tst=   '+str(means[7,3])+'+-'+str(stds[7,3]))
print('e->k')
print('NN: acc-tst=    '+str(means[8,0])+'+-'+str(stds[8,0]))
print('CORAL: acc-tst= '+str(means[8,1])+'+-'+str(stds[8,1]))
print('MMD: acc-tst=   '+str(means[8,2])+'+-'+str(stds[8,2]))
print('CMD: acc-tst=   '+str(means[8,3])+'+-'+str(stds[8,3]))
print('k->b')
print('NN: acc-tst=    '+str(means[9,0])+'+-'+str(stds[9,0]))
print('CORAL: acc-tst= '+str(means[9,1])+'+-'+str(stds[9,1]))
print('MMD: acc-tst=   '+str(means[9,2])+'+-'+str(stds[9,2]))
print('CMD: acc-tst=   '+str(means[9,3])+'+-'+str(stds[9,3]))
print('k->d')
print('NN: acc-tst=    '+str(means[10,0])+'+-'+str(stds[10,0]))
print('CORAL: acc-tst= '+str(means[10,1])+'+-'+str(stds[10,1]))
print('MMD: acc-tst=   '+str(means[10,2])+'+-'+str(stds[10,2]))
print('CMD: acc-tst=   '+str(means[10,3])+'+-'+str(stds[10,3]))
print('k->e')
print('NN: acc-tst=    '+str(means[11,0])+'+-'+str(stds[11,0]))
print('CORAL: acc-tst= '+str(means[11,1])+'+-'+str(stds[11,1]))
print('MMD: acc-tst=   '+str(means[11,2])+'+-'+str(stds[11,2]))
print('CMD: acc-tst=   '+str(means[11,3])+'+-'+str(stds[11,3]))
