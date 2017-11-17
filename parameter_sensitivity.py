#!/usr/bin/env python
"""
Reproduce the parameter sensitivity experiment in [Zellinger2017a]

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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# moment alignment neural network algorithm of paper based on CMD
from models.mann_sentiment_analysis import MANN
from models.central_moment_discrepancy import CMDRegularizer
from models.maximum_mean_discrepancy import MMDRegularizer

N_FEATURES = 5000
N_TR_SAMPLES = 2000
TEMP_FOLDER = 'temp/parameter_sensitivity/'
AMAZON_DATA_FILE = 'data/amazon_reviews_dataset/amazon.mat'
INIT_WEIGHTS_FILE = 'init_weights'
OUTPUT_FOLDER = 'output/parameter_sensitivity/'
# network architecture
N_HIDDEN_UNITS = 50
MAX_N_EPOCH = 1500
BATCH_SIZE = 300


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

def accuracy(y, y_true):
    """
    amount of right classified reviews
    """
    return 1-np.sum(np.abs(np.round(y).ravel()-y_true.ravel()))/y.shape[0]


print("\nLoading amazon review data...")
x, y, offset = load_dataset(N_FEATURES,AMAZON_DATA_FILE)
domains=['books','dvd','electronics','kitchen']


drws_cmd = np.linspace(0,3,11)[1:]
n_moments = np.array([1,2,3,4,5,6,7])
accs_cmd = np.zeros((12,drws_cmd.shape[0],n_moments.shape[0]))
task_count = 0
for d_s in range(4):
    for d_t in range(4):
        if d_s==d_t:
            continue
        x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                                     = split_data(d_s, d_t, x, y, offset,
                                                  N_TR_SAMPLES)
        for i,drw in enumerate(drws_cmd):
            for j,k in enumerate(n_moments):       
                cmd = CMDRegularizer(l=drw, n_moments=k)
                mann = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS,
                            folder=TEMP_FOLDER, n_epochs=MAX_N_EPOCH,
                            bsize=BATCH_SIZE, activity_regularizer=cmd,
                            save_weights='cmd')
                mann.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
                acc = accuracy(y_t_tst,mann.predict(x_t_tst))
                print('cmd task='+str(task_count)+' lambda='+str(drw)+
                      ' n_moments='+str(k)+' : '+str(acc))
                accs_cmd[task_count,i,j]=acc
        task_count+=1
np.save(TEMP_FOLDER+'sensitivity_cmd.npy',accs_cmd)
#
#
drws_mmd = np.linspace(0,100,21)[1:10]
betas= np.linspace(0.3,1.7,7)
tasks = np.arange(0,12)
accs_mmd = np.zeros((12,drws_mmd.shape[0], betas.shape[0]))
task_count = 0
for d_s in range(4):
    for d_t in range(4):
        if d_s==d_t:
            continue
        x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                                     = split_data(d_s, d_t, x, y, offset,
                                                  N_TR_SAMPLES)
        for i,drw in enumerate(drws_mmd):
            for j,beta in enumerate(betas):       
                mmd = MMDRegularizer(l=drw, beta=beta)
                nn_mmd = MANN(n_features=N_FEATURES, n_hiddens=N_HIDDEN_UNITS,
                              folder=TEMP_FOLDER, n_epochs=MAX_N_EPOCH,
                              bsize=BATCH_SIZE, activity_regularizer=mmd,
                              save_weights='mmd')
                nn_mmd.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
                acc = accuracy(y_t_tst,nn_mmd.predict(x_t_tst))
                print('mmd '+str(task_count)+' lambda='+str(drw)+' beta='+
                      str(beta)+' : '+str(acc))
                accs_mmd[task_count,i,j]=acc
        task_count+=1
np.save(TEMP_FOLDER+'sensitivity_mmd.npy',accs_mmd)


accs_cmd = np.load(TEMP_FOLDER+'sensitivity_cmd.npy')
accs_mmd = np.load(TEMP_FOLDER+'sensitivity_mmd.npy')

base_cmd = np.argmax(n_moments==5)
base_mmd = accs_mmd.mean(1).mean(0).argmax()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.gcf().subplots_adjust(bottom=0.15)
for i in range(12):
    ax1.plot(n_moments,accs_cmd.mean(1)[i,:]/accs_cmd.mean(1)[i,base_cmd])
    ax2.plot(range(1,betas.shape[0]+1),
             accs_mmd.mean(1)[i,:]/accs_mmd.mean(1)[i,base_mmd])
ax1.plot(n_moments, accs_cmd.mean(1).mean(0)/accs_cmd.mean(1).mean(0)[base_cmd],
         'k--', linewidth=4)
ax2.plot(range(1,betas.shape[0]+1),
         accs_mmd.mean(1).mean(0)/accs_mmd.mean(1).mean(0)[base_mmd],
         'k--',linewidth=4)
ax1.grid(True)
ax2.grid(True)
ax1.set_title('CMD')
ax2.set_title('MMD')
f.set_figheight(3)
plt.sca(ax1)
plt.xticks(range(1,len(n_moments)+1),n_moments)
plt.xlabel('number of moments', fontsize=12)
plt.ylabel('accuracy ratio', fontsize=12)
plt.sca(ax2)
plt.xticks(range(1,betas.shape[0]+1),betas.round(1))
plt.xlabel('kernel parameter', fontsize=12)
ax1.set_ylim([0.97,1.01])
plt.savefig(OUTPUT_FOLDER+'parameter_sensitivity.jpg',dpi=500)

