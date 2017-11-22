#!/usr/bin/env python
"""
Reproduce the artificial example described in the paper [Zellinger2017a]

The central moment discrepancy (CMD) is used for domain adaptation
as first described in the preliminary conference paper [Zellinger2017b].
It is implemented as keras objective function.
This implementation uses keras 1.1.0.

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

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(0)
from scipy import stats
from keras.layers import Dense, Input, merge
from keras.models import Model
from keras.optimizers import Adadelta
from keras import backend as K

plt.close('all')

N_HIDDEN_NODES = 15
N_MOMENTS = 5
N_CLASSES = 3
DATA_FOLDER = 'data/artificial_dataset/'
TMP_FOLDER = 'temp/artificial_example/'
OUTPUT_FOLDER = 'output/artificial_example/'
    

def cmd(labels, y_pred):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    
    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = y_pred[:,:N_HIDDEN_NODES]
    x2 = y_pred[:,N_HIDDEN_NODES:]
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = dm
    for i in range(N_MOMENTS-1):
        # moment diff of centralized samples
        scms+=moment_diff(sx1,sx2,i+2)
    return scms

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return ((x1-x2)**2).sum().sqrt()      

def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1**K.cast(k,'int32')).mean(0)
    ss2 = (sx2**K.cast(k,'int32')).mean(0)
    return l2diff(ss1,ss2)

def neural_network(domain_adaptation=False):
    """
    moment alignment neural network (MANN)
    
    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", arXiv preprint arXiv:1711.06114, 2017
    """
    # layer definition
    input_s = Input(shape=(2,), name='souce_input')
    input_t = Input(shape=(2,), name='target_input')
    encoding = Dense(N_HIDDEN_NODES,
                     activation='sigmoid',
                     name='hidden')
    prediction = Dense(N_CLASSES,
                       activation='softmax',
                       name='pred')
    # network architecture
    encoded_s = encoding(input_s)
    encoded_t = encoding(input_t)
    pred_s = prediction(encoded_s)
    pred_t = prediction(encoded_t)
    dense_s_t = merge([encoded_s,encoded_t], mode='concat', concat_axis=1)
    # input/output definition
    nn = Model(input=[input_s,input_t],
               output=[pred_s,pred_t,dense_s_t])
    # seperate model for activation visualization
    visualize_model = Model(input=[input_s,input_t],
                            output=[encoded_s,encoded_t])
    # compile model
    if domain_adaptation==False:
        cmd_weight = 0.
    else:
        # Please note that the loss weight of the cmd is one per default
        # (see paper).
        cmd_weight = 1.
    nn.compile(loss=['categorical_crossentropy',
                     'categorical_crossentropy',cmd],
               loss_weights=[1.,0.,cmd_weight],
               optimizer=Adadelta(),
               metrics=['accuracy'])
    return nn, visualize_model

def plot_classification_boarders(nn,save_name):
    """
    plot dataset and classification boarders
    """
    plt.figure()
    plt.plot(x_s[y_s[:,0]==1,0],x_s[y_s[:,0]==1,1],color='k',marker=r'$+$',
             linestyle='',ms=15)
    plt.plot(x_s[y_s[:,1]==1,0],x_s[y_s[:,1]==1,1],color='k',marker=r'$-$',
             linestyle='',ms=15)
    plt.plot(x_s[y_s[:,2]==1,0],x_s[y_s[:,2]==1,1],color='k',marker='*',
             linestyle='',ms=15)
    plt.plot(x_t[:,0],x_t[:,1],'k.')
    x_min = -1
    y_min = -0.75
    x_max = 1.2
    y_max = 1.3
    xy = np.mgrid[x_min:x_max:0.001, y_min:y_max:0.001].reshape(2,-1).T
    z = nn.predict([xy,xy])[0]
    ind = np.argmax(z,axis=1)
    z[ind!=0,0]=0
    z[ind!=1,1]=0
    x,y = np.mgrid[x_min:x_max:0.001, y_min:y_max:0.001]
    plt.contour(x,y,z[:,0].reshape(x.shape),levels = [0.1],
                colors=('k',),linestyles=('-',),linewidths=(2,))
    plt.contour(x,y,z[:,1].reshape(x.shape),levels = [0],
                colors=('k',),linestyles=('-',),linewidths=(2,))
    plt.axis('off')
    plt.savefig(save_name)
    
def plot_activations(a_s,a_t,save_name):
    """
    activation visualization via seaborn library
    """
    n_dim=a_s.shape[1]
    n_rows=1
    n_cols=int(n_dim/n_rows)
    fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols, sharey=True,
                            sharex=True)
    for k,ax in enumerate(axs.reshape(-1)):
        if k>=n_dim:
            continue
        sns.kdeplot(a_t[:,k],ax=ax, shade=True, label='target',
                    legend=False, color='0.4',bw=0.03)
        sns.kdeplot(a_s[:,k],ax=ax, shade=True, label='source',
                    legend=False, color='0',bw=0.03)
        plt.setp(ax.xaxis.get_ticklabels(),fontsize=10)
        plt.setp(ax.yaxis.get_ticklabels(),fontsize=10)
    fig.set_figheight(3)
    plt.setp(axs, xticks=[0, 0.5, 1])
    plt.setp(axs, ylim=[0,10])
    plt.savefig(save_name)


# load dataset
x_s = np.load(DATA_FOLDER+'x_s.npy')
y_s = np.load(DATA_FOLDER+'y_s.npy')
x_t = np.load(DATA_FOLDER+'x_t.npy')
y_t = np.load(DATA_FOLDER+'y_t.npy')


# plot dataset
plt.plot(x_s[y_s[:,0]==1,0],x_s[y_s[:,0]==1,1],color='k',marker=r'$+$',
         linestyle='',ms=15)
plt.plot(x_s[y_s[:,1]==1,0],x_s[y_s[:,1]==1,1],color='k',marker=r'$-$',
         linestyle='',ms=15)
plt.plot(x_s[y_s[:,2]==1,0],x_s[y_s[:,2]==1,1],color='k',marker='*',
         linestyle='',ms=15)
plt.plot(x_t[:,0],x_t[:,1],'k.')
plt.axis('off')
plt.savefig(OUTPUT_FOLDER+'dataset.jpg')


# train source model without domain adaptation
nn, nn_vis_model = neural_network(domain_adaptation=False)
nn.fit(x=[x_s,x_t],
       y=[y_s,y_t,np.zeros((x_s.shape[0],1))],
       shuffle=True,
       nb_epoch=10000,
       verbose=0,
       batch_size=x_s.shape[0])
# save the weights
nn.save_weights(TMP_FOLDER+'nn.hdf5')
# train another 5000 epochs for fair comparison
np.random.seed(0)
nn.fit(x=[x_s,x_t],
       y=[y_s,y_t,np.zeros((x_s.shape[0],1))],
       shuffle=True,
       nb_epoch=5000,
       verbose=0,
       batch_size=x_s.shape[0])
# plot the classification boarders (Fig. 3 left in paper)
plot_classification_boarders(nn, OUTPUT_FOLDER+'nn.jpg')
# predict the target accuracy
# The final source accuracy of the NN should be 100% and the final target
# accuracy should be around 89% depending on your system random numbers, theano
# configuration (float32,..), CuDNN version, etc.
print('\nNN acc='+str(nn.evaluate([x_s,x_t],[y_s,y_t,y_t])[-2]))


# adapt the network to to the target domain by means of the central
# moment discrepancy
mann, mann_vis_model = neural_network(domain_adaptation=True)
mann.load_weights(TMP_FOLDER+'nn.hdf5')
np.random.seed(0)
mann.fit(x=[x_s,x_t],
         y=[y_s,y_t,np.zeros((x_s.shape[0],1))],
         shuffle=True,
         nb_epoch=5000,
         verbose=0,
         batch_size=x_s.shape[0])
# plot the classification boarders (Fig. 3 right in paper)
plot_classification_boarders(mann, OUTPUT_FOLDER+'mann.jpg')
# predict the target accuracy
# The predicted acccuracy of the MANN is around 10% more than the
# accuracy of the NN before. This is the result of our approach and does not
# strongly depend on the random numbers.
print('\nMANN acc='+str(mann.evaluate([x_s,x_t],[y_s,y_t,y_t])[-2]))


# Plot activations of NN (Fig. 4 top)
a_s,a_t = nn_vis_model.predict([x_s,x_t])
# Find five most significantly (K-S test) different distributions
p_vals = np.zeros(a_s.shape[1])
for i in range(a_s.shape[1]):
    ksstat, p_vals[i] = stats.ks_2samp(a_s[:,i],a_t[:,i])
ind_worst = p_vals.argsort()[:5]
plot_activations(a_s[:,ind_worst],a_t[:,ind_worst],
                 OUTPUT_FOLDER+'activations_nn.jpg')


# Plot activations of MANN (Fig. 4 bottom)
a_s,a_t = mann_vis_model.predict([x_s,x_t])
# Find five most significantly (K-S test) different distributions
p_vals = np.zeros(a_s.shape[1])
for i in range(a_s.shape[1]):
    ksstat, p_vals[i] = stats.ks_2samp(a_s[:,i],a_t[:,i])
ind_worst = p_vals.argsort()[:5]
plot_activations(a_s[:,ind_worst],a_t[:,ind_worst],
                 OUTPUT_FOLDER+'activations_mann.jpg')

