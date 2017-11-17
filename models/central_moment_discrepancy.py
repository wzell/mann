#!/usr/bin/env python
"""
Central Moment Discrepancy (CMD)

The CMD is used for domain adaptation as first described in the conference
paper [Zellinger2017b] and further discussed in the journal version
[Zellinger2017a].
The CMD is implemented as keras regularizer that can be used for shared layers.
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


from keras import backend as K
from keras.regularizers import Regularizer

def cmd(x1, x2, n_moments=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    
    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = dm
    for i in range(n_moments-1):
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

class CMDRegularizer(Regularizer):
    """
    class structure to use the CMD as activity regularizer of a
    keras shared layer
    """
    def __init__(self,l=1.,n_moments=5):
        self.uses_learning_phase = 1
        self.l=l
        self.n_moments = n_moments

    def set_layer(self, layer):
        # needed for keras layer
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularizer_loss = loss
        sim = 0
        if len(self.layer.inbound_nodes)>1:
            # we are in a shared keras layer
            sim = cmd(self.layer.get_output_at(0),
                      self.layer.get_output_at(1),
                      self.n_moments)
        add_loss = K.switch(K.equal(len(self.layer.inbound_nodes),2),sim,0)
        regularizer_loss += self.l*add_loss
        return K.in_train_phase(regularizer_loss, loss)

    def get_config(self):
        # needed for keras layer
        return {'name': self.__class__.__name__,
                'l': float(self.l)}
        