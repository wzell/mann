#!/usr/bin/env python
"""
Correlation Alignment (CORAL)

The CORAL is implemented as keras regularizer that can be used for
shared layers. This implementation uis tested under keras 1.1.0.

- S. Baochen, and K. Saenko. "Deep coral: Correlation alignment for deep
domain adaptation," Computer Vision--ECCV 2016 Workshops. Springer International
Publishing, 2016.

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
    
def coral(x1, x2):
    """
    correlation alignment objective (CORAL)
    objective function for keras models (theano or tensorflow backend)
    
    - S. Baochen, and K. Saenko. "Deep coral: Correlation alignment for deep
    domain adaptation," Computer Vision--ECCV 2016 Workshops. Springer
    International Publishing, 2016.
    """
    c1 = 1./(x1.shape[0]-1)*(K.dot(K.transpose(x1),x1)-
             K.dot(K.transpose(x1.mean(axis=0)),x1.sum(axis=0)))
    c2 = 1./(x2.shape[0]-1)*(K.dot(K.transpose(x2),x2)-
             K.dot(K.transpose(x2.mean(axis=0)),x2.sum(axis=0)))
    return 1./(4*x1.shape[0]**2)*((c1-c2)**2).sum()

class CORALRegularizer(Regularizer):
    """
    class structure to use the CORAL as activity regularizer of a
    keras shared layer
    """
    def __init__(self,l=1):
        self.uses_learning_phase = 1
        self.l=l

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
            sim = coral(self.layer.get_output_at(0),
                        self.layer.get_output_at(1))
        add_loss = K.switch(K.equal(len(self.layer.inbound_nodes),2),sim,0)
        regularizer_loss += self.l*add_loss
        return K.in_train_phase(regularizer_loss, loss)

    def get_config(self):
        # needed for keras layer
        return {'name': self.__class__.__name__,
                'l': float(self.l)}
        