#!/usr/bin/env python
"""
Moment alignment neural network (MANN) for sentiment analysis

W. Zellinger, B.A. Moser, T. Grubinger, E. Lughofer,
T. Natschlaeger, and S. Saminger-Platz, "Robust unsupervised domain adaptation
for neural networks via moment alignment," arXiv preprint arXiv:1711.06114, 2017

__author__ = "Werner Zellinger"
__copyright__ = "Copyright 2017, Werner Zellinger"
__credits__ = ["Thomas Grubinger, Robert Pollak"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Werner Zellinger"
__email__ = "werner.zellinger@jku.at"
"""

import numpy as np
import datetime
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, SGD
from keras.layers import Dense, Input
from keras.models import Model

class MANN:
    """
    class structure for moment alignment neural networks
    """
    def __init__(self,
                 n_features=5000,
                 n_hiddens=50,
                 folder='temp/sentiment_analysis/',
                 n_epochs=1500,
                 bsize=300,
                 activity_regularizer=None,
                 save_weights='tmp_weights'):
        self.n_features = n_features
        self.nn = None
        self.n_epochs = n_epochs
        self.batch_size = bsize
        self.n_hiddens = n_hiddens
        self.activity_regularizer = activity_regularizer
        self.tmp_folder = folder
        self.save_weights = save_weights+'.hdf5'
        self.visualize_model = None
        
    def create(self):
        """
        create two layer classifier
        as in Algorithm 1 of the paper
        """
        # input
        input_s = Input(shape=(self.n_features,), name='souce_input')
        input_t = Input(shape=(self.n_features,), name='target_input')
        # layers
        if self.activity_regularizer:
            encoding = Dense(self.n_hiddens,
                             activation='sigmoid',
                             name='encoded',
                             activity_regularizer=self.activity_regularizer)
        else:
            encoding = Dense(self.n_hiddens,
                             activation='sigmoid',
                             name='encoded')
        prediction = Dense(2,
                           activation='softmax',
                           name='pred')
        # encoding
        encoded_s = encoding(input_s)
        encoded_t = encoding(input_t)
        # prediction
        pred_s = prediction(encoded_s)
        pred_t = prediction(encoded_t)
        # model definition
        self.nn = Model(input=[input_s,input_t],
                        output=[pred_s,pred_t])
        # adagrad optimizer, good choice for sparse data as in our case
        adagrad = Adagrad()  
        # model compilation
        self.nn.compile(loss='categorical_crossentropy',
                        optimizer=adagrad,
                        metrics=['accuracy'],
                        loss_weights=[1.,0.])
        # early stopping and save best model
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       verbose=0)
        checkpointer = ModelCheckpoint(filepath=self.tmp_folder+
                                                self.save_weights,
                                       monitor='val_pred_acc',
                                       verbose=0,
                                       save_best_only=True)
        self.callbacks = [early_stopping,checkpointer]
        # Create seperate model for activation visualization
        self.visualize_model = Model(input=[input_s,input_t],
                                     output=[encoded_s,encoded_t])
        
    def fit(self, x_s, y_s, x_t, val_set=None, init_weights=None, verbose=0):
        """
        train classifier
        """
        start = datetime.datetime.now().replace(microsecond=0)
        # init
        np.random.seed(0)
        self.create()
        if init_weights:
            # to use the same initial weights for all methods
            self.nn.load_weights(self.tmp_folder+init_weights+'.hdf5')
        dummy=np.zeros((x_t.shape[0],1))
        dummy[0]=1
        y_s = to_categorical(y_s.astype(int))
        y_t = to_categorical(dummy.astype(int))
        
        # main training function of keras
        # the early stopping criteria is a patience of 10 according to a 
        # validation set
        if not val_set:
            # the validation set is not given and randomly choosen
            self.nn.fit([x_s,x_t],
                        [y_s,y_t],
                        batch_size=self.batch_size,
                        shuffle=False,
                        nb_epoch=self.n_epochs,
                        callbacks=self.callbacks,
                        verbose=verbose,
                        validation_split=0.3)
        else:
            # the validation set is given
            # e.g. when applying the reverse cross-validation procedure
            y_val=to_categorical(val_set[1].astype(int))
            self.nn.fit([x_s,x_t],
                        [y_s,y_t],
                        batch_size=self.batch_size,
                        shuffle=False,
                        nb_epoch=self.n_epochs,
                        callbacks=self.callbacks,
                        verbose=verbose,
                        validation_data=([val_set[0],val_set[0]],
                                         [y_val,y_val]))
        self.load(self.save_weights)# use with checkpointer
        stop = datetime.datetime.now().replace(microsecond=0)
        if verbose:
            print('done in '+str(stop-start))
    
    def predict(self,x):
        """
        predict classifier
        """
        y = self.nn.predict([x,x])[1]
        out=np.zeros(y.shape[0])
        for i in range(out.shape[0]):
            out[i]=np.argmax(np.round(y[i,:]))
        return out
        
    def load(self,name):
        """
        load weights
        """
        self.create()
        self.nn.load_weights(self.tmp_folder+name)
        
    def save(self,name):
        """
        save weights
        """
        self.nn.save_weights(self.tmp_folder+name+'.hdf5',overwrite=True)
        
    def create_initial_weights(self,x_s,y_s,x_t,name):
        """
        create and save a random weight initialization
        done by setting up a randomly initialized keras model and perform
        one 1 epoch sgd update
        """
        input_s = Input(shape=(self.n_features,), name='souce_input')
        input_t = Input(shape=(self.n_features,), name='target_input')
        encoding = Dense(self.n_hiddens,
                         activation='sigmoid',
                         init='lecun_uniform',
                         name='encoded')
        prediction = Dense(2,
                           activation='softmax',
                           init='lecun_uniform',
                           name='pred')
        encoded_s = encoding(input_s)
        encoded_t = encoding(input_t)
        pred_s = prediction(encoded_s)
        pred_t = prediction(encoded_t)
        
        nn = Model(input=[input_s,input_t],
                        output=[pred_s,pred_t])
        sgd = SGD(0.1)
        nn.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['accuracy'],
                   loss_weights=[1.,0.])
        
        dummy=np.zeros((x_t.shape[0],1))
        dummy[0]=1
        y_s = to_categorical(y_s.astype(int))
        y_t = to_categorical(dummy.astype(int))
        nn.fit([x_s,x_t],[y_s,y_t],nb_epoch=1,validation_split=0.3,verbose=0)
        nn.save_weights(self.tmp_folder+name+'.hdf5',overwrite=True)
        
    def get_activations(self,x_s,x_t):
        """
        returns hidden activations
        """
        return self.visualize_model.predict([x_s,x_t])
        

