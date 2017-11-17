#!/usr/bin/env python
"""
Moment alignment neural network (MANN) for object recognition

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

from __future__ import print_function

import numpy as np
import datetime

from os.path import isfile
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import sgd, Adadelta
from keras.layers import Lambda
from keras.layers import Activation
import keras.backend as K
from keras.layers import merge

np.random.seed(0)

class Batches:
    """
    class structure for generating batches that are balanced w.r.t. classes
    """
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        
    def next_batch(self):
        """
        get next batch
        """
        if self.y.shape[0]>self.batch_size:
            # this case is not used in the experiments
            x, y = self.next_batch_smaller(self.x, self.y, self.batch_size)
        else:
            # only this case is used
            x, y = self.next_batch_bigger()
        return x,y
            
    def next_batch_smaller(self, x, y, batch_size):
        """
        downsample a batch
        """
        x_batch = np.array([])
        y_batch = np.array([])
        # n_min is the smallest class size
        n_min = int(np.min(self.y.sum(0)))
        n_rest = int(batch_size - n_min*y.shape[1])
        if n_rest<0:
            n_min = int(batch_size /y.shape[1])
            n_rest = batch_size %y.shape[1]
        ind_chos = np.array([])
        is_first = True
        # fill with n_min samples per class
        for cl in range(y.shape[1]):
            ind_cl = np.arange(y.shape[0])[y[:,cl]!=0]
            ind_cl_choose = \
            np.random.permutation(np.arange(ind_cl.shape[0]))[:n_min]
            if is_first:
                x_batch = x[ind_cl[ind_cl_choose]]
                y_batch = y[ind_cl[ind_cl_choose]]
                is_first = False
            else:
                x_batch = np.concatenate((x_batch,x[ind_cl[ind_cl_choose]]),
                                         axis=0)
                y_batch = np.concatenate((y_batch,y[ind_cl[ind_cl_choose]]),
                                         axis=0)
            ind_chos = np.concatenate((ind_chos,ind_cl[ind_cl_choose]))
        # fill with n_rest random samples
        mask = np.ones(x.shape[0],dtype=bool)
        mask[ind_chos.astype(int)] = False
        x_rem = x[mask]
        y_rem = y[mask]
        ind_choose = np.random.permutation(np.arange(x_rem.shape[0]))[:n_rest]
        x_batch = np.concatenate((x_batch,x_rem[ind_choose]),axis=0)
        y_batch = np.concatenate((y_batch,y_rem[ind_choose]),axis=0)
        return x_batch, y_batch
        
    def next_batch_bigger(self):
        """
        upsample a batch
        """
        n_remaining = self.batch_size
        is_first = True
        while n_remaining >= self.x.shape[0]:
            # copy full samples to the batch
            if is_first:
                x_batch = self.x
                y_batch = self.y
                is_first = False
            else:
                x_batch = np.concatenate((x_batch,self.x),axis=0)
                y_batch = np.concatenate((y_batch,self.y),axis=0)
            n_remaining -= self.x.shape[0]
        # fill the remaining samples such that the classes are balanced
        x_add, y_add = self.next_batch_smaller(self.x, self.y, n_remaining)
        x_batch = np.concatenate((x_batch,x_add),axis=0)
        y_batch = np.concatenate((y_batch,y_add),axis=0)
        return x_batch, y_batch
        
            
class MANN:
    """
    class structure for moment alignment neural networks
    """
    def __init__(self,
                 folder,
                 n_features=256,
                 max_n_epoch=10000,
                 activity_regularizer=None,
                 save_weights='save_weights',
                 optimizer = 'sgd'):
        self.nn = None
        self.exp_folder = folder
        self.max_n_epoch = max_n_epoch
        self.n_features = n_features
        self.save_weights = save_weights
        self.activity_regularizer = activity_regularizer
        self.visualize_model = None
        self.optimizer = optimizer
        
    def create(self):
        """
        create two layer classifier
        """
        # input
        img_repr_s = Input(shape=(4096,), name='souce_input')
        img_repr_t = Input(shape=(4096,), name='target_input')
        # layers
        if self.activity_regularizer:
            shared_dense = Dense(self.n_features,
                                 name='shared_dense',
                                 activation='sigmoid',
                                 init='he_normal',
                                 activity_regularizer=self.activity_regularizer)
        else:
            shared_dense = Dense(self.n_features,
                                 name='shared_dense',
                                 activation='sigmoid',
                                 init='he_normal')
        classifier = Dense(31,
                           name='clf',
                           activation='softmax')
        # encoding
        s_d_s = shared_dense(img_repr_s)
        s_d_s = Dropout(0.8)(s_d_s)
        s_d_t = shared_dense(img_repr_t)
        s_d_t = Dropout(0.8)(s_d_t)
        # prediction
        pred_s = classifier(s_d_s)
        pred_t = classifier(s_d_t)
        # model definition
        self.nn = Model(input=[img_repr_s, img_repr_t],
                        output=[pred_s, pred_t])
        # model compilation
        if self.optimizer=='sgd':
            opt = sgd(lr=1e-2,decay=1e-4, momentum=0.9)
        else:
            opt = Adadelta()
        self.nn.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['categorical_accuracy'],
                        loss_weights=[1.,0.])
    
    def create_img_repr_alexnet(self, weights_file, gen, save_name, max_n_imgs):
        """
        calculate image representation
        via pre-trained convolutional neural network
        """
        an = self.AlexNet(weights_file)
        an_repr = Model(input=an.input,
                           output=[an.get_layer('dense_2').output])
        an_repr.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['categorical_accuracy'])
        
        if not isfile(self.exp_folder+save_name+'_img_repr.npy'):
            print('Calculating image representations of '+str(save_name)+'..')
            repres = np.array([])
            labels = None
            batch_size = 0
            is_first = True
            n_processed = 0
            while True:
                (x, y) = gen.next()
                # crop img to use by pre-trained network
                if x.shape[2]>227:
                    cut = int((x.shape[2]-227)/2)
                    if cut%2==0:
                        x = x[:,[0,1,2],cut+1:x.shape[2]-cut,cut+1:x.shape[2]-cut]
                    else:
                        x = x[:,[0,1,2],cut:x.shape[2]-cut,cut:x.shape[2]-cut]
                # subtract imagenet mean of imagenet challange
                # according to best practice in domain adaptation
                if save_name=='amazon':
                    for img in x:
                        img[0,:,:]=img[0,:,:]-img[0,:,:].mean() + 104.0
                        img[1,:,:]=img[1,:,:]-img[1,:,:].mean() + 116.0
                        img[2,:,:]=img[2,:,:]-img[2,:,:].mean() + 122.0
                        n_processed += 1
                else:
                    n_processed += x.shape[0]
                    batch_size = x.shape[0]
                # calculate img representations
                img_repr = an_repr.predict(x)
                if is_first:
                    batch_size = x.shape[0]
                    is_first = False
                    repres = np.zeros([max_n_imgs,img_repr.shape[1]])
                    labels = np.zeros([max_n_imgs,y.shape[1]])
                if n_processed<=max_n_imgs:
                    repres[n_processed-batch_size:n_processed,:]=img_repr
                    labels[n_processed-batch_size:n_processed,:] = y
                else:
                    repres[max_n_imgs-batch_size:n_processed,:]=img_repr
                    labels[max_n_imgs-batch_size:n_processed,:] = y
                if n_processed >= max_n_imgs:
                    break
                if n_processed%100 == 0:
                    print('processing img '+str(n_processed)+'..')
            np.save(open(self.exp_folder+save_name+'_img_repr.npy', 'w'),
                    repres)
            np.save(open(self.exp_folder+save_name+'_labels.npy', 'w'), labels)
        else:
            print('Loading image representations of '+str(save_name)+'..')
            repres = np.load(open(self.exp_folder+save_name+'_img_repr.npy'))
            labels = np.load(open(self.exp_folder+save_name+'_labels.npy'))
        return repres, labels
        
    def fit(self, x_s, y_s, x_t, verbose=False, x_val=[], y_val=[]):
        """
        train classifier
        """
        start = datetime.datetime.now().replace(microsecond=0)
        # init
        self.create()            
        best_acc = 0
        best_loss = 0
        counter = 0
        dummy_y_t =np.zeros((x_t.shape[0],y_s.shape[1]))
        # batch size is 2000 (arbitrary) when working with augmented data
        # batch size is 
        # Note that such high numbers are not possible in fine-tuning with
        # the learning rates of lower layers >0. If we set the lower learning
        # rates to zero, this is equivalent to pre-computing image
        # representations, as we are doing.
        iter_batches = None
        if x_t.shape[0]>3000:
            # data augmentation is used : equal batches are computed
            batch_s = Batches(x_s, y_s, 2000)
            batch_t = Batches(x_t, dummy_y_t, 2000)
        elif x_t.shape[0]>=x_s.shape[0]:
            # target batch is larger than source batch
            # source batch will be up-sampled via class-balanced copies
            iter_batches = Batches(x_s, y_s, x_t.shape[0])
        else:
            # target batch is smaller than source batch
            # target batch will be randomly up-sampled
            iter_batches = Batches(x_t, dummy_y_t, x_s.shape[0])
        
        for i in range(self.max_n_epoch):
            if x_t.shape[0]>3000:
                # equal batches are generated
                x_s_batch, y_s_batch = batch_s.next_batch()
                x_t_batch, y_t_batch = batch_t.next_batch()
            elif x_t.shape[0]>=x_s.shape[0]:
                # source batch is up-sampled via class-balanced copies
                x_s_batch, y_s_batch = iter_batches.next_batch()
                x_t_batch, y_t_batch = x_t,dummy_y_t
            else:
                # target batch is randomly up-sampled
                x_s_batch, y_s_batch = x_s, y_s
                x_t_batch, y_t_batch = iter_batches.next_batch()
            # one full-batch update
            metrics = self.nn.train_on_batch([x_s_batch, x_t_batch],
                                             [y_s_batch, y_t_batch])
            if metrics[3]>best_acc:
                # an improvement happened
                self.save(self.save_weights)
                best_acc = metrics[3]
                best_loss = metrics[1]
                counter = 0  
            elif metrics[3]==best_acc and metrics[1]<best_loss:
                # save model with best accuracy and best loss
                self.save(self.save_weights)
                best_loss = metrics[1]
                best_acc = metrics[3]
                counter+=1
            else:
                counter+=1
            # Try the verbose command and you will get a fealing for the target
            # error during training. Maybe manually decreasing CMD weighting
            # can help the optimization, as used by various other works.
            if i%2 == 0 and verbose:
                accs = self.nn.evaluate([x_val, x_val],
                                        [y_val, y_val],
                                        verbose = 0)
                print('Batch update %.4d loss= %.4f tr-acc= %.4f tst-acc= %.4f'
                % (i, metrics[1], best_acc, accs[4]))
            if counter>1000:
                # early stopping after 1000 epochs
                # without accuracy increase
                break
        # load best model
        self.load(self.save_weights)
        stop = datetime.datetime.now().replace(microsecond=0)
        print('done in '+str(stop-start))
        
    def evaluate(self, x, y):
        """
        evaluate classifier
        """
        accs = self.nn.evaluate([x, x],
                                [y, y],
                                verbose = 0)
        return accs[4]
        
    def predict(self, x):
        """
        predict classifier
        """
        return self.nn.predict([x, x])[1]
        
    def save(self, name):
        """
        save weights
        """
        self.nn.save_weights(self.exp_folder+name+'.hdf5',overwrite=True) 
        
    def load(self,name):
        """
        load weights
        """
        self.create()
        self.nn.load_weights(self.exp_folder+name+'.hdf5')
        
    def AlexNet(self, weights_path=None):
        """
        AlexNet
        implemented in keras and weights ported from caffe by
        https://github.com/heuritech/convnets-keras
        
        A. Krizhevsky, I. Sutskever, and G. E. Hinton, "Imagenet classification
        with deep convolutional neural networks," in Advances in neural
        information processing systems, pp. 109--1105, 2012.
        """
        
        def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5,
                                      **kwargs):        
            def f(X):
                b, ch, r, c = X.shape
                half = n // 2
                square = K.square(X)
                extra_channels = \
                K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)),
                                     (0, half))
                extra_channels = \
                K.permute_dimensions(extra_channels, (0, 3, 1, 2))
                scale = k
                for i in range(n):
                    scale += alpha * extra_channels[:, i:i + ch, :, :]
                scale = scale ** beta
                return X / scale
            return Lambda(f, output_shape=lambda input_shape: input_shape,
                          **kwargs)
        
        def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
            def f(X):
                div = X.shape[axis] // ratio_split
                if axis == 0:
                    output = X[id_split * div:(id_split + 1) * div, :, :, :]
                elif axis == 1:
                    output = X[:, id_split * div:(id_split + 1) * div, :, :]
                elif axis == 2:
                    output = X[:, :, id_split * div:(id_split + 1) * div, :]
                elif axis == 3:
                    output = X[:, :, :, id_split * div:(id_split + 1) * div]
                else:
                    raise ValueError('This axis is not possible')
                return output
        
            def g(input_shape):
                output_shape = list(input_shape)
                output_shape[axis] = output_shape[axis] // ratio_split
                return tuple(output_shape)
        
            return Lambda(f, output_shape=lambda input_shape: g(input_shape),
                          **kwargs)
        
        inputs = Input(shape=(3, 227, 227))
    
        conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                               name='conv_1')(inputs)
    
        conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
        conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
        conv_2 = ZeroPadding2D((2, 2))(conv_2)
        conv_2 = merge([Convolution2D(128, 5, 5, activation='relu',
                                      name='conv_2_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_2)
                               ) for i in range(2)],
                                      mode='concat', concat_axis=1,
                                      name='conv_2')
    
        conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
        conv_3 = crosschannelnormalization()(conv_3)
        conv_3 = ZeroPadding2D((1, 1))(conv_3)
        conv_3 = Convolution2D(384, 3, 3, activation='relu',
                               name='conv_3')(conv_3)
    
        conv_4 = ZeroPadding2D((1, 1))(conv_3)
        conv_4 = merge([
                           Convolution2D(192, 3, 3, activation='relu',
                                         name='conv_4_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_4)
                               ) for i in range(2)], mode='concat',
                                         concat_axis=1, name='conv_4')
    
        conv_5 = ZeroPadding2D((1, 1))(conv_4)
        conv_5 = merge([
                           Convolution2D(128, 3, 3, activation='relu',
                                         name='conv_5_' + str(i + 1))(
                               splittensor(ratio_split=2, id_split=i)(conv_5)
                               ) for i in range(2)],
                                         mode='concat', concat_axis=1,
                                         name='conv_5')
    
        dense_1 = MaxPooling2D((3, 3), strides=(2, 2),
                               name='convpool_5')(conv_5)
    
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, name='dense_3')(dense_3)
        prediction = Activation('softmax', name='softmax')(dense_3)
    
        model = Model(input=inputs, output=prediction)
    
        if weights_path:
            model.load_weights(weights_path)
    
        return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        