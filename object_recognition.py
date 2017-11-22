#!/usr/bin/env python
"""
Reproduce the object recognition experiment in [Zellinger2017a]

The central moment discrepancy (CMD) is used for domain adaptation
as first described in the preliminary conference paper [Zellinger2017b].
It is implemented as keras regularizer that can be used by shared layers.
This implementation is tested under keras 1.1.0.

For faster evaluation, the full experiment is under comments. Un-comment these
parts to obtain the accuracy for the CNN and the CMD model in the table of the
paper.

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

# moment alignment neural network (MANN) for object recognition based on
# the central moment discrepancy (CMD)
from models.mann_object_recognition import MANN
# CMD as activity regularizer for keras models
from models.central_moment_discrepancy import CMDRegularizer

from keras.preprocessing.image import ImageDataGenerator


TMP_FOLDER = 'temp/object_recognition/'
DATA_FOLDER = 'data/office_dataset/'
PRETRAINED_WEIGHTS = 'alexnet_weights.h5'
N_IMAGES_AM = 2817
N_IMAGES_DSLR = 498
N_IMAGES_WC = 795
S_IMAGE = 256
S_BATCH = 2
N_REPETITIONS = 1
#N_REPETITIONS = 5

print("\nLoading office image data...")
datagen = ImageDataGenerator()
am_gen = datagen.flow_from_directory(DATA_FOLDER+'amazon/images',
                                     target_size=(S_IMAGE, S_IMAGE),
                                     batch_size=S_BATCH)
dslr_gen = datagen.flow_from_directory(DATA_FOLDER+'dslr/images',
                                       target_size=(S_IMAGE, S_IMAGE),
                                       batch_size=S_BATCH)
wc_gen = datagen.flow_from_directory(DATA_FOLDER+'webcam/images',
                                     target_size=(S_IMAGE, S_IMAGE),
                                     batch_size=S_BATCH)

## data augmentation: random croppig and mirroring of images
## standardly used by domain adaptation methods
## Note that this is not used by robust implementation of moment alignment
## neural networks (Robust-MANN)
#dg_augm = ImageDataGenerator(horizontal_flip=True,
#                             zoom_range=.3)
#am_augm_gen = dg_augm.flow_from_directory(DATA_FOLDER+'amazon/images',
#                                          target_size=(S_IMAGE, S_IMAGE),
#                                          batch_size=S_BATCH)
#wc_augm_gen = dg_augm.flow_from_directory(DATA_FOLDER+'webcam/images',
#                                          target_size=(S_IMAGE, S_IMAGE),
#                                          batch_size=S_BATCH)
#dslr_augm_gen = dg_augm.flow_from_directory(DATA_FOLDER+'dslr/images',
#                                            target_size=(S_IMAGE, S_IMAGE),
#                                            batch_size=S_BATCH)


# create image representations of deep convolutional neural network AlexNet
print("\nCreating/Loading image representations via AlexNet model...")
nn = MANN(TMP_FOLDER)
x_am, y_am = nn.create_img_repr_alexnet(DATA_FOLDER+PRETRAINED_WEIGHTS, am_gen,
                                        'amazon', N_IMAGES_AM)
x_wc, y_wc = nn.create_img_repr_alexnet(DATA_FOLDER+PRETRAINED_WEIGHTS, wc_gen,
                                        'webcam', N_IMAGES_WC)
x_dslr, y_dslr = nn.create_img_repr_alexnet(DATA_FOLDER+PRETRAINED_WEIGHTS,
                                            dslr_gen, 'dslr', N_IMAGES_DSLR) 
#x_am_augm, y_am_augm = nn.create_img_repr_alexnet(DATA_FOLDER+PRETRAINED_WEIGHTS,
#                                                  am_augm_gen, 'amazon_augm',
#                                                  100000)
#x_wc_augm, y_wc_augm = nn.create_img_repr_alexnet(DATA_FOLDER+PRETRAINED_WEIGHTS,
#                                                  wc_augm_gen, 'webcam_augm',
#                                                  100000)
#x_dslr_augm, y_dslr_augm = nn.create_img_repr_alexnet(DATA_FOLDER+PRETRAINED_WEIGHTS,
#                                                      dslr_augm_gen,
#                                                      'dslr_augm', 100000)

print("\nRandom Repetitions...")
cmd = CMDRegularizer()
print("wc->dslr:")
#acc_wcdslr_nn = np.array([])
#acc_wcdslr_mann = np.array([])
acc_wcdslr_robustmann = np.array([])
for i in range(N_REPETITIONS):
    np.random.seed(i)
    print('--')
#    # train/test convolutional neural network on augmented data
#    nn_wc = MANN(TMP_FOLDER, n_features=256)
#    nn_wc.fit(x_wc_augm, y_wc_augm, x_dslr_augm)
#    acc_tst = nn_wc.evaluate(x_dslr, y_dslr)
#    acc_wcdslr_nn = np.append(acc_wcdslr_nn,acc_tst)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' nn-accuracy= '+str(acc_tst))
#    # train/test moment alignment neural network on augmented data
#    # equivalent to fine-tuning with learning rate zero in lower layers
#    mann_wc = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd)
#    mann_wc.fit(x_wc_augm, y_wc_augm, x_dslr_augm)
#    acc_tst_mann = mann_wc.evaluate(x_dslr, y_dslr)
#    acc_wcdslr_mann = np.append(acc_wcdslr_mann,acc_tst_mann)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' mann-accuracy= '+
#          str(acc_tst_mann))
    # train/test moment alignment neural network without manually tuned
    # learning rates and without data augmentation
    robustmann_wc = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd,
                         optimizer='adadelta')
    robustmann_wc.fit(x_wc, y_wc, x_dslr)
    acc_tst_robustmann = robustmann_wc.evaluate(x_dslr, y_dslr)
    acc_wcdslr_robustmann = np.append(acc_wcdslr_robustmann,acc_tst_robustmann)
    print(str(i+1)+'/'+str(N_REPETITIONS)+' robustmann-accuracy= '+
          str(acc_tst_robustmann))
print("dslr->wc:")
#acc_dslrwc_nn = np.array([])
#acc_dslrwc_mann = np.array([])
acc_dslrwc_robustmann = np.array([])
for i in range(N_REPETITIONS):
    np.random.seed(i)
    print('--')
#    nn_dslr = MANN(TMP_FOLDER, n_features=256)
#    nn_dslr.fit(x_dslr_augm, y_dslr_augm, x_wc_augm)
#    acc_tst = nn_dslr.evaluate(x_wc, y_wc)
#    acc_dslrwc_nn = np.append(acc_dslrwc_nn,acc_tst)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' nn-accuracy= '+str(acc_tst))
#    mann_dslr = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd)
#    mann_dslr.fit(x_dslr_augm, y_dslr_augm, x_wc_augm)
#    acc_tst_mann = mann_dslr.evaluate(x_wc, y_wc)
#    acc_dslrwc_mann = np.append(acc_dslrwc_mann,acc_tst_mann)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' mann-accuracy= '+
#          str(acc_tst_mann))
    robustmann_dslr = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd,
                         optimizer='adadelta')
    robustmann_dslr.fit(x_dslr, y_dslr, x_wc)
    acc_tst_robustmann = robustmann_dslr.evaluate(x_wc, y_wc)
    acc_dslrwc_robustmann = np.append(acc_dslrwc_robustmann,acc_tst_robustmann)
    print(str(i+1)+'/'+str(N_REPETITIONS)+' robustmann-accuracy= '+
          str(acc_tst_robustmann))
print("am->wc:")
#acc_amwc_nn = np.array([])
#acc_amwc_mann = np.array([])
acc_amwc_robustmann = np.array([])
for i in range(N_REPETITIONS):
    np.random.seed(i)
    print('--')
#    nn_am = MANN(TMP_FOLDER, n_features=256)
#    nn_am.fit(x_am_augm, y_am_augm, x_wc_augm)
#    acc_tst = nn_am.evaluate(x_wc, y_wc)
#    acc_amwc_nn = np.append(acc_amwc_nn,acc_tst)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' nn-accuracy= '+str(acc_tst))
#    mann_am = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd)
#    mann_am.fit(x_am_augm, y_am_augm, x_wc_augm)
#    acc_tst_mann = mann_am.evaluate(x_wc, y_wc)
#    acc_amwc_mann = np.append(acc_amwc_mann,acc_tst_mann)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' mann-accuracy= '+
#          str(acc_tst_mann))
    robustmann_am = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd,
                         optimizer='adadelta')
    robustmann_am.fit(x_am, y_am, x_wc)
    acc_tst_robustmann = robustmann_am.evaluate(x_wc, y_wc)
    acc_amwc_robustmann = np.append(acc_amwc_robustmann,acc_tst_robustmann)
    print(str(i+1)+'/'+str(N_REPETITIONS)+' robustmann-accuracy= '+
          str(acc_tst_robustmann))
print("am->dslr:")
#acc_amdslr_nn = np.array([])
#acc_amdslr_mann = np.array([])
acc_amdslr_robustmann = np.array([])
for i in range(N_REPETITIONS):
    np.random.seed(i)
    print('--')
#    nn_am = MANN(TMP_FOLDER, n_features=256)
#    nn_am.fit(x_am_augm, y_am_augm, x_dslr_augm)
#    acc_tst = nn_am.evaluate(x_dslr, y_dslr)
#    acc_amdslr_nn = np.append(acc_amdslr_nn,acc_tst)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' nn-accuracy= '+str(acc_tst))
#    mann_am = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd)
#    mann_am.fit(x_am_augm, y_am_augm, x_dslr_augm)
#    acc_tst_mann = mann_am.evaluate(x_dslr, y_dslr)
#    acc_amdslr_mann = np.append(acc_amdslr_mann,acc_tst_mann)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' mann-accuracy= '+
#          str(acc_tst_mann))
    robustmann_am = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd,
                         optimizer='adadelta')
    robustmann_am.fit(x_am, y_am, x_dslr)
    acc_tst_robustmann = robustmann_am.evaluate(x_dslr, y_dslr)
    acc_amdslr_robustmann = np.append(acc_amdslr_robustmann,acc_tst_robustmann)
    print(str(i+1)+'/'+str(N_REPETITIONS)+' robustmann-accuracy= '+
          str(acc_tst_robustmann))
print("dslr->am:")
#acc_dslram_nn = np.array([])
#acc_dslram_mann = np.array([])
acc_dslram_robustmann = np.array([])
for i in range(N_REPETITIONS):
    np.random.seed(i)
    print('--')
#    nn_dslr = MANN(TMP_FOLDER, n_features=256)
#    nn_dslr.fit(x_dslr_augm, y_dslr_augm, x_am_augm)
#    acc_tst = nn_dslr.evaluate(x_am, y_am)
#    acc_dslram_nn = np.append(acc_dslram_nn,acc_tst)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' nn-accuracy= '+str(acc_tst))
#    mann_dslr = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd)
#    mann_dslr.fit(x_dslr_augm, y_dslr_augm, x_am_augm)
#    acc_tst_mann = mann_dslr.evaluate(x_am, y_am)
#    acc_dslram_mann = np.append(acc_dslram_mann,acc_tst_mann)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' mann-accuracy= '+
#          str(acc_tst_mann))
    robustmann_dslr = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd,
                         optimizer='adadelta')
    robustmann_dslr.fit(x_dslr, y_dslr, x_am)
    acc_tst_robustmann = robustmann_dslr.evaluate(x_am, y_am)
    acc_dslram_robustmann = np.append(acc_dslram_robustmann,acc_tst_robustmann)
    print(str(i+1)+'/'+str(N_REPETITIONS)+' robustmann-accuracy= '+
          str(acc_tst_robustmann))
print("wc->am:")
#acc_wcam_nn = np.array([])
#acc_wcam_mann = np.array([])
acc_wcam_robustmann = np.array([])
for i in range(N_REPETITIONS):
    np.random.seed(i)
    print('--')
#    nn_wc = MANN(TMP_FOLDER, n_features=256)
#    nn_wc.fit(x_wc_augm, y_wc_augm, x_am_augm)
#    acc_tst = nn_wc.evaluate(x_am, y_am)
#    acc_wcam_nn = np.append(acc_wcam_nn,acc_tst)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' nn-accuracy= '+str(acc_tst))
#    mann_wc = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd)
#    mann_wc.fit(x_wc_augm, y_wc_augm, x_am_augm)
#    acc_tst_mann = mann_wc.evaluate(x_am, y_am)
#    acc_wcam_mann = np.append(acc_wcam_mann,acc_tst_mann)
#    print(str(i+1)+'/'+str(N_REPETITIONS)+' mann-accuracy= '+
#          str(acc_tst_mann))
    robustmann_wc = MANN(TMP_FOLDER, n_features=256, activity_regularizer=cmd,
                         optimizer='adadelta')
    robustmann_wc.fit(x_wc, y_wc, x_am)
    acc_tst_robustmann = robustmann_wc.evaluate(x_am, y_am)
    acc_wcam_robustmann = np.append(acc_wcam_robustmann,acc_tst_robustmann)
    print(str(i+1)+'/'+str(N_REPETITIONS)+' robustmann-accuracy= '+
          str(acc_tst_robustmann)) 
print('------------------------------------------------------------------')
print("am->wc")
#print('CNN acc-tst= '+str(acc_amwc_nn.mean())+'+-'
#+str(acc_amwc_nn.std()))
#print('MANN acc-tst= '+str(acc_amwc_mann.mean())+'+-'
#+str(acc_amwc_mann.std()))
print('Robust-MANN acc-tst= '+str(acc_amwc_robustmann.mean())+'+-'
+str(acc_amwc_robustmann.std()))
print("dslr->wc")
#print('CNN acc-tst= '+str(acc_dslrwc_nn.mean())+'+-'
#+str(acc_dslrwc_nn.std()))
#print('MANN acc-tst= '+str(acc_dslrwc_mann.mean())+'+-'
#+str(acc_dslrwc_mann.std()))
print('Robust-MANN acc-tst= '+str(acc_dslrwc_robustmann.mean())+'+-'
+str(acc_dslrwc_robustmann.std()))
print("wc->dslr")
#print('CNN acc-tst= '+str(acc_wcdslr_nn.mean())+'+-'
#+str(acc_wcdslr_nn.std()))
#print('MANN acc-tst= '+str(acc_wcdslr_mann.mean())+'+-'
#+str(acc_wcdslr_mann.std()))
print('Robust-MANN acc-tst= '+str(acc_wcdslr_robustmann.mean())+'+-'
+str(acc_wcdslr_robustmann.std()))
print("am->dslr")
#print('CNN acc-tst= '+str(acc_amdslr_nn.mean())+'+-'
#+str(acc_amdslr_nn.std()))
#print('MANN acc-tst= '+str(acc_amdslr_mann.mean())+'+-'
#+str(acc_amdslr_mann.std()))
print('Robust-MANN acc-tst= '+str(acc_amdslr_robustmann.mean())+'+-'
+str(acc_amdslr_robustmann.std()))
print("dslr->am")
#print('CNN acc-tst= '+str(acc_dslram_nn.mean())+'+-'
#+str(acc_dslram_nn.std()))
#print('MANN acc-tst= '+str(acc_dslram_mann.mean())+'+-'
#+str(acc_dslram_mann.std()))
print('Robust-MANN acc-tst= '+str(acc_dslram_robustmann.mean())+'+-'
+str(acc_dslram_robustmann.std()))
print("wc->am")
#print('CNN acc-tst= '+str(acc_wcam_nn.mean())+'+-'
#+str(acc_wcam_nn.std()))
#print('MANN acc-tst= '+str(acc_wcam_mann.mean())+'+-'
#+str(acc_wcam_mann.std()))
print('Robust-MANN acc-tst= '+str(acc_wcam_robustmann.mean())+'+-'
+str(acc_wcam_robustmann.std()))
print('------------------------------------------------------------------')
    

