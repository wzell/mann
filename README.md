# Robust Unsupervised Domain Adaptation via Moment Alignment Neural Networks (Keras)

This repository contains code for reproducing the experiments reported in the paper
- W. Zellinger, B.A. Moser, T. Grubinger, E. Lughofer, T. Natschlaeger, and S. Saminger-Platz, "Robust unsupervised domain adaptation for neural networks via moment alignment," arXiv preprint arXiv:1711.06114, 2017

that extends the preliminary conference version

- W.Zellinger, T. Grubinger, E. Lughofer, T. Natschlaeger, and Susanne Saminger-Platz, "Central moment discrepancy (cmd) for domain-invariant representation learning," International Conference on Learning Representations (ICLR), 2017

# Requirements
The implementation is based on the neural networks library keras (version 1.1) and tested with theano backend. For installing theano and keras please follow the installation instruction on the respective github pages. You will also need: numpy, pandas, seaborn, matplotlib, and scipy.

# Datasets
In our paper, we report results for one artificial dataset and two benchmark datasets: AmazonReview and Office. In addition, the model weights of the AlexNet model pre-trained on Imagenet are used. The artificial dataset and the AmazonReviews dataset are provided. The Office data set can be downloaded from https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view. Copy the folders amazon, dslr and webcam to data/office_dataset/. Download the AlexNet weights file from http://files.heuritech.com/weights/alexnet_weights.h5 and copy it to  data/office_dataset/.

# Experiments
Use the files, artificial_example.py, object_recognition.py, sentiment_analysis.py and parameter_sensitivity.py to run the experiments and create the images from the paper. Change the N_REPETITIONS parameter for shorter evaluation times.

# System Independence of Results
We tested our code on different environments with different random seedsand different theano configurations. Sometimes, we experinced sligthly different random results. This behaviour is [known for keras on GPU](https://github.com/fchollet/keras/issues/850) and [the theano backend using the GPU](https://groups.google.com/forum/#!topic/theano-users/Q9tD4Af_7ho) and even for [different GPU configurations using tensorflow](https://github.com/tensorflow/tensorflow/issues/2652). However, although sligthly different numbers occured, we could not identify any contradictions to the claims of the paper.
