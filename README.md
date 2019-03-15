# Unsupervised Domain Adaptation with Moment Alignment Neural Networks (Keras)

This repository contains code for reproducing the experiments reported in the paper
- W. Zellinger, B.A. Moser, T. Grubinger, E. Lughofer, T. Natschlaeger, and S. Saminger-Platz, "Robust unsupervised domain adaptation for neural networks via moment alignment," [[Information Sciences 483: 174-191]], (https://doi.org/10.1016/j.ins.2019.01.025), [[arXiv preprint]](https://arxiv.org/abs/1711.06114), May 2019

that extends the preliminary conference version

- W.Zellinger, T. Grubinger, E. Lughofer, T. Natschlaeger, and Susanne Saminger-Platz, "Central moment discrepancy (cmd) for domain-invariant representation learning," International Conference on Learning Representations (ICLR), [[OpenReview.net]](https://openreview.net/forum?id=SkB-_mcel), 2017

# Requirements
The implementation is based on the neural networks library keras (version 1.1) and tested with theano backend (version 0.9). For installing theano and keras please follow the installation instruction on the respective github pages. You will also need: numpy, pandas, seaborn, matplotlib, and scipy.
The file artificial_example.py demonstrates how to update the approach for newer versions of keras (>2.0).

# Datasets
In our paper, we report results for one artificial dataset and two benchmark datasets: AmazonReview and Office. In addition, the model weights of the AlexNet model pre-trained on Imagenet are used. The artificial dataset and the AmazonReviews dataset are provided. The Office data set can be downloaded from https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view. Copy the folders amazon, dslr and webcam to data/office_dataset/. Download the AlexNet weights file from http://files.heuritech.com/weights/alexnet_weights.h5 and copy it to data/office_dataset/.

# Experiments
Use the files, artificial_example.py, object_recognition.py, sentiment_analysis.py and parameter_sensitivity.py to run the experiments and create the images from the paper. For faster evaluation times, the full object recognition experiment is under comments.

# System Configuration
Please note that the exact results depend on your system setup (CuDNN version, etc.), theano or tensorflow configuration (float32, etc.) and hardware (GPU etc.). In the paper, we report results based on the system described in the [requirements.txt](https://github.com/wzell/mann/blob/master/requirements.txt). Non-deterministic behaviour of [GPUs with keras](https://github.com/fchollet/keras/issues/850), [of theano on GPUs](https://groups.google.com/forum/#!topic/theano-users/Q9tD4Af_7ho) and [of tensorflow on GPUs](https://github.com/tensorflow/tensorflow/issues/2652) have been reported independently of our project.
