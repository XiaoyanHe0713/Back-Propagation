---
title: Experiment in Multi-Layers Neural Network of CIFAR-10 Dataset
author: Dongze Li, Yunfan Long, Xiaoyan He
Date: October 2022
---

# Experiment in Multi-Layers Neural Network of CIFAR-10 Dataset
Source code from CSE 151B Fall 2022 PA2

## Description

We use `NumPy` and basic Python packages to build multi-layer back propagation models to train on the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset, which contains 10 classes of 32x32 pixel-size images. Also, we use `Matplotlib` to visulize both loss and accuracy in training as well as testing. In this project, we explored the multi-layer neural network to classify images from different classes in CIFAR-10 dataset. We used softmax function as the activation function for output layer. In our original model, we use one hidden layer with 128 hidden units, and chose $\tanh$ function as the activation for hidden layer with learning rate as $1 \times 10^{-5}$, batch size as $128$, momentum gamma as $0.9$ with no regularization. We applied stochastic gradient descent in all parts of our experiment. With our original model, we reached the test accuracy of $47.36\%$. Then, in other parts of experiment, we tried L1 and L2 regularization, and also tried to use other activation function(ReLU \& Sigmoid) for hidden units and change the number of hidden units and hidden layers. We achieved $50.77\%$ as our highest test accuracy with ReLU activation for hidden units.

## Getting Started

### Dependencies

* `argparse`
* `copy`
* `Matplotlib`
* `NumPy`
* `os`
* `pandas`
* `pickle`
* `PIL`
* `Python3`
* `tqdm`
* `yaml`

### Installs

* `get_cifar10data.sh` (download the dataset)
    * Simply run the bash script, then the dataset will be downloaded.

### Files and Folders

* `configs` (contains all the configuration files)
    * `config_2b.yaml` (contains all the hyperparameters for part 2b)
    * `config_2c.yaml` (contains all the hyperparameters for part 2c)
    * `config_2d.yaml` (contains all the hyperparameters for part 2d)
    * `config_2e.yaml` (contains all the hyperparameters for part 2e)
    * `config_2f_i.yaml` (contains all the hyperparameters for part 2f, subpart i)
    * `config_2f_ii.yaml` (contains all the hyperparameters for part 2f, subpart ii)
* `data` (dataset folder)
    * `cifar-10-batches-py` (contains all the dataset)
* `plots` (contains all plots related to the dataset)
    * `test_activation` (contains all plots related to the activation function)
    * `test_hidden_layer` (contains all plots related to the number of hidden layers)
    * `test_hidden_unit` (contains all plots related to the number of hidden units)
    * `test_momentum` (contains all plots related to the momentum)
    * `test_regularization` (contains all plots related to the regularization)
* `constants.py` (contains all the constants)
* `get_cifar10data.sh` (download the dataset)
* `main.py` (main file to run the code)
* `neural_network.py` (contains all the neural network related functions)
* `README.md` (this file)
* `train.py` (contains all the training related functions)
* `utils.py` (contains all the utility functions)

### Executing Program

* Go to the correct directory where all files located
* In the terminal, run `python main.py` with or without the following argument phrases:
    * `[--experiment]`
    * `[-h]`

* Here are the explanations of different argument phrases:
    * `--experiment` (choose which experiment to run, default is `'test_gradient'`, which use the `config_2b.yaml` config file)
        * `'test_gradients'` (test between the numerical approximation of the gradients and the gradient of the neural network, which use the `config_2b.yaml` file as the hyperparameters)
        * `'test_momentum'` (train and test the network with momentum, which use the `config_2c.yaml` file as the hyperparameters)
        * `'test_regularization'` (train and test the network with regularization, which use the `config_2d.yaml` file as the hyperparameters)
        * `'test_activation'` (train and test the network with different activation functions, which use the `config_2e.yaml` file as the hyperparameters)
        * `'test_hidden_units'` (train and test the network with different number of hidden units, which use the `config_2f_i.yaml` file as the hyperparameters)
        * `'test_hidden_layers'` (train and test the network with different number of hidden layers, which use the `config_2f_ii.yaml` file as the hyperparameters)
    * `-h` (show the help message and exit)

* How to change the hyperparameters:
    * Go to the `configs` folder
    * Open the corresponding config file
    * Change the hyperparameters
    * Save the file

## Help
Make sure to download the dataset first before running anything. Notice that for the `plots` folder and its inside folders, it will be created automatically when running the code, and different folder names represent the hyperparameters and testing parts. Also, the `data` folder will be created automatically when running the `get_cifar10data.sh` bash script.

## Authors
Contributor names and contacts info (alphabetic order):

* Li, Dongze
    * dol005@ucsd.edu
* Long, Yunfan
    * yulong@ucsd.edu
* He, Xiaoyan
    * x6he@ucsd.edu

## Acknowledgments

We appreciate the help from the coruse website, Piazza, as well as TAs and Tutors' office hours. We also appreciate Professor [Garrison W. Cottrell](https://cseweb.ucsd.edu/~gary/) for his lectures and teachings.
