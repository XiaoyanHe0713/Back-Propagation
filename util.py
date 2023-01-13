import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants



def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)



def normalize_data(inp, p=False):
    """
    Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to seprate the channels and then undoing it while returning

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """

    inp = inp.reshape(-1, 1024, 3)
    inp = inp.transpose(0, 2, 1)

    if p:
        red = inp[0, 0, :]
        green = inp[0, 1, :]
        blue = inp[0, 2, :]
        print("mean of red: ", np.mean(red))
        print("std of red: ", np.std(red))
        print("mean of green: ", np.mean(green))
        print("std of green: ", np.std(green))
        print("mean of blue: ", np.mean(blue))
        print("std of blue: ", np.std(blue))

    inp = (inp - np.mean(inp, axis=2, keepdims=True)) / np.std(inp, axis=2, keepdims=True)
    inp = inp.transpose(0, 2, 1)
    inp = inp.reshape(-1, 3072)

    return inp



def one_hot_encoding(labels, num_classes=10):
    """
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (10 for CIFAR-10)

    returns:
        oneHot : N X num_classes 2D array

    """
    return np.eye(num_classes)[labels].reshape(-1, num_classes)



def generate_minibatches(dataset, batch_size=64):
    """
        Generates minibatches of the dataset

        args:
            dataset : 2D Array N (examples) X d (dimensions)
            batch_size: mini batch size. Default value=64

        yields:
            (X,y) tuple of size=batch_size

        """

    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def calculateCorrect(y,t):  #Feel free to use this function to return accuracy instead of number of correct predictions
    """
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        the number of correct predictions
    """
    
    return np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1))

def append_bias(X):
    """
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    
    return np.hstack((np.ones((len(X), 1)), X))

def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop, config, experiment):

    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. epoch=-1 means the last epoch was the best one
    """
    order = ('activation', 'learning_rate', 'batch_size', 'epochs', 'early_stop', 'early_stop_epoch',
             'isL2', 'regularization', 'momentum', 'momentum_gamma', 'weight_type', 'layer_specs')
    chart_config = '-'.join('{}_{}'.format(k, config[k]) for k in sorted(config, key=order.index) if k != 'layer_specs')
    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation + experiment + '/' + experiment + "_loss" + " - " + chart_config + ".eps")
    plt.show()

    # Save the loss plot
    fig1.savefig(constants.saveLocation + experiment + '/' + experiment + "_loss" + " - " + chart_config + ".png")

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation + experiment + '/' + experiment + "_accuracy" + " - " + chart_config + ".eps")
    plt.show()

    # Save the accuracy plot
    fig2.savefig(constants.saveLocation + experiment + '/' + experiment + "_accuracy" + " - " + chart_config + ".png")

    #Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation + experiment + '/' + experiment + "_trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation + experiment + '/' + experiment + "_valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation + experiment + '/' + experiment + "_trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation + experiment + '/' + experiment + "_valEpochAccuracy.csv")



def createTrainValSplit(x_train,y_train):
    """
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the train-val split.
    """
    
    #Shuffling the data
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    #Creating the train-val split
    split = int(0.8*len(x_train))
    x_train, x_val = x_train[:split], x_train[split:]
    y_train, y_val = y_train[:split], y_train[split:]

    return x_train, y_train, x_val, y_val



def load_data(path):
    """
    Loads, splits our dataset- CIFAR-10 into train, val and test sets and normalizes them

    args:
        path: Path to cifar-10 dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, constants.cifar10_directory)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    for i in range(1,constants.cifar10_trainBatchFiles+1):
        images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
        data = images_dict[b'data']
        label = images_dict[b'labels']
        train_labels.extend(label)
        train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels).reshape((len(train_labels),-1))
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images,train_labels)

    train_normalized_images =  normalize_data(train_images, p=True)
    train_one_hot_labels = one_hot_encoding(train_labels)

    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels)

    test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape((len(test_labels),-1))
    test_normalized_images= normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels)
    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels