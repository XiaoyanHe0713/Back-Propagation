################################################################################
# CSE 151B: Programming Assignment 2
# Fall 2022
# Code by Chaitanya Animesh & Shreyas Anantha Ramaprasad
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import gradient
from constants import *
from train import *
from gradient import *
import argparse

#TODO
def main(args):

    # Read the required config
    # Create different config files for different experiments
    configFile=None #Will contain the name of the config file to be loaded
    if (args.experiment == 'test_gradients'):  #2b
        configFile = "config_2b.yaml"
    elif(args.experiment=='test_momentum'):  #2c
        configFile = "config_2c.yaml"
    elif (args.experiment == 'test_regularization'): #2d
        configFile = "config_2d.yaml"
    elif (args.experiment == 'test_activation'): #2e
        configFile = "config_2e.yaml"
    elif (args.experiment == 'test_hidden_units'):  #2f-i
        configFile = "config_2f_i.yaml"
    elif (args.experiment == 'test_hidden_layers'):  #2f-ii
        configFile = "config_2f_ii.yaml"

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path=datasetDir)  # Set datasetDir in constants.py

    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile) # Set configYamlPath, configFile  in constants.py

    if(args.experiment == 'test_gradients'):
        gradient.checkGradient(x_train, y_train, config)
    
    else:
        # Create a Neural Network object which will be our model
        model = Neuralnetwork(config)

        # train the model. Use train.py's train method for this
        model, train_loss, train_acc, valid_loss, valid_acc, early_stop = train(model, x_train, y_train, x_valid, y_valid, config)

        # print("Training Loss: ", train_loss)
        # print("Training Accuracy: ", train_acc)
    
        # plot
        util.plots(train_loss, train_acc, valid_loss, valid_acc, early_stop, config, args.experiment)

        # test the model. Use train.py's modelTest method for this
        test_acc, test_loss =  modelTest(model, x_test, y_test)

        # Print test accuracy and test loss
        print('Test Accuracy:', test_acc, ' Test Loss:', test_loss)



if __name__ == "__main__":

    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum', help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)