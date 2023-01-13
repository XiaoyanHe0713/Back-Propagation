import copy
from neuralnet import *
from tqdm import tqdm

def train(model, x_train, y_train, x_valid, y_valid, config):
    epochs = config['epochs']
    batch_size = config['batch_size']

    trainEpochLoss = []
    trainEpochAccuracy = []
    valEpochLoss = []
    valEpochAccuracy = []

    bestLoss = np.inf
    earlyStop = -1
    bestweights = []

    for epoch in tqdm(range(epochs)):
        generator = util.generate_minibatches((x_train, y_train), batch_size)
        for x_batch, y_batch in generator:
            model.forward(x_batch)
            model.backward(targets = y_batch)
        
        train_pred = model.forward(x_train)
        train_loss = model.loss(train_pred, y_train)
        train_acc = util.calculateCorrect(train_pred, y_train)/len(y_train)
        trainEpochLoss.append(train_loss)
        trainEpochAccuracy.append(train_acc)

        val_pred = model.forward(x_valid)
        val_loss = model.loss(val_pred, y_valid)
        val_acc = util.calculateCorrect(val_pred, y_valid)/len(y_valid)
        valEpochLoss.append(val_loss)
        valEpochAccuracy.append(val_acc)

        if config['early_stop']:
            if val_loss < bestLoss:
                bestLoss = val_loss
                earlyStop = epoch
                bestweights = model.get_weight()
                model.set_weight(bestweights)
            
            if epoch - earlyStop > config['early_stop_epoch']:
                break
    

    if config['early_stop'] == False:
        bestweights = model.get_weight()
        model.set_weight(bestweights)
    
    return model, trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop

#This is the test method
def modelTest(model, X_test, y_test):
    test_pred = model.forward(X_test)
    test_loss = model.loss(test_pred, y_test)
    test_acc = util.calculateCorrect(test_pred, y_test)/len(y_test)

    return test_acc, test_loss