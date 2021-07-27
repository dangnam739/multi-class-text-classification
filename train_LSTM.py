import os
from datetime import datetime
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.lstm import LSTMClassifier
from utils.dataset import TextDataset
from utils.data_utils import *
from utils.visualize import PlotResults

torch.manual_seed(1)


DATA_DIR_PATH = '/home/admin_mcn/namkd/ml_test/dataset'
TRAIN_DIR_PATH = '/home/admin_mcn/namkd/ml_test/dataset/train'
TEST_DIR_PATH = '/home/admin_mcn/namkd/ml_test/dataset/test'

words_list = np.load(os.path.join(DATA_DIR_PATH, 'words_list.npy'))
words_list = words_list.tolist()
word_vectors = np.load(os.path.join(DATA_DIR_PATH, 'word_vectors.npy'))
word_vectors = np.float32(word_vectors)

# Hyperparameters
epochs = 50
learning_rate = 0.1
batch_size = 5
n_labels = 27
max_sen_len = 200
use_gpu = torch.cuda.is_available()
use_save = True

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 1

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


if __name__ == '__main__':

    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['embedding dim'] = EMBEDDING_DIM
    param['hidden dim'] = HIDDEN_DIM
    param['sentence len'] = max_sen_len

    log_file = 'log/log_' + datetime.now().strftime("%d-%h-%m-%s") + '.txt'
    f_log = open(log_file, "w")
    for param, value in param.items():
        f_log.write("%s: %s\n" %(param, value))      

    train_df = load_data_to_csv(TRAIN_DIR_PATH, train=True)
    # print(train_df.head())
    test_df = load_data_to_csv(TEST_DIR_PATH, train=False)
    # print(test_df.head())

    trainset = TextDataset(train_df, words_list, word_vectors, max_sen_len)
    testset = TextDataset(test_df, words_list, word_vectors, max_sen_len)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(words_list), label_size=n_labels, batch_size=batch_size, use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()

    print("======================Start Training=======================")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    

    for epoch in range(epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            train_inputs, train_labels = traindata
            train_labels = torch.squeeze(train_labels, -1)

            if use_gpu:
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else: 
                train_inputs = Variable(train_inputs)

            # print('Train label: ', train_labels)

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs.t())

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()

        train_loss.append(total_loss / total)
        train_acc.append(total_acc / total)

        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0

        for iter, testdata in enumerate(test_loader):
            test_inputs, test_labels = testdata
            # print('Test label: ', test_labels)
            test_labels = torch.squeeze(test_labels, -1)
            # print('Test label: ', test_labels)

            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
            else: 
                test_inputs = Variable(test_inputs)

            # print('Test label: ', test_labels)

            model.zero_grad()
            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs.t())

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.item()

        test_loss.append(total_loss / total)
        test_acc.append(total_acc / total)


        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, epochs, train_loss[epoch], test_loss[epoch], train_acc[epoch], test_acc[epoch]))

        # write to log file
        f_log.write('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f \n'
              % (epoch, epochs, train_loss[epoch], test_loss[epoch], train_acc[epoch], test_acc[epoch]))     
                
        #save model
        # save_path = f'weights/model_{epoch}.pth'
        # torch.save(model.state_dict(), save_path)

        
    result = {}
    result['train loss'] = train_loss
    result['test loss'] = test_loss
    result['train acc'] = train_acc
    result['test acc'] = test_acc
    result['param'] = 
    f_log.close()

    #plot results
    PlotResults(result, save=True)
    
    if use_save:
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)