
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SVHNDigit.generic import read_dataset, train_model
from SVHNDigit.models.cnn.model import LeNet5Mod

# Load SVHNDigit data

data_dir = 'data'
train_filename = 'train_32x32.mat'
test_filename = 'test_32x32.mat'

print "Loading SVHN Digit Dataset ..."
train_X, train_y, val_X, val_y, test_X, test_y = \
    read_dataset(data_dir, train_filename, test_filename,
                 val_size=20132, reshape=False)

# num_samples = 2560
# train_X_small = train_X[0:num_samples, :, :, :]
# train_y_small = train_y[0:num_samples]

# Hyperparameters selected by tuning
lr = 0.08
decay = 3e-3
reg_factor = 5e-6
dropout_param = 0.1
momentum = 0.8

model_define_params = {'reg_factor': reg_factor,
                       'init': 'glorot_normal',
                       'use_dropout': True,
                       'dropout_param': dropout_param,
                       'use_batchnorm': True}

model_train_params = {'loss': 'categorical_crossentropy',
                      'optimizer': 'sgd',
                      'lr': lr,
                      'momentum': momentum,
                      'decay': decay,
                      'nesterov': True,
                      'metrics': ['accuracy'],
                      'batch_size': 256,
                      'nb_epochs': 20}


input_dim = train_X.shape[1:]
cnn = LeNet5Mod(model_define_params, input_dim)
cnn.define(verbose=0)
history = train_model(cnn, model_train_params,
                      train_X, train_y,
                      val_X, val_y,
                      verbose=1)
