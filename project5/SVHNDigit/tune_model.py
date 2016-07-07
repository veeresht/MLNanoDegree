
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SVHNDigit.generic import read_dataset
from SVHNDigit.generic import build_tune_model


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

lr = 1e-2
decay = 1e-3
reg_factor = 2e-6
dropout_param = 0.05
momentum = 0.9

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
                      'nb_epochs': 2}

model_tune_params = {  # [5e-3, 1e-1]
                       'lr': [-2.3, -1],
                       # [1e-4, 1e-2]
                       'decay': [-4, -2],
                       # [1e-6, 1e-5]
                       'reg_factor': [-6, -4],
                       # [0.01, 0.2]
                       'dropout_param': [-2, -0.7],
                       # [0.8, 0.99]
                       'momentum': [-0.1, -0.004]}

num_iters = 50

build_tune_model(model_tune_params,
                 model_train_params,
                 model_define_params,
                 train_X, train_y,
                 val_X, val_y,
                 num_iters,
                 verbose=1)
