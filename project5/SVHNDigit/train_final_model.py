
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SVHNDigit.generic import train_model_from_images
from SVHNDigit.models.cnn.model import LeNet5Mod, HintonNet1, SermanetNet, CNN_B

# Load SVHNDigit data

train_data_dir = 'data/imgs/train'
validation_data_dir = 'data/imgs/validation'

# num_samples = 2560
# train_X_small = train_X[0:num_samples, :, :, :]
# train_y_small = train_y[0:num_samples]

# Hyperparameters selected by tuning
# lr = 0.08
# decay = 3e-3
# reg_factor = 5e-6
# dropout_param = 0.1
# momentum = 0.8

lr = 5e-2
decay = 1e-3
reg_factor = 1e-5
dropout_param = 0.1
momentum = 0.9

model_define_params = {'reg_factor': reg_factor,
                       'init': 'glorot_normal',
                       'use_dropout': False,
                       'dropout_param': dropout_param,
                       'use_batchnorm': True}

model_train_params = {'loss': 'categorical_crossentropy',
                      'optimizer': 'sgd',
                      'lr': lr,
                      'momentum': momentum,
                      'decay': decay,
                      'nesterov': True,
                      'metrics': ['accuracy'],
                      'batch_size': 128,
                      'nb_epochs': 20,
                      'nb_train_samples': 99712,
                      'nb_validation_samples': 6000}


input_dim = (3, 32, 32)
cnn = CNN_B(model_define_params, input_dim)
# cnn = LeNet5Mod(model_define_params, input_dim)
#cnn = SermanetNet(model_define_params, input_dim)
cnn.define(verbose=1)
history = train_model_from_images(cnn, model_train_params,
                                  train_data_dir, validation_data_dir,
                                  verbose=1)
