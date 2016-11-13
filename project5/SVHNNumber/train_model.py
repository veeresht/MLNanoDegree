
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SVHNNumber.generic import train_model_from_images
from SVHNNumber.models.cnn import CNN_B, LeNet5Mod, DigitConvNet

# Load SVHNDigit data

train_data_dir = 'data/final/train'
train_metadata_file = 'data/final/train/train.p'
validation_data_dir = 'data/final/validation'
validation_metadata_file = 'data/final/validation/validation.p'

# Hyperparameters selected by tuning (CNN_B)
lr = 0.03
reg_factor = 3e-6

# Hyperparameters selected by tuning (LeNet5Mod)
# lr = 0.01
# reg_factor = 2e-5

decay = 0
dropout_param = 0.05
momentum = 0.9
# lr = 5e-2
# decay = 1e-3
# reg_factor = 1e-5
# dropout_param = 0.1
# momentum = 0.9

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
                      'nb_epochs': 6,
                      'nb_train_samples': 225664,
                      'nb_validation_samples': 10000}


input_dim = (3, 64, 64)
#cnn = CNN_B(model_define_params, input_dim)
# cnn = LeNet5Mod(model_define_params, input_dim)
cnn = DigitConvNet(model_define_params, input_dim)
cnn.define(verbose=1)
history = train_model_from_images(cnn, model_train_params,
                                  train_data_dir, train_metadata_file,
                                  validation_data_dir, validation_metadata_file,
                                  verbose=1, save_to_s3=True, early_stopping=False)
