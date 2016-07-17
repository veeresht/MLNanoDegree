
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SVHNDigit.generic import build_tune_model_from_images

# Load SVHNDigit data
train_data_dir = 'data/imgs/train'
validation_data_dir = 'data/imgs/validation'


# num_samples = 2560
# train_X_small = train_X[0:num_samples, :, :, :]
# train_y_small = train_y[0:num_samples]

lr = 1e-2
decay = 0
reg_factor = 2e-6
dropout_param = 0.05
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
                      'nb_epochs': 3,
                      'nb_train_samples': 99712 * 6,
                      'nb_validation_samples': 6000}

model_tune_params = {  # [1e-3, 1e-1]
                       'lr': [-3, -1],
                       # [1e-4, 1e-2]
                       #'decay': [-5, -3],
                       # [1e-6, 1e-5]
                       'reg_factor': [-6, -3]}
                       # [0.01, 0.2]
                       #'dropout_param': [-2, -0.7]}
                       # [0.8, 0.99]
                       #'momentum': [-0.1, -0.004]}

num_iters = 10
model_name = 'LeNet5Mod'
build_tune_model_from_images(model_name, model_tune_params,
                             model_train_params,
                             model_define_params,
                             train_data_dir,
                             validation_data_dir,
                             num_iters,
                             verbose=1)
