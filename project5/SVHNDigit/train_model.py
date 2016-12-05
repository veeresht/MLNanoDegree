
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cPickle as pickle
import time
import boto3

from SVHNDigit.generic import train_model_from_images
from SVHNDigit.models.cnn import (
    LeNet5Mod,
    HintonNet1,
    SermanetNet,
    VGGNetMod_1,
    InceptionNet)

# Load SVHNDigit data

train_data_dir = 'data/imgs/train'
validation_data_dir = 'data/imgs/validation'

# Hyperparameters selected by tuning (LeNet5Mod)
lr = 0.01
reg_factor = 1e-5

# Hyperparameters selected by tuning (CNN_B)
# lr = 0.03
# reg_factor = 3e-6

decay = 0
dropout_param = 0.1
momentum = 0.9
# lr = 5e-2
# decay = 1e-3
# reg_factor = 1e-5
# dropout_param = 0.1
# momentum = 0.9

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
                      'batch_size': 128,
                      'nb_epochs': 3,
                      'nb_train_samples': 99712 * 6,
                      #'nb_train_samples': 320,
                      'nb_validation_samples': 6000}


input_dim = (3, 32, 32)
cnn = VGGNetMod_1(model_define_params, input_dim)
# cnn = LeNet5Mod(model_define_params, input_dim)
# cnn = InceptionNet(model_define_params, input_dim)
#cnn = SermanetNet(model_define_params, input_dim)
cnn.define(verbose=1)
history = train_model_from_images(cnn, model_train_params,
                                  train_data_dir, validation_data_dir,
                                  verbose=1, save_to_s3=False, tb_logs=False,
                                  csv_log=True, early_stopping=True)

data_store = (model_define_params, model_train_params, history.history, history.params)

data_store_file_name = cnn.name + '_log_' + time.strftime("%x").replace("/", "_") + '.p'
data_store_file = open(data_store_file_name, 'a+')
pickle.dump(data_store, data_store_file)
data_store_file.close()

s3 = boto3.resource('s3')
if s3 is not None:
    # Upload to AWS S3 bucket
    bucket = s3.Bucket('mlnd')
    bucket.upload_file(data_store_file_name, data_store_file_name)
