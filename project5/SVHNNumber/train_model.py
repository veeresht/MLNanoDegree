
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import boto3
import cPickle as pickle

from SVHNNumber.generic import train_model_from_images
from SVHNNumber.models.cnn import (
    LeNet5Mod,
    DigitConvNet,
    VGGNetMod_1,
    VGGNetMod_2,
    VGGNetMod_3
    )

# Load SVHNDigit data

train_data_dir = 'data/final/train'
train_metadata_file = 'data/final/train/train.p'
validation_data_dir = 'data/final/validation'
validation_metadata_file = 'data/final/validation/validation.p'

# Hyperparameters selected by tuning (CNN_B)
lr = float(sys.argv[1])
reg_factor = float(sys.argv[2])

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
                      'batch_size': 32,
                      'nb_epochs': 3,
                      #'nb_train_samples': 1024,
                      'nb_train_samples': 225664,
                      'nb_validation_samples': 10000}
                      #'nb_validation_samples': 1024}


input_dim = (3, 64, 64)
cnn = VGGNetMod_3(model_define_params, input_dim)
#cnn = CNN_B(model_define_params, input_dim)
#cnn = LeNet5Mod(model_define_params, input_dim)
# cnn = DigitConvNet(model_define_params, input_dim)
cnn.define(verbose=1)
history = train_model_from_images(cnn, model_train_params, model_define_params,
                                  train_data_dir, train_metadata_file,
                                  validation_data_dir, validation_metadata_file,
                                  verbose=1, save_to_s3=False, early_stopping=False,
                                  csv_log=True)

data_store = (model_define_params, model_train_params, history.history, history.params)

data_store_file_name = cnn.name + '_log_lr_' + str(lr) + '_l2weightdecay_' + str(reg_factor) + '_' + time.strftime("%x").replace("/", "_") + '.p'
data_store_file = open(data_store_file_name, 'a+')
pickle.dump(data_store, data_store_file)
data_store_file.close()

s3 = boto3.resource('s3')
if s3 is not None:
    # Upload to AWS S3 bucket
    bucket = s3.Bucket('mlnd')
    bucket.upload_file(data_store_file_name, data_store_file_name)
