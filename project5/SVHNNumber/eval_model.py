# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import boto3
import cPickle as pickle

from SVHNNumber.generic import train_model_from_images, eval_model_from_images
from SVHNNumber.models.cnn import LeNet5Mod, DigitConvNet, VGGNetMod_1

# Load SVHNDigit data

train_data_dir = 'data/final/train'
train_metadata_file = 'data/final/train/train.p'
validation_data_dir = 'data/final/validation'
validation_metadata_file = 'data/final/validation/validation.p'
test_data_dir = 'data/final/test'
test_metadata_file = 'data/final/test/test.p'

# Hyperparameters selected by tuning (CNN_B)
lr = 0.005
reg_factor = 1e-5

eval_dir_str = sys.argv[1]

if eval_dir_str == 'train':
    eval_dir = train_data_dir
    eval_metadata_file = train_metadata_file
    batch_size = 128
    nb_test_samples = 225664
elif eval_dir_str == 'val':
    eval_dir = validation_data_dir
    eval_metadata_file = validation_metadata_file
    batch_size = 100
    nb_test_samples = 10000
elif eval_dir_str == 'test':
    eval_dir = test_data_dir
    eval_metadata_file = test_metadata_file
    batch_size = 36
    nb_test_samples = 13068

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
                      'batch_size': batch_size,
                      'nb_epochs': 3,
                      #'nb_train_samples': 1024,
                      'nb_train_samples': 225664,
                      'nb_validation_samples': 10000,
                      'nb_test_samples': nb_test_samples}
                      #'nb_test_samples': 13068}
                      #'nb_validation_samples': 1024}


input_dim = (3, 64, 64)
cnn = VGGNetMod_1(model_define_params, input_dim)
#cnn = CNN_B(model_define_params, input_dim)
#cnn = LeNet5Mod(model_define_params, input_dim)
# cnn = DigitConvNet(model_define_params, input_dim)
cnn.define(verbose=0)
cnn.model.load_weights('trained_model_info/VGGNetMod_1/1/VGGNetMod_1.02-0.77.hdf5')

results = eval_model_from_images(cnn, model_train_params, train_data_dir, train_metadata_file,
                                 eval_dir, eval_metadata_file,
                                 verbose=0)

print "Accuracy(%): ", results['total_num_correct']/float(results['total_num_samples'])
print "Length Accuracy(%): ", results['length_num_correct']/float(results['length_num_samples'])
print "Digit-1 Accuracy(%): ", results['digit1_num_correct']/float(results['digit1_num_samples'])
print "Digit-2 Accuracy(%): ", results['digit2_num_correct']/float(results['digit2_num_samples'])
print "Digit-3 Accuracy(%): ", results['digit3_num_correct']/float(results['digit3_num_samples'])
print "Digit-4 Accuracy(%): ", results['digit4_num_correct']/float(results['digit4_num_samples'])
print "Digit-5 Accuracy(%): ", results['digit5_num_correct']/float(results['digit5_num_samples'])


