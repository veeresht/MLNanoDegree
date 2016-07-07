import numpy as np
import scipy.io as scipy_io
import cPickle as pickle
import boto3
import socket
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, EarlyStopping, Callback

from sklearn.cross_validation import train_test_split

from SVHNDigit.models.cnn.model import CNN_1, LeNet5Mod


def read_dataset(data_dir,
                 train_filename,
                 test_filename,
                 val_size=5033,
                 seed=131,
                 reshape=False,
                 verbose=1):

    # Load SVHN Dataset (single digits)
    train_data = scipy_io.loadmat(data_dir + '/' + train_filename)
    test_data = scipy_io.loadmat(data_dir + '/' + test_filename)

    image_size = train_data['X'].shape[0]
    image_depth = train_data['X'].shape[2]

    train_X, train_y = train_data['X'], train_data['y']
    test_X, test_y = test_data['X'], test_data['y']

    # Reshape images from 3D to 2D
    # Useful only for fully connected neural networks
    if reshape:
        train_X = train_X.reshape((image_size *
                                   image_size *
                                   image_depth, -1)).astype(np.float32)
        test_X = test_X.reshape((image_size *
                                 image_size *
                                 image_depth, -1)).astype(np.float32)
    else:
        train_X = train_X.astype(np.float32)
        test_X = test_X.astype(np.float32)

    train_X = train_X.T
    test_X = test_X.T

    train_X /= 255.0
    test_X /= 255.0
    # train_X = (train_X - np.mean(train_X, axis=0))/np.std(train_X, axis=0)
    # test_X = (test_X - np.mean(test_X, axis=0))/np.std(test_X, axis=0)

    train_y = train_y.squeeze()
    test_y = test_y.squeeze()

    # Change labels for '0' digit from 10 to 0
    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0

    # Use stratified split for training and validation set
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,
                                                      test_size=val_size,
                                                      random_state=seed,
                                                      stratify=train_y)

    # Convert target variable into categorical form
    nb_classes = 10
    train_y = np_utils.to_categorical(train_y, nb_classes)
    val_y = np_utils.to_categorical(val_y, nb_classes)
    test_y = np_utils.to_categorical(test_y, nb_classes)

    if verbose == 1:
        print "Training data shape: ", train_X.shape, train_y.shape
        print "Validation data shape: ", val_X.shape, val_y.shape
        print "Test data shape: ", test_X.shape, test_y.shape

    return train_X, train_y, val_X, val_y, test_X, test_y


class EarlyBatchTermination(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In 'min' mode,
            training will stop when the quantity
            monitored has stopped decreasing; in 'max'
            mode it will stop when the quantity
            monitored has stopped increasing.
    '''

    def __init__(self, monitor='loss', interval=10, verbose=0):
        super(EarlyBatchTermination, self).__init__()

        self.monitor = monitor
        self.interval = interval
        self.verbose = verbose
        self.batch_count = 0

    def on_train_begin(self, logs={}):
        self.prev_batch_loss = 1000
        self.current_batch_loss = 0

    def on_batch_end(self, batch, logs={}):

        if self.batch_count % self.interval == 0:
            self.current_batch_loss = logs.get('loss')
            if (((abs(self.prev_batch_loss - self.current_batch_loss)) /
                 (self.prev_batch_loss + 1.0e-8)) < 0.1):
                if self.verbose > 0:
                    print('Early Batch Termination')
                self.model.stop_training = True
            self.prev_batch_loss = self.current_batch_loss
        self.batch_count += 1

    def on_epoch_end(self, epoch, logs={}):
        self.prev_batch_loss = 1000
        self.current_batch_loss = 0
        self.batch_count = 0


def train_model(network, model_train_params,
                train_X, train_y,
                val_X, val_y,
                verbose=0,
                tb_logs=False):

    loss = model_train_params['loss']
    optimizer = model_train_params['optimizer']
    metrics = model_train_params['metrics']
    batch_size = model_train_params['batch_size']
    nb_epochs = model_train_params['nb_epochs']

    if optimizer == 'sgd':
        optimizer = SGD(lr=model_train_params['lr'],
                        momentum=model_train_params['momentum'],
                        decay=model_train_params['decay'],
                        nesterov=model_train_params['nesterov'])
    elif optimizer == 'adam':
        optimizer = Adam(lr=model_train_params['lr'])

    network.model.compile(loss=loss,
                          metrics=metrics,
                          optimizer=optimizer)

    callbacks = []
    if tb_logs:
        tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        callbacks = [tb_callback]

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=1, verbose=0, mode='auto')
    callbacks.append(early_stopping)

    history = network.model.fit(train_X, train_y,
                                batch_size=batch_size,
                                nb_epoch=nb_epochs,
                                callbacks=callbacks,
                                validation_split=0.1,
                                verbose=verbose)

    return history


def build_tune_model(model_tune_params,
                     model_train_params,
                     model_define_params,
                     train_X, train_y,
                     val_X, val_y,
                     num_iters,
                     verbose=1):

    s3 = boto3.resource('s3')

    for i in range(num_iters):
        print "==========================================================="
        print "Iteration Count: ", i
        for k in model_tune_params:
            # Learning Rate and Momentum
            if k == 'lr' or k == 'decay' or k == 'momentum':
                model_train_params[k] = 10 ** \
                    np.random.uniform(
                    model_tune_params[k][0],
                    model_tune_params[k][1])
                if verbose > 0:
                    print k, ':', model_train_params[k]

            # Regularization Factor and Dropout Parameter
            if k == 'reg_factor' or k == 'dropout_param':
                model_define_params[k] = 10 ** \
                    np.random.uniform(
                    model_tune_params[k][0],
                    model_tune_params[k][1])
                if verbose > 0:
                    print k, ':', model_define_params[k]

        input_dim = train_X.shape[1:]
        cnn = LeNet5Mod(model_define_params, input_dim)
        cnn.define(verbose=0)
        history = train_model(cnn, model_train_params,
                              train_X, train_y,
                              val_X, val_y,
                              verbose=verbose)

        score, acc = cnn.model.evaluate(val_X, val_y, batch_size=256,
                                        verbose=verbose)
        if verbose == 1:
            print 'Validation Loss: %0.4f' % (score)
            print 'Validation Accuracy: %0.4f' % (acc)

        data_store = (model_define_params, model_train_params,
                      history.history, history.params, score, acc)

        data_store_file_name = cnn.name + '_tuning.p'
        data_store_file = open(data_store_file_name, 'a+')
        pickle.dump(data_store, data_store_file)
        data_store_file.close()


        key_file_name = cnn.name + '_tuning_' + socket.gethostname() + '.p'
        # Upload to AWS S3 bucket
        bucket = s3.Bucket('mlnd')
        bucket.upload_file(data_store_file_name, key_file_name)
