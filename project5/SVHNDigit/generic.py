import numpy as np
import scipy.io as scipy_io

from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard

from sklearn.cross_validation import train_test_split


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


def train_model(model, model_train_params,
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
                        decay=0.0,
                        nesterov=model_train_params['nesterov'])
    elif optimizer == 'adam':
        optimizer = Adam(lr=model_train_params['lr'])

    model.compile(loss=loss,
                  metrics=metrics,
                  optimizer=optimizer)

    callbacks = []
    if tb_logs:
        tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        callbacks = [tb_callback]

    model.fit(train_X, train_y,
              batch_size=batch_size,
              nb_epoch=nb_epochs,
              show_accuracy=False,
              callbacks=callbacks,
              verbose=verbose)

    score, acc = model.evaluate(val_X, val_y, verbose=verbose)
    if verbose == 1:
        print('Validation accuracy:', acc)
