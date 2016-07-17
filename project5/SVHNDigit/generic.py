import numpy as np
import scipy.io as scipy_io
import cPickle as pickle
import warnings
import boto3
import socket
# import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, EarlyStopping, Callback, ModelCheckpoint
from keras.callbacks import LearningRateScheduler

from sklearn.cross_validation import train_test_split

from SVHNDigit.models.cnn.model import CNN_1, LeNet5Mod, HintonNet1, SermanetNet, CNN_B


def read_dataset(data_dir,
                 train_filename,
                 test_filename,
                 extra_filename,
                 val_size=5033,
                 seed=131,
                 reshape=False,
                 applyLCN=False,
                 verbose=1):

    # Load SVHN Dataset (single digits)
    train_data = scipy_io.loadmat(data_dir + '/' + train_filename)
    test_data = scipy_io.loadmat(data_dir + '/' + test_filename)
    extra_data = scipy_io.loadmat(data_dir + '/' + extra_filename)

    image_size = train_data['X'].shape[0]
    image_depth = train_data['X'].shape[2]

    train_X, train_y = train_data['X'], train_data['y']
    test_X, test_y = test_data['X'], test_data['y']
    # extra_X, extra_y = extra_data['X'][:, :, :, 0:200000], extra_data['y'][0:200000]
    extra_X, extra_y = extra_data['X'], extra_data['y']

    del extra_data

    train_X = np.concatenate((train_X, extra_X), axis=3)
    train_y = np.vstack((train_y, extra_y))

    # Reshape images from 3D to 2D
    # Useful only for fully connected neural networks
    if reshape:
        train_X = train_X.reshape((image_size *
                                   image_size *
                                   image_depth, -1)).astype(np.float32)
        test_X = test_X.reshape((image_size *
                                 image_size *
                                 image_depth, -1)).astype(np.float32)
        train_X /= 255.0
        test_X /= 255.0
    else:

        # Apply local contrast normalization (adaptive histogram equalization)
        # using 8 x 8 windows
        # if applyLCN:
        #     for i in range(train_X.shape[3]):
        #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7))
        #         frame = train_X[:, :, :, i]
        #         yuv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        #         yuv_frame[:, :, 0] = clahe.apply(yuv_frame[:, :, 0])
        #         frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
        #         frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
        #         frame[:, :, 1] = cv2.equalizeHist(frame[:, :, 1])
        #         frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])
        #         train_X[:, :, :, i] = frame

        #     for i in range(test_X.shape[3]):
        #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7))
        #         frame = test_X[:, :, :, i]
        #         yuv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        #         yuv_frame[:, :, 0] = clahe.apply(yuv_frame[:, :, 0])
        #         frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
        #         frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
        #         frame[:, :, 1] = cv2.equalizeHist(frame[:, :, 1])
        #         frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])
        #         test_X[:, :, :, i] = frame

        train_X = train_X.T
        test_X = test_X.T

        train_X = train_X.astype(np.float32)
        test_X = test_X.astype(np.float32)

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


class ModelCheckpoint2S3(Callback):

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', s3resource=None,
                 s3filename=None):

        super(ModelCheckpoint2S3, self).__init__()

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.s3resource = s3resource
        self.s3filename = s3filename

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):

        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)
                    if self.s3resource is not None:
                        # Upload to AWS S3 bucket
                        bucket = self.s3resource.Bucket('mlnd')
                        bucket.upload_file(self.s3filename, filepath)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)


def train_model_from_images(network, model_train_params,
                            train_data_dir, validation_data_dir,
                            verbose=0, tb_logs=False, early_stopping=False,
                            save_to_s3=False):

    # dimensions of our images.
    img_width, img_height = network.input_dim[1:]
    loss = model_train_params['loss']
    optimizer = model_train_params['optimizer']
    metrics = model_train_params['metrics']
    batch_size = model_train_params['batch_size']
    nb_epochs = model_train_params['nb_epochs']
    nb_train_samples = model_train_params['nb_train_samples']
    nb_validation_samples = model_train_params['nb_validation_samples']

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1.0/255)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = \
        train_datagen.flow_from_directory(train_data_dir,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          classes=['0', '1', '2', '3', '4',
                                                   '5', '6', '7', '8', '9'],
                                          class_mode='categorical')

    validation_generator = \
        test_datagen.flow_from_directory(validation_data_dir,
                                         target_size=(img_width, img_height),
                                         batch_size=batch_size,
                                         classes=['0', '1', '2', '3', '4',
                                                  '5', '6', '7', '8', '9'],
                                         class_mode='categorical')

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

    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3, verbose=0, mode='auto')
        callbacks.append(early_stopping)

    if save_to_s3:
        s3 = boto3.resource('s3')
    else:
        s3 = None

    def scheduler(epoch):
        if epoch == 1 or epoch == 2:
            network.model.lr.set_value(network.lr.get_value() * 0.5)
        return network.model.lr.get_value()

    change_lr = LearningRateScheduler(scheduler)

    callbacks.append(change_lr)

    model_checkpoint_cb = ModelCheckpoint2S3(filepath=network.name + '.h5',
                                             monitor='val_loss',
                                             save_best_only=True,
                                             verbose=0,
                                             s3resource=s3,
                                             s3filename=network.name + '.h5')

    callbacks.append(model_checkpoint_cb)

    history = network.model.fit_generator(train_generator,
                                          samples_per_epoch=nb_train_samples,
                                          nb_epoch=nb_epochs,
                                          validation_data=validation_generator,
                                          nb_val_samples=nb_validation_samples,
                                          verbose=1)

    return history


def train_model(network, model_train_params,
                train_X, train_y,
                val_X, val_y,
                verbose=0,
                tb_logs=False,
                early_stopping=False,
                save_to_s3=False):

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

    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3, verbose=0, mode='auto')
        callbacks.append(early_stopping)

    if save_to_s3:
        s3 = boto3.resource('s3')
    else:
        s3 = None

    model_checkpoint_cb = ModelCheckpoint2S3(filepath=network.name + '.h5',
                                             monitor='val_loss',
                                             save_best_only=True,
                                             verbose=0,
                                             s3resource=s3,
                                             s3filename=network.name + '.h5')

    callbacks.append(model_checkpoint_cb)

    history = network.model.fit(train_X, train_y,
                                batch_size=batch_size,
                                nb_epoch=nb_epochs,
                                callbacks=callbacks,
                                validation_split=0.1,
                                verbose=verbose)

    return history


def build_tune_model_from_images(model_name, model_tune_params,
                                 model_train_params,
                                 model_define_params,
                                 train_data_dir,
                                 validation_data_dir,
                                 num_iters,
                                 verbose=1):

    s3 = boto3.resource('s3')

    print "Tuning", model_name, "..."

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

        input_dim = (3, 32, 32)
        if model_name == 'HintonNet1':
            cnn = HintonNet1(model_define_params, input_dim)
        elif model_name == 'SermanetNet':
            cnn = SermanetNet(model_define_params, input_dim)
        elif model_name == 'LeNet5Mod':
            cnn = LeNet5Mod(model_define_params, input_dim)
        elif model_name == 'CNN_B':
            cnn = CNN_B(model_define_params, input_dim)
        cnn.define(verbose=0)
        history = train_model_from_images(cnn, model_train_params,
                                          train_data_dir,
                                          validation_data_dir,
                                          verbose=verbose, save_to_s3=True,
                                          early_stopping=True)

        data_store = (model_define_params, model_train_params,
                      history.history, history.params)

        data_store_file_name = cnn.name + '_tuning.p'
        data_store_file = open(data_store_file_name, 'a+')
        pickle.dump(data_store, data_store_file)
        data_store_file.close()

        key_file_name = cnn.name + '_tuning_' + socket.gethostname() + '.p'
        # Upload to AWS S3 bucket
        bucket = s3.Bucket('mlnd')
        bucket.upload_file(data_store_file_name, key_file_name)
