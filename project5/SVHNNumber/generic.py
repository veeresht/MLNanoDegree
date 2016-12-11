import numpy as np
import cPickle as pickle
import warnings
import boto3
import socket
import time

import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, EarlyStopping, Callback, CSVLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler

from SVHNNumber.imagedatagen import SVHNImageDataGenerator
from SVHNNumber.models.cnn import VGGNetMod_1

# def seqdet_metric(y_true, y_pred):

#     print y_true
#     print y_pred

#     length_log_probs = K.log(y_pred[0])
#     digit_log_probs = K.log(K.stack(y_pred[1:], axis=0))


#     probs = y_pred.eval()
#     y = y_true.eval()
#     length_log_probs = np.log(probs[0])
#     digit_log_probs = np.log(np.dstack(probs[1:]))
#     preds = np.argmax(digit_log_probs, axis=1)
#     max_log_probs = np.max(digit_log_probs, axis=1)
#     max_log_probs = np.hstack((np.zeros([batch_size, 1]),
#                                max_log_probs, np.zeros([batch_size, 1])))
#     sum_max_log_probs = np.cumsum(max_log_probs, axis=1)
#     seq_log_probs = sum_max_log_probs + length_log_probs
#     seq_length_preds = np.argmax(seq_log_probs, axis=1)
#     for idx in range(probs[0].shape[0]):
#         preds[idx, seq_length_preds[idx]:] = 0
#     length_comp = seq_length_preds == np.argmax(y[0], axis=1)
#     digit1_comp = preds[:, 0] == np.argmax(y[1], axis=1)
#     digit2_comp = preds[:, 1] == np.argmax(y[2], axis=1)
#     digit3_comp = preds[:, 2] == np.argmax(y[3], axis=1)
#     digit4_comp = preds[:, 3] == np.argmax(y[4], axis=1)
#     digit5_comp = preds[:, 4] == np.argmax(y[5], axis=1)
#     comp_stack = np.vstack((length_comp, digit1_comp, digit2_comp,
#                             digit3_comp, digit4_comp, digit5_comp)).T
#     comp = np.all(comp_stack, axis=1)
#     num_correct = np.sum(comp)
#     total_num_correct += num_correct
#     length_num_correct += np.sum(length_comp)
#     digit1_num_correct += np.sum(digit1_comp)
#     digit2_num_correct += np.sum(digit2_comp)
#     digit3_num_correct += np.sum(digit3_comp)
#     digit4_num_correct += np.sum(digit4_comp)
#     digit5_num_correct += np.sum(digit5_comp)

#     acc = total_num_correct / float(probs[0].shape[0])

#     return K.convert_to_tensor(acc.eval())

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


class LossHistory(Callback):

    def __init__(self, filepath, s3resource=None,
                 s3filename=None):
        self.filepath = filepath
        self.s3resource = s3resource
        self.s3filename = s3filename

    def on_train_begin(self, logs={}):
        self.losses = []
        self.length_acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.length_acc.append(logs.get('length_acc'))

    def on_epoch_end(self, epoch, logs={}):
        np.savetxt(self.filepath, (self.losses, self.length_acc), delimiter=",")
        if self.s3resource is not None:
            # Upload to AWS S3 bucket
            bucket = self.s3resource.Bucket('mlnd')
            bucket.upload_file(self.s3filename, self.filepath)

    def on_train_end(self, logs={}):
        np.savetxt(self.filepath, (self.losses, self.length_acc), delimiter=",")
        if self.s3resource is not None:
            # Upload to AWS S3 bucket
            bucket = self.s3resource.Bucket('mlnd')
            bucket.upload_file(self.s3filename, self.filepath)


def train_model_from_images(network, model_train_params, model_define_params,
                            train_data_dir, train_metadata_file,
                            validation_data_dir, validation_metadata_file,
                            verbose=0, tb_logs=False, csv_log=False,
                            early_stopping=False,
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

    train_datagen = SVHNImageDataGenerator(rescale=1.0/255)

    train_generator = \
        train_datagen.flow_from_directory(train_data_dir, train_metadata_file,
                                          target_size=(img_width, img_height),
                                          batch_size=100, seed=131)

    print "Fetching images for feature wise centering ..."
    total_samples = 5000
    image_set = np.empty([0, 3, img_width, img_height])
    #label_set = np.empty([0, 10])
    for i in range(total_samples/batch_size):
        (X, y) = train_generator.next()
        image_set = np.concatenate((image_set, X['input_image']), axis=0)
        #label_set = np.concatenate((label_set, y), axis=0)

    print "Fitting image set for feature wise centering ..."
    train_datagen = SVHNImageDataGenerator(rescale=1.0/255,
                                           samplewise_center=False,
                                           featurewise_center=True)

    train_datagen.fit(image_set)

    val_datagen = SVHNImageDataGenerator(rescale=1.0/255,
                                          samplewise_center=False,
                                          featurewise_center=True,)

    val_datagen.fit(image_set)

    train_generator = \
        train_datagen.flow_from_directory(train_data_dir, train_metadata_file,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size)

    validation_generator = \
        val_datagen.flow_from_directory(validation_data_dir, validation_metadata_file,
                                         target_size=(img_width, img_height),
                                         batch_size=batch_size)

    if optimizer == 'sgd':
        optimizer = SGD(lr=model_train_params['lr'],
                        momentum=model_train_params['momentum'],
                        decay=model_train_params['decay'],
                        nesterov=model_train_params['nesterov'])
    elif optimizer == 'adam':
        optimizer = Adam(lr=model_train_params['lr'])

    network.model.compile(loss={'digit1': 'categorical_crossentropy',
                                'digit2': 'categorical_crossentropy',
                                'digit3': 'categorical_crossentropy',
                                'digit4': 'categorical_crossentropy',
                                'digit5': 'categorical_crossentropy',
                                'length': 'categorical_crossentropy'},
                          metrics=metrics,
                          optimizer=optimizer)

    callbacks = []
    if tb_logs:
        tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        callbacks = [tb_callback]

    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_acc',
                                        patience=2, min_delta=0.005,
                                        verbose=0, mode='auto')
        callbacks.append(early_stopping)

    if csv_log:
        csv_filename = (network.name + '_lr_' + str(model_train_params['lr']) +
                       '_l2weightdecay_' + str(model_train_params['decay']) + '.csv')
        csv_logger_callback = CSVLogger(filename=csv_filename, append=True)
        callbacks.append(csv_logger_callback)

    if save_to_s3:
        s3 = boto3.resource('s3')
    else:
        s3 = None

    def scheduler(epoch):
        init_lr = model_train_params['lr']
        return init_lr * (0.9**epoch)

    change_lr = LearningRateScheduler(scheduler)
    callbacks.append(change_lr)

    # model_checkpoint_cb = ModelCheckpoint2S3(filepath=network.name + '.h5',
    #                                          monitor='val_loss',
    #                                          save_best_only=True,
    #                                          verbose=0,
    #                                          s3resource=s3,
    #                                          s3filename=network.name + '.h5')
    # callbacks.append(model_checkpoint_cb)

    model_checkpoint_cb = ModelCheckpoint(filepath=network.name + '.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss')
    callbacks.append(model_checkpoint_cb)

    filename = network.name + '_lr_' + str(model_train_params['lr']) + '_l2weightdecay_' + str(model_define_params['reg_factor']) + '_' + time.strftime("%x").replace("/", "_") + "_trainloss.csv"
    loss_history = LossHistory(filepath=filename,
                                s3resource=s3, s3filename=filename)
    callbacks.append(loss_history)

    history = network.model.fit_generator(train_generator,
                                          samples_per_epoch=nb_train_samples,
                                          nb_epoch=nb_epochs,
                                          validation_data=validation_generator,
                                          nb_val_samples=nb_validation_samples,
                                          callbacks=callbacks,
                                          verbose=1)

    return history

def eval_model_from_images(network, model_train_params,
                           train_data_dir, train_metadata_file,
                           test_data_dir, test_metadata_file, verbose=0):

    # dimensions of our images.
    img_width, img_height = network.input_dim[1:]
    optimizer = model_train_params['optimizer']
    metrics = model_train_params['metrics']
    batch_size = model_train_params['batch_size']
    nb_test_samples = model_train_params['nb_test_samples']

    train_datagen = SVHNImageDataGenerator(rescale=1.0/255)

    train_generator = \
        train_datagen.flow_from_directory(train_data_dir, train_metadata_file,
                                          target_size=(img_width, img_height),
                                          batch_size=100, seed=131)

    print "Fetching images for feature wise centering ..."
    total_samples = 5000
    image_set = np.empty([0, 3, img_width, img_height])
    #label_set = np.empty([0, 10])
    for i in range(total_samples/batch_size):
        (X, y) = train_generator.next()
        image_set = np.concatenate((image_set, X['input_image']), axis=0)
        #label_set = np.concatenate((label_set, y), axis=0)

    print "Fitting image set for feature wise centering ..."

    eval_datagen = SVHNImageDataGenerator(rescale=1.0/255,
                                          samplewise_center=False,
                                          featurewise_center=True)

    eval_datagen.fit(image_set)

    eval_generator = \
        eval_datagen.flow_from_directory(test_data_dir, test_metadata_file,
                                         target_size=(img_width, img_height),
                                         batch_size=batch_size, shuffle=False)

    if optimizer == 'sgd':
        optimizer = SGD(lr=model_train_params['lr'],
                        momentum=model_train_params['momentum'],
                        decay=model_train_params['decay'],
                        nesterov=model_train_params['nesterov'])
    elif optimizer == 'adam':
        optimizer = Adam(lr=model_train_params['lr'])

    network.model.compile(loss={'digit1': 'categorical_crossentropy',
                                'digit2': 'categorical_crossentropy',
                                'digit3': 'categorical_crossentropy',
                                'digit4': 'categorical_crossentropy',
                                'digit5': 'categorical_crossentropy',
                                'length': 'categorical_crossentropy'},
                          metrics=metrics,
                          optimizer=optimizer)

    total_num_correct = 0
    length_num_correct = 0
    digit1_num_correct = 0
    digit2_num_correct = 0
    digit3_num_correct = 0
    digit4_num_correct = 0
    digit5_num_correct = 0

    digit1_num_samples = 0
    digit2_num_samples = 0
    digit3_num_samples = 0
    digit4_num_samples = 0
    digit5_num_samples = 0
    batch_count = 0

    for (X, y) in eval_generator:
        probs = network.model.predict_on_batch(X)
        length_log_probs = np.log(probs[0])
        digit_log_probs = np.log(np.dstack(probs[1:]))
        preds = np.argmax(digit_log_probs, axis=1)
        max_log_probs = np.max(digit_log_probs, axis=1)
        max_log_probs = np.hstack((np.zeros([batch_size, 1]), max_log_probs,
                                                np.zeros([batch_size, 1])))
        sum_max_log_probs = np.cumsum(max_log_probs, axis=1)
        seq_log_probs = sum_max_log_probs + length_log_probs
        seq_length_preds = np.argmax(seq_log_probs, axis=1)
        for idx in range(batch_size):
            preds[idx, seq_length_preds[idx]:] = 0
        length_comp = seq_length_preds == np.argmax(y['length'], axis=1)
        digit1_comp = preds[:, 0] == np.argmax(y['digit1'], axis=1)
        digit2_comp = preds[:, 1] == np.argmax(y['digit2'], axis=1)
        digit3_comp = preds[:, 2] == np.argmax(y['digit3'], axis=1)
        digit4_comp = preds[:, 3] == np.argmax(y['digit4'], axis=1)
        digit5_comp = preds[:, 4] == np.argmax(y['digit5'], axis=1)

        comp_stack = np.vstack((length_comp, digit1_comp, digit2_comp,
                                digit3_comp, digit4_comp, digit5_comp)).T
        comp = np.all(comp_stack, axis=1)
        num_correct = np.sum(comp)
        total_num_correct += num_correct
        length_num_correct += np.sum(length_comp)

        valid_idxs = np.sum(y['digit1'], axis=1) == 1
        digit1_comp = preds[:, 0][valid_idxs] == np.argmax(y['digit1'], axis=1)[valid_idxs]
        digit1_num_correct += np.sum(digit1_comp)
        digit1_num_samples += np.sum(valid_idxs)

        valid_idxs = np.sum(y['digit2'], axis=1) == 1
        digit2_comp = preds[:, 1][valid_idxs] == np.argmax(y['digit2'], axis=1)[valid_idxs]
        digit2_num_correct += np.sum(digit2_comp)
        digit2_num_samples += np.sum(valid_idxs)

        valid_idxs = np.sum(y['digit3'], axis=1) == 1
        digit3_comp = preds[:, 2][valid_idxs] == np.argmax(y['digit3'], axis=1)[valid_idxs]
        digit3_num_correct += np.sum(digit3_comp)
        digit3_num_samples += np.sum(valid_idxs)

        valid_idxs = np.sum(y['digit4'], axis=1) == 1
        digit4_comp = preds[:, 3][valid_idxs] == np.argmax(y['digit4'], axis=1)[valid_idxs]
        digit4_num_correct += np.sum(digit4_comp)
        digit4_num_samples +=np.sum(valid_idxs)

        valid_idxs = np.sum(y['digit5'], axis=1) == 1
        digit5_comp = preds[:, 4][valid_idxs] == np.argmax(y['digit5'], axis=1)[valid_idxs]
        digit5_num_correct += np.sum(digit5_comp)
        digit5_num_samples += np.sum(valid_idxs)

        batch_count += 1
        print "# of Samples Processed: ", batch_count * batch_size, "/", nb_test_samples
        if batch_count * batch_size >= nb_test_samples:
            break

    results_dict = {'total_num_correct': total_num_correct,
                    'total_num_samples': batch_count * batch_size,
                    'length_num_correct': length_num_correct,
                    'length_num_samples': batch_count * batch_size,
                    'digit1_num_correct': digit1_num_correct,
                    'digit1_num_samples': digit1_num_samples,
                    'digit2_num_correct': digit2_num_correct,
                    'digit2_num_samples': digit2_num_samples,
                    'digit3_num_correct': digit3_num_correct,
                    'digit3_num_samples': digit3_num_samples,
                    'digit4_num_correct': digit4_num_correct,
                    'digit4_num_samples': digit4_num_samples,
                    'digit5_num_correct': digit5_num_correct,
                    'digit5_num_samples': digit5_num_samples}

    return results_dict


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


def save_bottleneck_features(network, model_train_params,
                             train_data_dir, validation_data_dir,
                             test_data_dir, verbose=0):

    # dimensions of our images.
    img_width, img_height = network.input_dim[1:]
    loss = model_train_params['loss']
    optimizer = model_train_params['optimizer']
    metrics = model_train_params['metrics']
    batch_size = model_train_params['batch_size']
    nb_epochs = model_train_params['nb_epochs']
    nb_train_samples = model_train_params['nb_train_samples']
    nb_validation_samples = model_train_params['nb_validation_samples']
    nb_test_samples = model_train_params['nb_test_samples']

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # train_generator = \
    #     train_datagen.flow_from_directory(train_data_dir,
    #                                       target_size=(img_width, img_height),
    #                                       batch_size=batch_size,
    #                                       class_mode=None,
    #                                       shuffle=False)

    # bottleneck_features_train = \
    #     network.model.predict_generator(train_generator, nb_train_samples)
    # np.save(open('data/features/' + network.name + '_bottleneck_features_train.npy', 'w'),
    #         bottleneck_features_train)

    # validation_generator = \
    #     validation_datagen.flow_from_directory(validation_data_dir,
    #                                            target_size=(img_width, img_height),
    #                                            batch_size=batch_size,
    #                                            class_mode=None,
    #                                            shuffle=False)

    # bottleneck_features_validation = \
    #     network.model.predict_generator(validation_generator, nb_validation_samples)
    # np.save(open('data/features/' + network.name + '_bottleneck_features_validation.npy', 'w'),
    #         bottleneck_features_validation)

    test_generator = \
        test_datagen.flow_from_directory(test_data_dir,
                                         target_size=(img_width, img_height),
                                         batch_size=batch_size,
                                         class_mode=None,
                                         shuffle=False)

    bottleneck_features_test = \
        network.model.predict_generator(test_generator, nb_test_samples)
    np.save(open('data/features/' + network.name + '_bottleneck_features_test.npy', 'w'),
            bottleneck_features_test)
