import numpy as np
import scipy.io as scipy_io
import skimage.io as skimage_io
import os

from sklearn.cross_validation import train_test_split


def gen_train_val_test_images(data_dir, seed=131):

    # Load SVHN Dataset (single digits)
    train_data = scipy_io.loadmat(data_dir + '/train_32x32.mat')
    test_data = scipy_io.loadmat(data_dir + '/test_32x32.mat')
    extra_data = scipy_io.loadmat(data_dir + '/extra_32x32.mat')

    train_X, train_y = train_data['X'], train_data['y']
    test_X, test_y = test_data['X'], test_data['y']
    extra_X, extra_y = extra_data['X'], extra_data['y']

    train_y = train_y.squeeze()
    test_y = test_y.squeeze()
    extra_y = extra_y.squeeze()

    # Change labels for '0' digit from 10 to 0
    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0
    extra_y[extra_y == 10] = 0

    del extra_data

    num_classes = 10

    train_val_sample_idxs = np.array([], int)
    for i in range(num_classes):
        class_idxs = np.arange(len(train_y))[train_y == i]
        sel_class_idxs = np.random.choice(class_idxs, size=400)
        train_val_sample_idxs = np.concatenate((train_val_sample_idxs,
                                                sel_class_idxs))
    not_train_val_sample_idxs = np.setdiff1d(np.arange(len(train_y)),
                                             train_val_sample_idxs)

    val_X = train_X[:, :, :, train_val_sample_idxs]
    val_y = train_y[train_val_sample_idxs]

    extra_val_sample_idxs = np.array([], int)
    for i in range(num_classes):
        class_idxs = np.arange(len(extra_y))[extra_y == i]
        sel_class_idxs = np.random.choice(class_idxs, size=200)
        extra_val_sample_idxs = np.concatenate((extra_val_sample_idxs,
                                                sel_class_idxs))
    not_extra_val_sample_idxs = np.setdiff1d(np.arange(len(extra_y)),
                                             extra_val_sample_idxs)

    val_X = np.concatenate((val_X, extra_X[:, :, :, extra_val_sample_idxs]), axis=3)
    val_y = np.hstack((val_y, extra_y[extra_val_sample_idxs]))

    train_X = np.concatenate((train_X[:, :, :, not_train_val_sample_idxs],
                              extra_X[:, :, :, not_extra_val_sample_idxs]), axis=3)
    train_y = np.hstack((train_y[not_train_val_sample_idxs],
                         extra_y[not_extra_val_sample_idxs]))

    # Create directories and save images
    train_dir = data_dir + '/imgs/train'
    test_dir = data_dir + '/imgs/test'
    validation_dir = data_dir + '/imgs/validation'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for i in range(num_classes):
        if not os.path.exists(train_dir + '/' + str(i)):
            os.makedirs(train_dir + '/' + str(i))

        if not os.path.exists(validation_dir + '/' + str(i)):
            os.makedirs(validation_dir + '/' + str(i))

        if not os.path.exists(test_dir + '/' + str(i)):
            os.makedirs(test_dir + '/' + str(i))

    print "Creating train images ... "
    for i in range(len(train_y)):
        filename = train_dir + '/' + str(train_y[i]) + '/' + str(i) + '.png'
        skimage_io.imsave(filename, train_X[:, :, :, i])

    print "Creating validation images ... "
    for i in range(len(val_y)):
        filename = validation_dir + '/' + str(val_y[i]) + '/' + str(i) + '.png'
        skimage_io.imsave(filename, val_X[:, :, :, i])

    print "Creating test images ... "
    for i in range(len(test_y)):
        filename = test_dir + '/' + str(test_y[i]) + '/' + str(i) + '.png'
        skimage_io.imsave(filename, test_X[:, :, :, i])
