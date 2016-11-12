
import numpy as np
import h5py
import cPickle as pkl
import os
import shutil
from skimage import io, transform


def read_process_h5(filename):
    """ Reads and processes the mat files provided in the SVHN dataset.
        Input: filename
        Ouptut: list of python dictionaries
    """

    f = h5py.File(filename, 'r')
    groups = f['digitStruct'].items()
    bbox_ds = np.array(groups[0][1]).squeeze()
    names_ds = np.array(groups[1][1]).squeeze()

    data_list = []
    num_files = bbox_ds.shape[0]
    count = 0

    for objref1, objref2 in zip(bbox_ds, names_ds):

        data_dict = {}

        # Extract image name
        names_ds = np.array(f[objref2]).squeeze()
        filename = ''.join(chr(x) for x in names_ds)
        data_dict['filename'] = filename

        #print filename

        # Extract other properties
        items1 = f[objref1].items()

        # Extract image label
        labels_ds = np.array(items1[1][1]).squeeze()
        try:
            label_vals = [int(f[ref][:][0, 0]) for ref in labels_ds]
        except TypeError:
            label_vals = [labels_ds]
        data_dict['labels'] = label_vals
        data_dict['length'] = len(label_vals)

        # Extract image height
        height_ds = np.array(items1[0][1]).squeeze()
        try:
            height_vals = [f[ref][:][0, 0] for ref in height_ds]
        except TypeError:
            height_vals = [height_ds]
        data_dict['height'] = height_vals

        # Extract image left coords
        left_ds = np.array(items1[2][1]).squeeze()
        try:
            left_vals = [f[ref][:][0, 0] for ref in left_ds]
        except TypeError:
            left_vals = [left_ds]
        data_dict['left'] = left_vals

        # Extract image top coords
        top_ds = np.array(items1[3][1]).squeeze()
        try:
            top_vals = [f[ref][:][0, 0] for ref in top_ds]
        except TypeError:
            top_vals = [top_ds]
        data_dict['top'] = top_vals

        # Extract image width
        width_ds = np.array(items1[4][1]).squeeze()
        try:
            width_vals = [f[ref][:][0, 0] for ref in width_ds]
        except TypeError:
            width_vals = [width_ds]
        data_dict['width'] = width_vals

        data_list.append(data_dict)

        count += 1
        print 'Processed: %d/%d' % (count, num_files)

    return data_list


def read_process_metadata(data_dir):
    """ Read and process digitStruct.mat files and store them as simple pickled
        list of dictionaries. """

    train_dir = data_dir + '/train/'
    extra_dir = data_dir + '/extra/'
    test_dir = data_dir + '/test/'

    print('Processing train metadata ... ')
    train_filename = train_dir + 'digitStruct.mat'
    train_data_list = read_process_h5(train_filename)
    pkl.dump(train_data_list, open(data_dir + '/metadata/train.p', 'wb'))

    print('Processing extra metadata ... ')
    extra_filename = extra_dir + 'digitStruct.mat'
    extra_data_list = read_process_h5(extra_filename)
    pkl.dump(extra_data_list, open(data_dir + '/metadata/extra.p', 'wb'))

    print('Processing test metadata ... ')
    test_filename = test_dir + 'digitStruct.mat'
    test_data_list = read_process_h5(test_filename)
    pkl.dump(test_data_list, open(data_dir + '/metadata/test.p', 'wb'))


def resize_crop_image_file(read_dir, write_dir, data_dict):
    """ Helper function to crop and resize images to 64 x 64 pixels. """

    filename = read_dir + data_dict['filename']
    im = io.imread(filename)

    top = np.min(data_dict['top'])
    bottom = np.max(np.array(data_dict['top']) + np.array(data_dict['height']))
    left = np.min(data_dict['left'])
    width = np.sum(data_dict['width'])

    top = top - int(0.15 * (bottom - top))
    if top < 0:
        top = 0
    bottom = bottom + int(0.15 * (bottom - top))
    left = left - int(0.15 * width)
    if left < 0:
        left = 0
    width = width + int(0.3 * width)

    crop_im = im[top:bottom, left:left+width]
    #print top, height, left, width
    resized_im = transform.resize(crop_im, (64, 64))

    target_filename = write_dir + data_dict['filename']
    io.imsave(target_filename, resized_im)


def resize_crop_images(data_dir, verbose=1):
    """ Function to crop and resize images to 64 x 64 pixels
        in the train, extra and test folders. """

    # num_train_files = 33402
    # num_extra_files = 202353
    # num_test_files = 13068

    print 'Reading and processing train files ...'
    train_metadata_list = pkl.load(open(data_dir + '/metadata/train.p', 'rb'))
    read_dir = data_dir + '/train/'
    write_dir = data_dir + '/processed/train/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    num_train_files = len(train_metadata_list)
    for i, data_dict in enumerate(train_metadata_list):
        resize_crop_image_file(read_dir, write_dir, data_dict)
        if verbose == 1:
            print 'Processed files: %d/%d' % (i, num_train_files)

    print 'Reading and processing extra files ...'
    extra_metadata_list = pkl.load(open(data_dir + '/metadata/extra.p', 'rb'))
    read_dir = data_dir + '/extra/'
    write_dir = data_dir + '/processed/extra/'
    num_extra_files = len(extra_metadata_list)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    for i, data_dict in enumerate(extra_metadata_list):
        resize_crop_image_file(read_dir, write_dir, data_dict)
        if verbose == 1:
            print 'Processed files: %d/%d' % (i, num_extra_files)

    print 'Reading and processing test files ...'
    test_metadata_list = pkl.load(open(data_dir + '/metadata/test.p', 'rb'))
    read_dir = data_dir + '/test/'
    write_dir = data_dir + '/processed/test/'
    num_test_files = len(test_metadata_list)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    for i, data_dict in enumerate(test_metadata_list):
        resize_crop_image_file(read_dir, write_dir, data_dict)
        if verbose == 1:
            print 'Processed files: %d/%d' % (i, num_test_files)


def rename_extra_images(data_dir, metadata_file, verbose=1):
    """ Helper function to rename files in the extra folder and
        update corresponding metadata. """

    extra_metadata_list = pkl.load(open(metadata_file, 'rb'))
    num_train_images = len(os.listdir(data_dir + '/train'))
    num_extra_images = len(os.listdir(data_dir + '/extra'))
    for i in xrange(num_extra_images, 0, -1):
        new_filename = str(i + num_train_images) + '.png'
        os.rename(data_dir + '/extra/' + str(i) + '.png',
                  data_dir + '/extra/' + new_filename)
        extra_metadata_list[i-1]['filename'] = new_filename
        if verbose == 1:
            print('Renamed Image %s to %s' % (str(i) + '.png', new_filename))

    f = open(metadata_file, 'wb')
    pkl.dump(extra_metadata_list, f)
    f.close()


def create_final_datasets(read_data_dir, metadata_dir,
                          write_data_dir, seed=131,
                          nb_validation_samples=10000):

    np.random.seed(seed)

    train_metadata_list = pkl.load(open(metadata_dir + '/train.p', 'rb'))
    extra_metadata_list = pkl.load(open(metadata_dir + '/extra.p', 'rb'))

    nb_train_samples = len(train_metadata_list)
    nb_extra_samples = len(extra_metadata_list)

    train_index_array = np.arange(nb_train_samples)
    extra_index_array = np.arange(nb_extra_samples)
    train_index_array_p = np.random.permutation(train_index_array)
    extra_index_array_p = np.random.permutation(extra_index_array)

    val_train_metadata_list = [train_metadata_list[idx] for idx in train_index_array_p[0:6000]]
    val_extra_metadata_list = [extra_metadata_list[idx] for idx in extra_index_array_p[0:4000]]

    train_train_metadata_list = [train_metadata_list[idx] for idx in train_index_array_p[6000:]]
    train_extra_metadata_list = [extra_metadata_list[idx] for idx in extra_index_array_p[4000:]]

    final_val_metadata_list = []
    final_val_metadata_list.extend(val_train_metadata_list)
    final_val_metadata_list.extend(val_extra_metadata_list)
    final_val_metadata_list = sorted(final_val_metadata_list,
                                     key=lambda x: x['filename'])

    final_train_metadata_list = []
    final_train_metadata_list.extend(train_train_metadata_list)
    final_train_metadata_list.extend(train_extra_metadata_list)
    final_train_metadata_list = sorted(final_train_metadata_list,
                                       key=lambda x: x['filename'])

    if not os.path.exists(write_data_dir + '/train'):
        os.makedirs(write_data_dir + '/train')

    if not os.path.exists(write_data_dir + '/validation'):
        os.makedirs(write_data_dir + '/validation')

    print('Copying selected train files to validation ... ')
    for item in val_train_metadata_list:
        src_file = read_data_dir + '/train/' + item['filename']
        dest_dir = write_data_dir + '/validation/'
        shutil.copy(src_file, dest_dir)

    print('Copying selected extra files to validation ... ')
    for item in val_extra_metadata_list:
        src_file = read_data_dir + '/extra/' + item['filename']
        dest_dir = write_data_dir + '/validation/'
        shutil.copy(src_file, dest_dir)

    print('Writing validation set metadata file ... ')
    f = open(write_data_dir + '/validation/validation.p', 'wb')
    pkl.dump(final_val_metadata_list, f)
    f.close()

    print('Copying selected train files to train ... ')
    for item in train_train_metadata_list:
        src_file = read_data_dir + '/train/' + item['filename']
        dest_dir = write_data_dir + '/train/'
        shutil.copy(src_file, dest_dir)

    print('Copying selected extra files to train ... ')
    for item in train_extra_metadata_list:
        src_file = read_data_dir + '/extra/' + item['filename']
        dest_dir = write_data_dir + '/train/'
        shutil.copy(src_file, dest_dir)

    print('Writing train set metadata file ... ')
    f = open(write_data_dir + '/train/train.p', 'wb')
    pkl.dump(final_train_metadata_list, f)
    f.close()

    print('Copying test files ... ')
    src_dir = read_data_dir + '/test/'
    dest_dir = write_data_dir + '/test/'
    shutil.copytree(src_dir, dest_dir)

    print('Copying test set metadata file ... ')
    src_file = metadata_dir + '/test.p'
    dest_dir = write_data_dir + '/test/'
    shutil.copy(src_file, dest_dir)
