
# Append to PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, util
from keras.optimizers import SGD
from SVHNDigit.models.cnn import CNN_B


def get_cnn_model():

    reg_factor = 3e-6
    lr = 0.03
    decay = 0
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
                          'nb_epochs': 20,
                          'nb_train_samples': 99712 * 6,
                          'nb_validation_samples': 6000}

    input_dim = (3, 32, 32)

    cnn = CNN_B(model_define_params, input_dim)
    cnn.define(verbose=0)
    cnn.model.load_weights('model_data/CNN_B_20Epochs_Best.h5')

    optimizer = SGD(lr=model_train_params['lr'],
                    momentum=model_train_params['momentum'],
                    decay=model_train_params['decay'],
                    nesterov=model_train_params['nesterov'])

    cnn.model.compile(loss=model_train_params['loss'],
                      metrics=model_train_params['metrics'],
                      optimizer=optimizer)

    return cnn

def find_region_proposal(input_im, cnn_model):

    win_size = 32
    step_size = 4
    #input_im = np.hstack((input_im, np.zeros([input_im.shape[0], win_size-1, 3])))
    #input_im = np.vstack((input_im, np.zeros([win_size-1, input_im.shape[1], 3])))

    nrows = input_im.shape[0]
    ncols = input_im.shape[1]

    heat_map = np.zeros([nrows, ncols])
    entropy_map = np.zeros([nrows, ncols])

    process_im = input_im.transpose((2, 0, 1))
    process_im = process_im * 1.0/255.0

    window_shape = (3, win_size, win_size)
    im_crops = util.view_as_windows(process_im, window_shape, step=step_size)
    heat_map_crops = util.view_as_windows(heat_map, (win_size, win_size), step=step_size)
    entropy_map_crops = util.view_as_windows(entropy_map, (win_size, win_size), step=step_size)

    #print(im_crops.shape)

    for i in xrange(im_crops.shape[1]):
        for j in xrange(im_crops.shape[2]):
            predict_im = im_crops[0, i, j]
            #print(predict_im.shape)
            predict_im = predict_im[np.newaxis, :, :]
            pred_probs = cnn_model.model.predict_proba(predict_im,
                                                       batch_size=1, verbose=0)
            entropy = np.sum(-pred_probs*np.log2(pred_probs))
            #print entropy
            max_prob = np.max(pred_probs)
            heat_map_crops[i, j][0, 0] = max_prob
            entropy_map_crops[i, j][0, 0] = entropy

    heat_map_th = heat_map.copy()
    threshold = 0.995
    heat_map_th[heat_map_th >= threshold] = 1
    heat_map_th[heat_map_th < threshold] = 0

    entropy_map_th = entropy_map.copy()
    # Max entropy for uniform distribution is 3.33
    entropy_threshold = 1.5
    entropy_map_th[entropy_map_th > entropy_threshold] = 0
    entropy_map_th[entropy_map_th <= entropy_threshold] = 1

    final_heat_map = np.logical_and(entropy_map_th, heat_map_th)

    kernel = np.ones([6, 6], int)
    dilated_heat_map = morphology.binary_dilation(final_heat_map, kernel)
    # dilated_entropy_map = morphology.binary_dilation(entropy_map_th, kernel)

    labeled_heat_map = measure.label(dilated_heat_map, connectivity=2)
    label_dict = measure.regionprops(labeled_heat_map)

    # labeled_entropy_map = measure.label(dilated_entropy_map, connectivity=2)
    # entropy_label_dict = measure.regionprops(labeled_entropy_map)

    areas = [x['area'] for x in label_dict]
    largest_ccomp = label_dict[np.argmax(areas)]
    min_row, min_col, max_row, max_col = largest_ccomp['bbox']

#     if min_row - (win_size/2) < 0:
#         min_row = 0
#     else:
#         min_row = min_row - (win_size/2)

#     if min_col - (win_size/2) < 0:
#         min_col = 0
#     else:
#         min_col = min_col - (win_size/2)

    return (final_heat_map, dilated_heat_map,
            input_im[min_row:max_row+(win_size), min_col:max_col+(win_size)])

print('Loading CNN model ... ')
cnn_model = get_cnn_model()
print('CNN model loaded ... ')
cap = cv2.VideoCapture(0)
ramp_frames = 30
for i in range(ramp_frames):
    ret, frame = cap.read()
ret, frame = cap.read()
cv2.waitKey(1)
cap.release()

nrows = frame.shape[0]
ncols = frame.shape[1]
print(nrows, ncols)
if ncols > nrows:
    min_axis = 0
    min_dim = nrows
    max_axis = 1
    max_dim = ncols
    crop_frame = frame[:, (max_dim-min_dim)/2:(max_dim+min_dim)/2, :]
else:
    min_axis = 1
    min_dim = ncols
    max_axis = 0
    max_dim = nrows
    crop_frame = frame[(max_dim-min_dim)/2:(max_dim+min_dim)/2, :, :]

dim = (128, 128)
display_frame = cv2.resize(crop_frame, dim, interpolation=cv2.INTER_AREA)
display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

# dim = (128, 128)
# process_frame = cv2.resize(crop_frame, dim, interpolation=cv2.INTER_AREA)
# process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
print('Finding region proposal ... ')
heat_map, dheat_map, rprops_frame = find_region_proposal(display_frame, cnn_model)
dim = (32, 32)
rprops_frame = cv2.resize(rprops_frame, dim, interpolation=cv2.INTER_AREA)
rprops_frame = rprops_frame.transpose((2, 0, 1))
rprops_frame = rprops_frame * 1.0/255
rprops_frame = rprops_frame[np.newaxis, :, :, :]

print('Predicting digit value ... ')
pred_y = cnn_model.model.predict_classes(rprops_frame, batch_size=1, verbose=0)
print(pred_y)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(display_frame)
plt.title('Captured Image \n Predicted Label: %d' % (pred_y[0]))

plt.subplot(2, 2, 2)
plt.imshow(heat_map, cmap='bone')
plt.colorbar()
plt.title('Thresholded Heat Map')

plt.subplot(2, 2, 3)
plt.imshow(rprops_frame[0].transpose((1, 2, 0)))
plt.title('ConvNet Input Image')

plt.subplot(2, 2, 4)
plt.imshow(dheat_map, cmap='bone')
plt.colorbar()
plt.title('Dilated Heat Map')

plt.tight_layout()
# plt.show()

plt.savefig("example.pdf")




