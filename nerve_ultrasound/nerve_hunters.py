__author__ = 'Alex Galea'
# This script is based one created by Kaggle user ZFTurbo

import argparse
parser = argparse.ArgumentParser(description='Neural network KFold testing and predicting.')
# Run 'python nerve_hunters.py --KFold' to do cross validation on training data
# Run 'python nerve_hunters.py' to make predictions on test data
# Run 'python nerve_hunters.py --both' to do to cross validation and make predictions
# Add '--small' to use subset of training and test data
parser.add_argument('--KFold', action='store_true')
parser.add_argument('--both', action='store_true')
parser.add_argument('--small', action='store_true')
args = parser.parse_args()

import numpy as np
np.random.seed(84)
import cv2
import os
import glob
import time
import json
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, Callback

def get_raw_images(folder):
    ''' Load images from train/test folder using cv2.
    ---
    folder : str
        Name of folder to import from.
    '''
    if args.small:
        folder += '_small'
    all_paths = glob.glob(os.path.join(folder, '*.tif'))
    img_paths = [img for img in all_paths if 'mask' not in img]
    imgs, masks, ids = [], [], []
    for img_p in img_paths:
        ids.append(img_p[len(folder)+1:-4])
        img = cv2.imread(img_p, 0)
        imgs.append(img)
        if folder == 'train' or folder == 'train_small':
            mask = cv2.imread(img_p[:-4]+'_mask.tif', 0)
            masks.append(mask)
    return imgs, masks, np.array(ids)

def process_raw_images(imgs, n_rows, n_cols):
    ''' Transform images into features for classification model.
    ---
    imgs : list of numpy arrays with shape=[n_dim, n_rows, n_columns]
        Ultrasound images (2D) where n_dim=1 (one color chanel).
    n_rows / n_cols : int
        Number of rows / columns in returned image.
    '''
    proc_imgs = []
    for img in imgs:
        proc_img = image_shrink(img, n_rows=n_rows, n_cols=n_cols)
        proc_imgs.append([proc_img])
    # Must normalize such that max pixel value is 1.0
    return np.array(proc_imgs) / 255

def image_shrink(img, n_rows, n_cols):
    ''' Reduce image size.
    ---
    img : numpy array, shape=[n_rows, n_columns]
        Ultrasound image (2D).
    n_rows / n_cols : int
        Number of rows / columns in returned image.
    '''
    proc_img = cv2.resize(img, (n_cols, n_rows), cv2.INTER_LINEAR)
    return proc_img

def process_raw_masks(masks):
    ''' Convert array to binary classification.
    ---
    masks : list of numpy arrays with shape=[n_rows, n_columns]
        Areas where the nerve problem has been labeled.
    '''
    proc_masks = [1*(sum(m.flatten()) != 0) for m in masks]
    proc_masks = np_utils.to_categorical(proc_masks)
    return proc_masks

def clf_model_compile(X_train):
    ''' Neural network to classify ultrasound images as
    positive (1) or negative (0).
    ---
    X_train : numpy array, shape=[n_dim, n_samples, n_rows, n_columns]
        Set of training data (2D) where n_dim=1 (just one color chanel).
    y_train : numpy array, shape=[n_samples, 2]
        One-hot-encoded labels for training data.
    '''
    clf = Sequential()
    clf.add(Convolution2D(4, 3, 3, border_mode='same', init='he_normal',
                          input_shape=X_train[0].shape))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    
    clf.add(Flatten())
    clf.add(Dense(2))
    clf.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    clf.compile(optimizer=sgd, loss='categorical_crossentropy')

    return clf

class LossHistory(Callback):
    ''' Custom keras callback for extracting losses at each epoch. '''
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

#    model = Sequential()
#    model.add(Convolution2D(4, 3, 3, border_mode='same', init='he_normal',
#                            input_shape=X_train[0].shape))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.15))
#
#    model.add(Convolution2D(8, 3, 3, border_mode='same', init='he_normal'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.15))
#
#    model.add(Flatten())
#    model.add(Dense(2))
#    model.add(Activation('softmax'))
#
#    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')
#    return model


def write_submission(f_name, y_pred, X_test_id, mask):
    if f_name:
        # Create submission
        mask_rle = run_length_encode(mask)
        lines = []
        for y, y_id in zip(y_pred, X_test_id):
            # y look like [1., 0.] or [0., 1.]
            if y[0] == 0:
                # Positive prediction
                lines.append((int(y_id), y_id+','+mask_rle+'\n'))
            else:
                # Negative prediction
                lines.append((int(y_id), y_id+',\n'))
        lines = sorted(lines, key=lambda x: x[0])
        with open(f_name, 'w') as f:
            f.write('img,pixels\n')
            for _, line in lines:
                f.write(line)

    # Return the percentage of positive classifications
    p = y_pred.sum(axis=0)[1]/len(y_pred)
    return p

def run_length_encode(img):
    ''' Encode in compressed format for submission.
    img : numpy array, shape=[n_rows n_columns]
        Mask image to be encoded.
    order : str
        Decided the order in which to resize (flatten) the array.
    '''
    flattened = img.reshape(img.shape[0] * img.shape[1], order='F')
    runs = []
    r = 0
    pos = 1
    for c in flattened:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    if r != 0:
        runs.append((pos, r))
        pos += r

    z = ''
    for rr in runs:
        z += str(rr[0]) + ' ' + str(rr[1]) + ' '
    return z[:-1]

def get_average_mask():
    ''' Get mask to output if image given positive classification. '''
    masks = glob.glob(os.path.join('train', '*_mask.tif'))
    average_mask = np.zeros(shape=cv2.imread(masks[0], 0).shape,
                            dtype=np.float32)
    for mask in masks:
        # 0 indicates reading in grayscale
        m = cv2.imread(mask, 0)
        average_mask += m

    max_value = average_mask.max()
    threshold = max_value*0.5
    average_mask[average_mask < threshold] = 0
    average_mask[average_mask >= threshold] = 255
#    average_mask = average_mask.astype(np.uint8)
    return average_mask

def main():

    average_mask = get_average_mask()
    print('Generated average mask')

    n_rows, n_cols = 32, 32
    print('Reducing images to {} by {} pixels'.format(n_rows, n_cols))

    # Get training data
    t0 = time.time()
    imgs, masks, X_train_id = get_raw_images('train')
    t1 = time.time()
    print('Got raw training data in {} seconds'.format(t1-t0))
    X_train = process_raw_images(imgs, n_rows, n_cols)
    print('X_train shape:', X_train.shape)
    y_train = process_raw_masks(masks)
    print('y_train (target) shape:', y_train.shape)

    # Get test data
    t0 = time.time()
    imgs, masks, X_test_id = get_raw_images('test')
    t1 = time.time()
    print('Got raw test data in {} seconds'.format(t1-t0))
    X_test = process_raw_images(imgs, n_rows, n_cols)
    print('X_test shape:', X_test.shape)

    # Train neural network
    batch_size = 30
    nb_epoch = 50

    if args.KFold or args.both:

        print('--------------------------------------------------------')
        print('Starting KFold cross validation with batch_size={}, nb_epoch={}'.format(batch_size, nb_epoch))
        n_folds = 10
        kf = KFold(n=X_train.shape[0], n_folds=n_folds,
                  shuffle=True, random_state=81)
        kf_average = []
        kf_scores = {}
        for i, (train_index, test_index) in enumerate(kf):
            print('KFold {}/{}, train size = {}, test size = {}'.format(i+1, n_folds, len(train_index), len(test_index)))
            t0 = time.time()

            clf = clf_model_compile(X_train[train_index])
            loss_log = LossHistory()
            clf.fit(X_train[train_index], y_train[train_index],
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, shuffle=True, callbacks=[loss_log])
            kf_scores[i] = [float(L) for L in loss_log.losses]
            
            y_pred = clf.predict(X_train[test_index], batch_size=batch_size)
            print('Percent positive', write_submission('', y_pred, X_train_id[test_index], average_mask))
            # Ensure that y_pred consists only of [0.,1.] and [1.,0.]
            y_pred = np.array([[1.0*(y>0.5) for y in y_] for y_ in y_pred])
            score = accuracy_score(y_train[test_index], y_pred)
            t1 = time.time()
            print('Fold {0:d}; accuracy score: {1:.4f}, time: {2:.0f} seconds'.format(i+1, score, t1-t0))
            kf_average.append(score)

        score = (np.mean(kf_average), np.std(kf_average))
        print('Average score: {0:.4f} +/- {1:.4f}'.format(score[0], score[1]))
        json.dump(kf_scores, open('kf_scores_acc_{0:.3f}({1:.3f}).json'.format(score[0], score[1]), 'w'))

    if not args.KFold or args.both:

        print('--------------------------------------------------------')
        print('Starting test set prediction with batch_size={}, nb_epoch={}'.format(batch_size, nb_epoch))
        clf = clf_model_compile(X_train)
        print('Compiled clf model')
        loss_log = LossHistory()
        # Train the model
        clf.fit(X_train, y_train,
                batch_size=batch_size, nb_epoch=nb_epoch,
                verbose=1, shuffle=True, callbacks=[loss_log])
        test_scores = [float(L) for L in loss_log.losses]

        # Predict for the test data
        y_pred = clf.predict(X_test, batch_size=batch_size, verbose=1)
        # Ensure that y_pred consists only of [0.,1.] and [1.,0.]
        y_pred = np.array([[1.0*(y>0.5) for y in y_] for y_ in y_pred])
        p = write_submission('submission_{0:.3f}.csv'.format(test_scores[-1]), y_pred, X_test_id, average_mask)
        print('Percent positive predictions:', p)
        json.dump(test_scores, open('test_scores_{0:.3f}.json'.format(test_scores[-1]), 'w'))

if __name__ == '__main__':
    print('--------------------------------------------------------')
    print('Starting convolutional neural network training algorithm'+\
          '\nfor Kaggle dataset: Ultrasound Nerve Segmentation')
    print('--------------------------------------------------------')
    t0 = time.time()
    main()
    t1 = time.time()
    print('Total runtime = {} min'.format((t1-t0)/60))
