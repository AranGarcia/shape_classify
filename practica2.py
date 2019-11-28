'''
Funciones de matlab para sacar atributos
im_crop
graycomatrix
region_props
'''

import argparse
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from skimage.io import imread, imshow_collection, show
from skimage.measure import label, regionprops
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder

import imgproc

# This disables the tensorflow warning about AVX (which is good but annoying)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
IMG_SHAPE = (100, 100)
CLASS_FILE = 'training/classdata.txt'
IMG_PROPS = ['area', 'perimeter']


def _make_training_dir():
    if not os.path.exists('training'):
        os.mkdir('training')


def _classify(imattr, xclass):
    sim = cdist([imattr], xclass)
    return np.argmin(sim)


def train(traindir):
    # 1. Get images
    tr_objs = {}
    # Every folder in the directory will be considered a dataset
    # for a specific object.
    for d in os.listdir(traindir):
        classdir = os.path.join(traindir, d)
        if os.path.isdir(classdir):
            print(' -', d)
            for f in os.listdir(classdir):
                imfile = os.path.join(classdir, f)
                if os.path.isfile(imfile):
                    if d not in tr_objs:
                        tr_objs[d] = [imfile]
                    else:
                        tr_objs[d].append(imfile)

    # Obtain image properties
    X = []
    y = []
    for obclass, imfiles in tr_objs.items():
        for imf in imfiles:
            im = imgproc.readbin_resize(imf, IMG_SHAPE)
            improps = regionprops(im)[0]
            attrs = [improps[k] for k in IMG_PROPS]
            X.append(attrs)
            y.append(obclass)
            print(f'{obclass}: {attrs}')
    X = np.array(X)
    y = np.array(y)

    # Save class data to file
    le = LabelEncoder()
    le.fit(y)
    _make_training_dir()
    with open(CLASS_FILE, 'w') as f:
        for cl in le.classes_:
            class_X = X[y == cl].mean(axis=0)
            f.write(f'{cl}:{",".join(map(str, class_X))}\n')

    print('\nClass data found:')
    for cl in le.classes_:
        print(cl, X[y == cl].mean(axis=0))

    print('Class data written to', CLASS_FILE)


def identify(impath):
    with open(CLASS_FILE) as f:
        lines = f.readlines()

    # Structure class data
    labels = {}
    print('Class data:')
    for l in lines:
        lab, attrs = l.split(':')
        labels[lab] = [float(s.strip()) for s in attrs.split(',')]
        print(lab, labels[lab])
    class_x = np.array([v for v in labels.values()])
    class_y = np.array([k for k in labels.keys()])
    print('Classes found:')
    for i, cy in enumerate(class_y):
        print(f' - {cy}: {class_x[i]}')

    # Read image and find objects
    im = imread(impath, as_gray=True)
    obj_im = label(imgproc.imbinarize(im))
    col_im = imgproc.color_labels(obj_im)
    frame_coords, frame_labels = imgproc.frame_shapes(obj_im)
    frames = [obj_im[fc[0]:fc[2], fc[1]:fc[3]] for fc in frame_coords]
    filtered_objects = imgproc.label_filter(frames, frame_labels)
    filtered_objects = np.array([imgproc.imlbl_binarize(resize(fo, IMG_SHAPE))
                                 for fo in filtered_objects])

    regprops = [regionprops(fo)[0] for fo in filtered_objects]
    obj_count = {}
    for i, rp in enumerate(regprops):
        x = [rp[k] for k in IMG_PROPS]
        print('Classifying', x)
        y_hat = class_y[_classify(x, class_x)]

        if y_hat not in obj_count:
            obj_count[y_hat] = 1
        else:
            obj_count[y_hat] += 1

    print(obj_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image labeler for basic geometric shapes.')
    parser.add_argument('--train', required=False)
    parser.add_argument('--label', required=False)
    args = parser.parse_args()

    if args.train is None and args.label is None:
        print(
            'ERROR: at least one option must be selected (train or label).', file=sys.stderr)
    if args.train is not None:
        train(args.train)
    elif args.label is not None:
        identify(args.label)
