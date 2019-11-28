'''
skimage interface for basic image processing functions.
'''

from os import listdir, path
from colorsys import hsv_to_rgb

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread, imshow, show
from skimage.measure import label
from skimage.transform import resize


def imbinarize(imarr, thres=0.5):
    '''
    Creates a binarized version of an image array. Values bel
    '''
    if imarr.ndim == 3:
        imarr = rgb2gray(imarr)
    return np.array(imarr < thres, dtype='uint8')


def color_labels(imgarr):
    '''
    '''
    if imgarr.ndim != 2:
        raise ValueError('array must be 2-D.')

    # White (label 0) doesn't count as an object
    labs = np.unique(imgarr[imgarr != 0])
    nlab = labs.size

    colord_img = np.full((*imgarr.shape, 3), 255, dtype=np.uint8)
    rgb_colors = [hsv_to_rgb(h/nlab, .75, 1) for h in range(nlab)]
    for i, l in enumerate(labs):
        colord_img[imgarr == l] = [round(255 * rgbc)
                                   for rgbc in rgb_colors[i]]

    return colord_img


def frame_shapes(imgarr):
    '''
    Finds the indices of the upper-left and lower-right corner of the rectangle that
    frames an object with its label.
    '''
    if imgarr.ndim != 2:
        raise ValueError('array must be 2-D.')
    labs = np.unique(imgarr[imgarr != 0])

    indices = []
    for l in labs:
        idx = np.where(imgarr == l)
        indices.append([np.min(idx[0]), np.min(idx[1]),
                        np.max(idx[0]), np.max(idx[1])])

    return np.array(indices), labs


def label_filter(imgs, labels):
    if not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    if np.any([im.ndim != 2 for im in imgs]):
        raise ValueError('imgs must have a 3-D array.')
    if labels.ndim != 1:
        raise ValueError('labels must be a 1-D array.')

    if imgs.shape[0] != labels.shape[0]:
        raise ValueError('there must be a label for every image array.')

    # TODO: Find a way to optimize this memory-wise.
    filtered = []
    for i, l in enumerate(labels):
        fil = np.copy(imgs[i])
        fil[np.where(fil != l)] = 0
        filtered.append(fil)

    return np.array(filtered)


def readbin_resize(impath, dim, thres=0.5):
    rim = resize(imread(impath), dim)
    rim[rim > thres] = 1
    rim[rim <= thres] = 0

    return rim.astype('uint8')

def bin_resize(im, dim, thres=0.5):
    rim = resize(im, dim)
    rim[rim > thres] = 1
    rim[rim <= thres] = 0
    return rim.astype('uint8')
