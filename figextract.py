import os

import imgproc

import numpy as np
from skimage.io import imread, imshow, imshow_collection, show, imsave
from skimage.color import rgb2gray
from skimage.measure import label

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage\n\tpractica2 [file]', file=sys.stderr)
        exit(1)

    output_dir = 'xtrct_output'

    im = imread(sys.argv[1], as_gray=True)
    obj_im = label(imgproc.imbinarize(im))
    col_im = imgproc.color_labels(obj_im)
    frame_coords, frame_labels = imgproc.frame_shapes(obj_im)

    frames = [obj_im[fc[0]:fc[2], fc[1]:fc[3]] for fc in frame_coords]
    filtered_objects = imgproc.label_filter(frames, frame_labels)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        imgcounter = 0
    else:
        try:
            with open(output_dir + '/.imgcount', 'r') as f:
                imgcounter = int(f.read())
        except FileNotFoundError:
            imgcounter = 0

    for i, fo in enumerate(filtered_objects):
        fo = np.round((fo / fo.max()) * 255)
        # Not checking contrast to supress warnings of data image range.
        # TODO: Make sure that grayscale image data is uint8
        imsave(f'{output_dir}/fig{imgcounter}.png',
               fo.astype('uint8'), check_contrast=False)
        imgcounter += 1

    with open(output_dir + '/.imgcount', 'w') as f:
        f.write(str(imgcounter))
