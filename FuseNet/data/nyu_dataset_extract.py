# ============================================================== #
#                         NYU Convert                            #
#                                                                #
#                                                                #
# nyu_convert is used to extract images, annotations and class   #
# mappings from raw NYUv2 data set, also splits dataset into     #
# training and testing sets and maps annots to a set of labels   #
# ============================================================== #

from __future__ import print_function

from joblib import Parallel, delayed
from PIL import Image

import numpy as np
import scipy.io
import h5py
import argparse
import os


def map_annots(annot):
    """
    map annots in given image to a set of labels
    defined by 'annots_mapping' input. if no mapping
    is providied, default annots are used 0->894:
    ----------
    Args:
        annot: Array (width, height)

    Returns:
        Array (width, height) with mapped annots
    """

    annot = np.array(annot, dtype = np.int16)
    annot -= 1
    annot[annot < 0] = 0
    mappedannots = annots_map[annot]

    return mappedannots


def normalize_img(img):
    """
    normalize given image. Used to normalized
    depth maps values to be in 0->255 range:
    ----------
    Args:
        img: Array (width, height)

    Returns:
        Array (width, height) normalized
    """

    assert (len(img[img == 0.0]) >= 0), (
            "[ERROR   ]\tDepth map is missing values!")

    maxdepth = np.nanmax(img)
    mindepth = np.nanmin(img)
    img = img.copy()
    img -= mindepth
    img /= (maxdepth - mindepth)
    img *= 255

    return img


def process_image(i, class_id, color, depth, annot):
    """
    process given raw image into its 3 components
    color, depth, annot images and save them with
    format (datatype/class/img) ex: (testing/bedroom/img.png)
    All components are cropped to remove the white border,
    depth image is normalized and annot image is
    mapped to a set of labels given by 'annots_mapping':
    ----------
    Args:
        i:        Index of image
        class_id: class id (bedroom, classroom..)
        color:    Raw color image
        depth:    Raw depth image
        annot:    Raw annots image

    Returns:
        Array (color_path, depth_path, annot_depth, class_id)
    """

    idx = int(i) + 1
    print("[PROGRESS]\timage", idx, "/", len(images))

    if idx in train_images:
        data_type = "training"
    else:
        assert idx in test_images, (
                "[ERROR   ]\tindex %d neither found in training set nor in test set" % idx)
        data_type = "testing"

    folder   = "%s/%s/%s" % (output_dir, data_type, classes_names[class_id])
    filename = "%s/%05d" % (folder, i)

    if not os.path.exists(folder):
        os.makedirs(folder)

    color_image = Image.fromarray(color, mode='RGB')
    crop_box  = (10, 8, color_image.size[0] - 10, color_image.size[1] - 8)

    color_image = color_image.crop(crop_box)
    color_image.save(filename + "_color.png")

    depth_image = normalize_img(depth)
    depth_image = Image.fromarray(depth_image.astype('uint8'), mode='L')
    depth_image = depth_image.crop(crop_box)
    depth_image.save(filename + "_depth.png")

    annot_image = map_annots(annot)
    annot_image = Image.fromarray(annot_image.astype('uint8'), mode='L')
    annot_image = annot_image.crop(crop_box)
    annot_image.save(filename + "_annot.png")

    rel_path   = "%s/%s/%05d" % (data_type, classes_names[class_id], i)

    return [ idx in train_images, [rel_path + "_color.png", 
            rel_path +"_depth.png", rel_path + "_annot.png", class_id]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Extract and process NYUDV2 dataset')
    parser.add_argument('-i', '--input_data', help = 'Input dataset (.mat)', required = True)
    parser.add_argument('-s', '--split_map', help = 'Training and testing split (.mat)', required = True)
    parser.add_argument('-l', '--annots_map', help = 'Annotations mapping (.mat)', required = False)
    parser.add_argument('-o', '--output_dir', help = 'Output directory', required = True)
    parser.add_argument('-n', '--threads', help = 'Number of threads', required = False, default = -1)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset      = h5py.File(args.input_data, "r")
    split_map    = scipy.io.loadmat(args.split_map)
    output_dir   = args.output_dir
    test_images  = set([int(x) for x in split_map["testNdxs"]])
    train_images = set([int(x) for x in split_map["trainNdxs"]])

    images = dataset['images']
    depths = dataset['depths']
    annots = dataset['labels']

    classes = [u''.join(unichr(c) for c in dataset[obj_ref]) for obj_ref in dataset['sceneTypes'][0]]
    annots_names = [u''.join(unichr(c) for c in dataset[obj_ref]) for obj_ref in dataset['names'][0]]


    # if annotations mapping is provided use it 
    # other wise stick to the default 894 annotations
    if args.annots_map is not None:
        annots_map = scipy.io.loadmat(args.annots_map)
        annots_map = annots_map["mapping"][0]
    else:
        annots_map = range(len(annots_names))

    # Extract classes names and their corresponding ids
    # so names can be retrived from ids ex: 0->basement
    classes_names, classes_ids = np.unique(classes, return_inverse = True)
    np.savetxt("%s/class_names.txt" % output_dir, classes_names, "%s")

    # Extract annotations names according to the given mapping
    # currently names are given by the first element
    # in the group. 40 class mapping 'Gupta et al' is
    # handled correctly following naming convetion used:
    # https://people.eecs.berkeley.edu/~sgupta/pdf/GuptaArbelaezMalikCVPR13.pdf
    mappedannots_names = [u''.join("none")]
    for annot_id in range(1, np.amax(annots_map) + 1):
        mappedannots_names.append([annots_names[i] for i in np.where(annots_map == annot_id)[0]][0])
    if np.amax(annots_map) == 40:
        mappedannots_names[38] = u''.join("other struct")
        mappedannots_names[39] = u''.join("other furntr")
        mappedannots_names[40] = u''.join("other prop")
    np.savetxt("%s/annotation_names.txt" % output_dir, mappedannots_names, "%s")


    # Process and save images using either
    # single or multiple threads
    data_paths = []

    if int(args.threads) == 1:
        print("[INFO    ]\tSingle-threaded mode")
        for i, image in enumerate(images):
            data_paths.append(process_image(i, classes_ids[i], image.T, depths[i, :, :].T, annots[i, :, :].T))
    else:
        print("[INFO    ]\tMulti-threaded mode")
        data_paths = Parallel(args.threads)(delayed(process_image)(i, classes_ids[i], images[i, :, :].T, depths[i, :, :].T, annots[i, :, :].T) for i in range(len(images)))


    # Split data paths into training and testing
    # sets so each can be saved in a sperate file
    train_data = []
    test_data  = []

    for d in data_paths:
        if d[0] == True:
            train_data.append(d[1])
        else:
            test_data.append(d[1])

    # Save train & test data files using this format:
    # [color_img_path, depth_img_path, annot_img_depth, class_id]
    np.savetxt("%s/train_data.txt" % output_dir, train_data, "%s")
    np.savetxt("%s/test_data.txt" % output_dir, test_data, "%s")
    
    print("[INFO    ]\tData Statistics:")
    print("\t\t%d\t: Training images" % len(train_images))
    print("\t\t%d\t: Testing images" % len(test_images))
    print("\t\t%d\t: classes" % len(classes_names))
    print("\t\t%d\t: annotations" % len(annots_names))
