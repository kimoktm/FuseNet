# ============================================================== #
#              Dataset to Object (feed approach)                 #
#                                                                #
#                                                                #
# Datasets object to read, preprocess (resize & encode) data     #
# from given directory. Works with nyu_convert.py output format  #
# Datasets object holds training, testing and mapping data       #
# ============================================================== #

from __future__ import print_function

from PIL import Image

import numpy as np
import os


def dense_to_one_hot(annots_dense, num_labels):
    """
    Convert class annots from scalars to one-hot-vectors
    ex: 2->[0,1,0...] where size depends on classes num:
    ----------
    Args:
        annots_dense: Array of scalars
        num_labels:  Num of classes

    Returns:
        Array (input size, num_labels)
    """

    num_annots = annots_dense.shape[0]
    index_offset = np.arange(num_annots) * num_labels
    annots_one_hot = np.zeros((num_annots, num_labels))
    annots_one_hot.flat[index_offset + annots_dense.ravel()] = 1

    return annots_one_hot


def image_to_one_hot(annots_img, num_labels):
    """
    Convert a 2D annots image into a one-hot-image
    works by applying one-hot-vector for each pixel
    with (n x w x h x num_labels) format:
    ----------
    Args:
        annots_img:  Array of 2D scalar images
        num_labels: Num of classes

    Returns:
        Array (input size, width, height, num_labels)
    """

    img_one_hot = np.zeros((annots_img.shape[0], annots_img.shape[1], annots_img.shape[2], num_labels))
    for row in range(annots_img.shape[1]):
            for col in range(annots_img.shape[2]):
                    single = annots_img[:, row, col, 0]
                    one_hot = dense_to_one_hot(single, num_labels)
                    img_one_hot[:, row, col, :] = one_hot

    return img_one_hot


def load_data(file_path, width, height, one_hot = False, annots_num = 41, classes_num = 27):
    """
    Load data from given data paths file. Color, depth and annot
    images are resized to the specified dimensions and one_hot
    conversion of annots and classes is calculated if set:
    ----------
    Args:
        file_path:  CSV file that hold image paths and class ids
        width:      Target width
        height:     Target height
        one_hot:    One hot flag
        annots_num: Number of annots in dataset for one-hot
        classes_num: Number of classes in dataset for one-hot

    Returns:
        color, depth, annot and class data
    """

    print("[PROGRESS]\tloading data from %s" % file_path)

    data_dir = os.path.dirname(file_path)
    data_path = np.loadtxt(file_path, dtype = "string", delimiter = ",")

    for i in range(len(data_path)):
        color_img = Image.open(os.path.join(data_dir, data_path[i, 0]))
        color_img = color_img.resize((width, height))
        color = np.array(color_img).reshape(1, width, height, 3)

        depth_img = Image.open(os.path.join(data_dir, data_path[i, 1]))
        depth_img = depth_img.resize((width, height), Image.NEAREST)
        depth = np.array(depth_img).reshape(1, width, height, 1)

        annot_img = Image.open(os.path.join(data_dir, data_path[i, 2]))
        annot_img = annot_img.resize((width, height), Image.NEAREST)
        annot = np.array(annot_img).reshape(1, width, height, 1)

        clss = [int(data_path[i, 3])]

        if i == 0:
            color_data = color
            depth_data = depth
            annot_data = annot
            class_data = clss
        else:
            color_data = np.concatenate((color_data, color), axis = 0)
            depth_data = np.concatenate((depth_data, depth), axis = 0)
            annot_data = np.concatenate((annot_data, annot), axis = 0)
            class_data = np.concatenate((class_data, clss), axis = 0)

    if one_hot:
        annot_data = image_to_one_hot(annot_data, annots_num)
        class_data = dense_to_one_hot(class_data, classes_num)

    return color_data, depth_data, annot_data, class_data


class DataSet(object):
    """
    Dataset class to hold images, depth maps, annots and classes.
    """

    def __init__(self, images, depths, annots, classes):
        assert images.shape[0] == annots.shape[0] == \
            depths.shape[0] == classes.shape[0], "[ERROR   ]\tData are not the same size"

        self._size   = images.shape[0]
        self._images = images
        self._depths = depths
        self._annots = annots
        self._classes = classes
        self._epochs_completed = 0
        self._index_in_epoch   = 0

    @property
    def images(self):
        return self._images

    @property
    def depths(self):
        return self._depths

    @property
    def annots(self):
        return self._annots

    @property
    def classes(self):
        return self._classes

    @property
    def size(self):
        return self._size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """
        Get the next `batch_size` examples from
        this data set:
        ----------
        Args:
            batch_size:  Size of batch

        Returns:
            The next batch (img, depth, annot, class)
        """

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._size:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._size)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._depths = self._depths[perm]
            self._annots = self._annots[perm]
            self._classes = self._classes[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._size
        end = self._index_in_epoch

        return self._images[start:end], self._depths[start:end],\
               self._annots[start:end], self._classes[start:end]


def read_data_sets(dataset_dir, width = 224, height = 224, one_hot = False):
    """
    Given dataset dir read training and testing data
    data are saved in Datasets object for easy access:
    ----------
    Args:
        dataset_dir: Dataset dir 
        width:      Target width
        height:     Target height
        one_hot:    One hot flag

    Returns:
        Full dataset split into training and testing
    """

    class DataSets(object):
        pass
    data_sets = DataSets()

    train_data_paths = os.path.join(dataset_dir, "train_data.csv")
    test_data_paths  = os.path.join(dataset_dir, "test_data.csv")
    annot_names = os.path.join(dataset_dir, "annot_names.csv")
    class_names = os.path.join(dataset_dir, "class_names.csv")

    data_sets.annot_names = np.loadtxt(annot_names, dtype = "string", delimiter = ",")
    data_sets.class_names = np.loadtxt(class_names, dtype = "string", delimiter = ",")

    annots_num = len(data_sets.annot_names)
    classes_num = len(data_sets.class_names)

    train_c, train_d, train_l, train_s =load_data(train_data_paths, width, height, one_hot, annots_num, classes_num)
    test_c, test_d, test_l, test_s = load_data(test_data_paths, width, height, one_hot, annots_num, classes_num)

    data_sets.train = DataSet(train_c, train_d, train_l, train_s)
    data_sets.test  = DataSet(test_c, test_d, test_l, test_s)

    return data_sets
