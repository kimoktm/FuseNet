# ============================================================== #
#             Dataset to tfrecords (open pipeline)               #
#                                                                #
#                                                                #
# Converts image data to TFRecords format with Example protos    #
# The image data set is expected to reside in img files located  #
# in the following structure data_dir/class_0/image0.png...      #
# ============================================================== #

from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import random
import sys
import threading

from image_coder import ImageCoder


# Basic model parameters as external flags.
FLAGS = None


def int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto:
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """
    Wrapper for inserting bytes features into Example proto:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(filename, image_buffer, depth_buffer, annot_buffer,
                                                clss, height, width):
    """
    Build an Example proto for an example:
    ----------
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.png'
        image_buffer: string, PNG encoding of RGB image
        clss: integer, identifier for the ground truth for the network
        height: integer, image height in pixels
        width: integer, image width in pixels

    Returns:
            Example proto
    """

    image_format = 'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(tf.compat.as_bytes(os.path.basename(os.path.normpath(filename)))),
            'image/format': bytes_feature(tf.compat.as_bytes(image_format)),
            'image/class': int64_feature(clss),
            'image/encoded/color': bytes_feature(tf.compat.as_bytes(image_buffer)),
            'image/encoded/depth': bytes_feature(tf.compat.as_bytes(depth_buffer)),
            'image/encoded/annot': bytes_feature(tf.compat.as_bytes(annot_buffer))
            }))
    return example


def process_image(filename, coder):
    """
    Process a single image file:
    ----------
    Args:
        filename: string, path to an image file e.g., '/path/to/example.png'
        coder: instance of ImageCoder to provide TensorFlow image coding utils

    Returns:
        image_buffer: string, PNG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """

    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    # Convert any JPEG to PNG for consistency
    if coder.is_jpeg(filename):
        print('[PROGRESS]\tConverting JPEG to PNG for %s' % filename)
        image_data = coder.jpeg_to_png(image_data)

    # Decode the PNG
    image = coder.decode_png(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]

    # Check that color has 3 channels while depth, annot just 1
    if FLAGS.name_color in filename:
        assert image.shape[2] == 3
    else:
        assert image.shape[2] == 1

    return image_data, height, width


def process_image_files_batch(coder, thread_index, ranges, name, filenames,
                                                             classes, num_shards):
    """
    Processes and saves list of images as TFRecord in 1 thread
    ----------
    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
            analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        classes: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set

    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64].
    """

    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):

        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d.tfrecords' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            clss = classes[i]

            # Concatenate naming conventions to get color, depth, annot true paths
            color_file = filename % FLAGS.name_color
            depth_file = filename % FLAGS.name_depth
            annot_file = filename % FLAGS.name_annot

            image_buffer, height, width = process_image(color_file, coder)
            depth_buffer, height, width = process_image(depth_file, coder)
            annot_buffer, height, width = process_image(annot_file, coder)

            example = convert_to_example(filename, image_buffer, depth_buffer,
                                         annot_buffer, clss, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

        writer.close()
        print('[THREAD %d]\tWrote %d images to %s' %
                    (thread_index, shard_counter, output_file))
        shard_counter = 0

    print('[THREAD %d]\tWrote %d images to %d shards.' %
                (thread_index, counter, num_files_in_thread))


def process_image_files(name, filenames, classes, num_shards):
    """
    Process and save list of images as TFRecord of Example protos:
    ----------
    Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        classes: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set
    """

    assert len(filenames) == len(classes)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []

    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('[PROGRESS]\tLaunching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                        classes, num_shards)
        t = threading.Thread(target=process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('[INFO    ]\tFinished writing all %d images in data set.' % len(filenames))


def find_image_files(data_dir, classes_file):
    """
    Build a list of all images files and classes in the data set:
    ----------
    Args:
        data_dir: string, path to the root directory of images. Assumes
        (data_dir/kitchen/image.png) format, where kitchen is the class
        classes_file: string, path to the classes file. Assumes a list
        of classes one per line, where 1 maps to the first line class.

    Returns:
        filenames: list of strings; each string is a path to an image file
        following the format data_dir/class/image%s.png, %s will be replaced
        with color, depth or annoation extension image_color.png
        classes: list of integer; each integer identifies the ground truth
    """

    print('[PROGRESS]\tDetermining list of input files and classes from %s.' % data_dir)

    unique_classes = [l.strip() for l in tf.gfile.FastGFile(
            classes_file, 'r').readlines()]

    # Leave clss index 0 empty as a background class.
    class_index = 1
    classes   = []
    filenames = []

    # Construct the list of JPEG files and classes.
    for text in unique_classes:
        if tf.gfile.Exists(os.path.join(data_dir, text)):
            color_file_path = '%s/%s/*%s.*' % (data_dir, text, FLAGS.name_color)
            depth_file_path = '%s/%s/*%s.*' % (data_dir, text, FLAGS.name_depth)
            annot_file_path = '%s/%s/*%s.*' % (data_dir, text, FLAGS.name_annot)

            color_files = tf.gfile.Glob(color_file_path)
            depth_files = tf.gfile.Glob(depth_file_path)
            annot_files = tf.gfile.Glob(annot_file_path)

            assert len(color_files) == len(depth_files) == len(annot_files)

            matching_files = [ x.replace(FLAGS.name_color, '%s') for x in color_files ]

            classes.extend([class_index] * len(matching_files))
            filenames.extend(matching_files)

        class_index += 1

    # Shuffle the ordering of all image files in order to guarantee randomness
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    classes   = [classes[i] for i in shuffled_index]
    filenames = [filenames[i] for i in shuffled_index]

    print('[INFO    ]\tFound %d images across %d classes inside %s.' %
                (len(filenames), len(unique_classes), data_dir))

    return filenames, classes


def process_dataset(name, directory, num_shards, classes_file):
    """
    Process a complete data set and save it as a TFRecord
    ----------
    Args:
        name: string, unique identifier specifying the data set
        directory: string, root path to the data set
        num_shards: integer number of shards for this data set
        classes_file: string, path to the classes file
    """

    filenames, classes = find_image_files(directory, classes_file)
    process_image_files(name, filenames, classes, num_shards)


def main(_):

    assert not FLAGS.num_shards % FLAGS.num_threads, (
            '[ERROR   ]\tPlease make the FLAGS.num_threads commensurate with FLAGS.num_shards')

    print('[INFO    ]\tSaving results to %s' % FLAGS.output_dir)

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    process_dataset(os.path.basename(os.path.normpath(FLAGS.data_dir)), FLAGS.data_dir,
                                     FLAGS.num_shards, FLAGS.classes_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Converts image data to TFRecords file format')
    parser.add_argument('--data_dir', help = 'Data directory', required = True)
    parser.add_argument('--output_dir', help = 'Output data directory', required = False)
    parser.add_argument('--classes_file', help = 'classes file', required = True)
    parser.add_argument('--num_threads', help = 'Number of threads', type = int, default = 1)
    parser.add_argument('--num_shards', help = 'Number of shards in training TFRecord files', type = int, default = 1)
    parser.add_argument('--name_color', help = 'Color images name format', default = '_color')
    parser.add_argument('--name_depth', help = 'Depth image name format', default = '_depth')
    parser.add_argument('--name_annot', help = 'annotation images name format', default = '_annot')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
