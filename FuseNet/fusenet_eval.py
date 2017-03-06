# ============================================================== #
#                         Fusnet eval                            #
#                                                                #
#                                                                #
# Eval fusnet with processed dataset in tfrecords format         #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf

import os
import glob
import argparse

import data.dataset_loader as dataset_loader
import data.tfrecords_downloader as tfrecords_downloader
import fusenet
from PIL import Image
import numpy as np

# Basic model parameters as external flags.
FLAGS = None


def maybe_download_and_extract():
    """
    Check if tfrecords exist if not download them
    (processed dataset into tfrecords with 40 labels & 10 classes)
    """

    if not tf.gfile.Exists(FLAGS.tfrecords_dir):
        tf.gfile.MakeDirs(FLAGS.tfrecords_dir)

    testing_tfrecords = glob.glob(os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'testing'))

    if not testing_tfrecords:
        print('[INFO    ]\tNo test tfrecords found. Downloading them in %s' % FLAGS.tfrecords_dir)
        tfrecords_downloader.download_and_extract_tfrecords(False, False, True, FLAGS.tfrecords_dir)


def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '%s-*' % type)
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size


def maybe_save_images(images, filenames):
    """
    Save images to disk
    -------------
    Args:
        images: numpy array     [batch_size, image_size, image_size]
        filenames: numpy string array, filenames corresponding to the images   [batch_size]
    """

    if FLAGS.output_dir is not None:
        batch_size = images.shape[0]
        for i in xrange(batch_size):
            image_array = images[i, :, :]
            file_path = os.path.join(FLAGS.output_dir, filenames[i])
            image = Image.fromarray(np.uint8(image_array))
            image.save(file_path)
    
    
def evaluate():
    """
    Eval fusenet using specified args:
    """

    data_files, data_size = load_datafiles('testing')
    images, depths, annots, classes, filenames = dataset_loader.inputs(
                                                     data_files = data_files,
                                                     image_size = FLAGS.image_size,
                                                     batch_size = FLAGS.batch_size,
                                                     num_epochs = 1,
                                                     train = False)

    annot_logits, class_logits = fusenet.build(images, depths, FLAGS.num_annots, FLAGS.num_classes, False)

    mask = tf.not_equal(annots, 0)
    annots_without_class_zero = tf.boolean_mask(annots, mask)
    mask = tf.reshape(mask, [-1])
    annot_logits_without_class_zero = tf.boolean_mask(annot_logits, mask)
    
    predicted_images = fusenet.predictions(annot_logits, FLAGS.batch_size, FLAGS.image_size)
    
    total_acc, seg_acc , class_acc = fusenet.accuracy(annot_logits_without_class_zero, annots_without_class_zero, class_logits, classes)

    true_positives, false_positives, true_negatives, false_negatives = fusenet.segmentation_metrics(annot_logits_without_class_zero, annots_without_class_zero)
    
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    saver = tf.train.Saver()

    if not tf.gfile.Exists(FLAGS.checkpoint_path + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        print('[INFO    ]\tFound checkpoint file, restoring model.')
        saver.restore(sess, FLAGS.checkpoint_path)
    
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    total_true_positives = np.zeros([FLAGS.num_annots], dtype=np.int64)
    total_false_positives = np.zeros([FLAGS.num_annots], dtype=np.int64)
    total_true_negatives = np.zeros([FLAGS.num_annots], dtype=np.int64)
    total_false_negatives = np.zeros([FLAGS.num_annots], dtype=np.int64)
    
    try:
        step = 0
        while not coord.should_stop():
            acc_value, predicted_images_value, filenames_value = sess.run([seg_acc, predicted_images, filenames])

            tp_val, fp_val, tn_val, fn_val = sess.run([true_positives, false_positives, true_negatives, false_negatives])

            total_true_positives += tp_val
            total_false_positives += fp_val
            total_true_negatives += tn_val
            total_false_negatives += fn_val
            
            maybe_save_images(predicted_images_value, filenames_value)
            print('[PROGRESS]\tSegmentation accuracy for current batch: %.3f' % acc_value)
            step += 1

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone evaluating in %d steps.' % step)

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    print (total_true_positives)
    print (total_false_positives)
    global_accuracy, classwise_accuracy, intersection_over_union = segmentation_accuracies(total_true_positives, total_false_positives, total_true_negatives, total_false_negatives)
    print('[RESULT  ]\tGlobal accuracy = %.5f, classwise accuracy = %.5f, intersection over union = %.5f' % (global_accuracy, classwise_accuracy, intersection_over_union))
    
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    


def segmentation_accuracies(true_positives, false_positives, true_negatives, false_negatives):
    """
    Segmentation accuracies:
    ----------
    Args:
        true_positives:  Tensor [num_annots]
        false_positives: Tensor [num_annots]
        true_negatives:  Tensor [num_annots]
        false_negatives: Tensor [num_annots]

    Returns:
        global_accuracy:         true_positives / pixel_count
        classwise_accuracy:      1/class_count * SUM(over all classes) [ class_true_positives / (class_true_positives + class_false_positives) ]
        intersection_over_union: 1/class_count * SUM(over all classes) [ class_true_positives / (class_true_positives + class_false_positives + class_false_negatives) ]
    """

    classes = true_positives.shape[0]
    total_pixel_count = (np.sum(true_positives) + np.sum(false_positives) + np.sum(true_negatives) + np.sum(false_negatives)) * 1.0 / classes
    global_accuracy = np.sum(true_positives) / total_pixel_count
    classwise_accuracy = np.sum(1.0 * true_positives / (true_positives + false_positives))/ (classes - 1)
    intersection_over_union = np.sum(1.0 * true_positives / (true_positives + false_positives + false_negatives)) / (classes-1)

    return global_accuracy, classwise_accuracy, intersection_over_union

def main(_):
    """
    Run fusenet prediction on input tfrecords
    """
    maybe_download_and_extract()

    if FLAGS.output_dir is not None:
        if not tf.gfile.Exists(FLAGS.output_dir):
            print('[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(FLAGS.output_dir)
        
    evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Eval FuseNet on given tfrecords.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory', default = '../Datasets/NYU/tfrecords')
    parser.add_argument('--checkpoint_path', help = 'Path of checkpoint to restore. (Ex: ../Datasets/NYU/checkpoints/fusenet.ckpt-80000)')
    parser.add_argument('--num_annots', help = 'Number of segmentation labels', type = int, default = 41)
    parser.add_argument('--num_classes', help = 'Number of Classification labels', type = int, default = 11)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 4)
    parser.add_argument('--visualize', help = 'Visualize predicted annotations', type = bool, default = False)
    parser.add_argument('--output_dir', help = 'Output directory for the prediction files. If this is not set then predictions will not be saved')
    
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
