# ============================================================== #
#                         Fusnet train                           #
#                                                                #
#                                                                #
# Train fusnet with processed dataset in tfrecords format        #
# ============================================================== #

from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import time
import glob

import data.dataset_loader as dataset_loader
import data.tfrecords_downloader as tfrecords_downloader
import utils.tools as tools
import fusenet


# Basic model parameters as external flags.
FLAGS = None


def maybe_download_and_extract():
    """
    Check if tfrecords exist if not download them
    (processed dataset into tfrecords with 40 labels & 10 classes)
    """

    if not tf.gfile.Exists(FLAGS.tfrecords_dir):
        tf.gfile.MakeDirs(FLAGS.tfrecords_dir)


    tfrecords = glob.glob(os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'train'))
    testing_tfrecords = glob.glob(os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'test'))

    download_training_records = False

    if not tfrecords:
        print('[INFO    ]\tNo train tfrecords found. Downloading them in %s' %FLAGS.tfrecords_dir)
        download_training_records = True
    
    tfrecords_downloader.download_and_extract_tfrecords(download_training_records, False, FLAGS.tfrecords_dir)


def load_datafiles():
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'train')
    data_files = tf.gfile.Glob(tf_record_pattern)

    return data_files


def initialize_session(sess):
    """
    Initializes a new session that wasn't loaded from a checkpoint
    """
    print('[INFO    ]\tCreated a new session without restoring checkpoint, loading vgg weights')
    tools.load_vgg_weights(FLAGS.vgg_path, tf.get_default_graph(), sess)

    
def train():
    """
    Train fusenet using specified args:
    """

    data_files = load_datafiles()
    images, depths, annots, classes = dataset_loader.inputs(
                                            data_files = data_files,
                                            image_size = FLAGS.image_size,
                                            batch_size = FLAGS.batch_size,
                                            num_epochs = FLAGS.num_epochs,
                                            train = True)

    annot_logits, class_logits = fusenet.build(images, depths, FLAGS.num_annots, FLAGS.num_classes, True)

    total_acc, seg_acc, class_acc = fusenet.accuracy(annot_logits, annots, class_logits, classes)

    loss = fusenet.loss(annot_logits, annots, class_logits, classes, FLAGS.weight_decay_rate)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = fusenet.train(loss, FLAGS.learning_rate, FLAGS.learning_rate_decay_steps, FLAGS.learning_rate_decay_rate, global_step)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()
    session_manager = tf.train.SessionManager()

    sess = session_manager.prepare_session("", init_op=init, saver=saver, checkpoint_dir=FLAGS.checkpoint_dir, init_fn=initialize_session)
    
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    try:
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            step = tf.train.global_step(sess, global_step)
            if step % 1000 == 0:
                acc_total_value, acc_seg_value, acc_clss_value = sess.run([total_acc, seg_acc, class_acc])
                
                print('[PROGRESS]\tStep %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                print('\t\tTraining segmentation accuracy = %.2f, classifcation accuracy = %.2f, total accuracy = %.2f'
                     % (acc_seg_value, acc_clss_value, acc_total_value))

            if step % 5000 == 0:
                print('[PROGRESS]\tSaving checkpoint')
                saver.save(sess, FLAGS.checkpoint_dir, global_step = global_step)

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(_):
    """
    Download processed dataset if missing & train
    """

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        print('[INFO    ]\tCheckpoint directory does not exist, creating directory: ' + os.path.abspath(FLAGS.checkpoint_dir))
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    maybe_download_and_extract()
    train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train FuseNet on given tfrecords directory.\
                                    Automatically download and process data to tfrecords if dir not found.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory', default = '../Datasets/NYU/tfrecords')
    parser.add_argument('--checkpoint_dir', help = 'Checkpoints directory', default = '../Datasets/NYU/checkpoints')
    parser.add_argument('--num_annots', help = 'Number of segmentation labels', type = int, default = 41)
    parser.add_argument('--num_classes', help = 'Number of Classification labels', type = int, default = 11)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--learning_rate', help = 'Learning rate', type = float, default = 0.001)
    parser.add_argument('--learning_rate_decay_steps', help = 'Learning rate decay steps', type = int, default = 50000)
    parser.add_argument('--learning_rate_decay_rate', help = 'Learning rate decay rate', type = float, default = 0.9)
    parser.add_argument('--weight_decay_rate', help = 'Weight decay rate', type = float, default = 0.0005)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 4)
    parser.add_argument('--vgg_path', help = 'VGG weights path (.npy) ignore if set to None', default = '../Datasets/vgg16.npy')
    parser.add_argument('--num_epochs', help = 'Number of epochs', type = int, default = 5000)

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
