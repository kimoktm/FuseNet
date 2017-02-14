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

    training_tfrecords   = glob.glob(os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'training'))
    validation_tfrecords = glob.glob(os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'validation'))

    if not training_tfrecords or not validation_tfrecords:
        print('[INFO    ]\tNo train tfrecords found. Downloading them in %s' % FLAGS.tfrecords_dir)
        tfrecords_downloader.download_and_extract_tfrecords(True, True, False, FLAGS.tfrecords_dir)


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

    data_files, data_size = load_datafiles('training')
    images, depths, annots, classes, filenames = dataset_loader.inputs(
                                                     data_files = data_files,
                                                     image_size = FLAGS.image_size,
                                                     batch_size = FLAGS.batch_size,
                                                     num_epochs = FLAGS.num_epochs,
                                                     train = True)

    validation_files, validation_size = load_datafiles('validation')
    val_images, val_depths, val_annots, val_classes, _ = dataset_loader.inputs(
                                                     data_files = validation_files,
                                                     image_size = FLAGS.image_size,
                                                     batch_size = validation_size,
                                                     num_epochs = FLAGS.num_epochs,
                                                     train = True)

    data_image   = tf.placeholder(tf.float32, shape = (None, FLAGS.image_size, FLAGS.image_size, 3))
    data_depth   = tf.placeholder(tf.float32, shape = (None, FLAGS.image_size, FLAGS.image_size, 1))
    data_annots  = tf.placeholder(tf.float32, shape = (None, FLAGS.image_size, FLAGS.image_size, 1))
    data_classes = tf.placeholder(tf.int64,   shape = (None))

    annot_logits, class_logits = fusenet.build(data_image, data_depth, FLAGS.num_annots, FLAGS.num_classes, True)

    total_acc, seg_acc, class_acc = fusenet.accuracy(annot_logits, data_annots, class_logits, data_classes)

    loss = fusenet.loss(annot_logits, data_annots, class_logits, data_classes, FLAGS.weight_decay_rate)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = fusenet.train(loss, FLAGS.learning_rate, FLAGS.learning_rate_decay_steps, FLAGS.learning_rate_decay_rate, global_step)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()

    session_manager = tf.train.SessionManager()

    sess = session_manager.prepare_session("", init_op = init_op, saver = saver, checkpoint_dir = FLAGS.checkpoint_dir, init_fn = initialize_session)
    
    writer = tf.train.SummaryWriter(FLAGS.checkpoint_dir + "/train_logs", sess.graph)

    merged = tf.summary.merge_all()

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    start_time = time.time()

    curr_val_acc = 0.0

    try:
        while not coord.should_stop():
            step = tf.train.global_step(sess, global_step)

            image_batch, depth_batch, annots_batch, classes_batch = sess.run([images, depths, annots, classes])
            feed_dict_train = {data_image : image_batch, data_depth : depth_batch, data_annots : annots_batch, data_classes : classes_batch}
            
            _, loss_value, summary = sess.run([train_op, loss, merged], feed_dict = feed_dict_train)
            writer.add_summary(summary, step)

            if step % 1000 == 0:
                image_val, depth_val, annots_val, classes_val = sess.run([val_images, val_depths, val_annots, val_classes])
                feed_dict_val   = {data_image : image_val, data_depth : depth_val, data_annots : annots_val, data_classes : classes_val}

                acc_total_value, acc_seg_value, acc_clss_value = sess.run([total_acc, seg_acc, class_acc], feed_dict = feed_dict_train)
                val_acc_total_value, val_acc_seg_value, val_acc_clss_value = sess.run([total_acc, seg_acc, class_acc], feed_dict = feed_dict_val)

                if val_acc_seg_value > curr_val_acc:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'fusenet_top_validation.ckpt')
                    saver.save(sess, checkpoint_path, global_step = global_step)
                    curr_val_acc = val_acc_seg_value
                    improved_str = '*'
                else:
                    improved_str = ''

                epoch = step * FLAGS.batch_size / data_size
                duration = time.time() - start_time
                start_time = time.time()

                print('[PROGRESS]\tEpoch %d, Step %d: loss = %.2f (%.3f sec)' % (epoch, step, loss_value, duration))
                print('\t\tTraining   segmentation accuracy = %.2f, classifcation accuracy = %.2f, total accuracy = %.2f'
                     % (acc_seg_value, acc_clss_value, acc_total_value))
                print('\t\tValidation segmentation accuracy = %.2f, classifcation accuracy = %.2f, total accuracy = %.2f %s\n'
                     % (val_acc_seg_value, val_acc_clss_value, val_acc_total_value, improved_str))

            if step % 5000 == 0:
                print('[PROGRESS]\tSaving checkpoint')
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'fusenet.ckpt')
                saver.save(sess, checkpoint_path, global_step = global_step)

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    writer.close()
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
    parser.add_argument('--checkpoint_dir', help = 'Checkpoints directory', default = '../Datasets/NYU/checkpoints/')
    parser.add_argument('--num_annots', help = 'Number of segmentation labels', type = int, default = 41)
    parser.add_argument('--num_classes', help = 'Number of Classification labels', type = int, default = 11)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--learning_rate', help = 'Learning rate', type = float, default = 10e-5)
    parser.add_argument('--learning_rate_decay_steps', help = 'Learning rate decay steps', type = int, default = 50000)
    parser.add_argument('--learning_rate_decay_rate', help = 'Learning rate decay rate', type = float, default = 0.98)
    parser.add_argument('--weight_decay_rate', help = 'Weight decay rate', type = float, default = 0.0005)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 4)
    parser.add_argument('--vgg_path', help = 'VGG weights path (.npy) ignore if set to None', default = '../Datasets/vgg16.npy')
    parser.add_argument('--num_epochs', help = 'Number of epochs', type = int, default = 5000)

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
