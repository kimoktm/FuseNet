# ============================================================== #
#                         Fusnet eval                            #
#                                                                #
#                                                                #
# Eval fusnet with processed dataset in tfrecords format         #
# ============================================================== #

import tensorflow as tf

import os
import glob
import argparse
import data.dataset_loader as dataset_loader
import data.tfrecords_downloader as tfrecords_downloader
import fusenet

FLAGS = None

def maybe_download_and_extract():
    """
    Check if tfrecords exist if not download them
    (processed dataset into tfrecords with 40 labels & 10 classes)
    """
    if not tf.gfile.Exists(FLAGS.tfrecords_dir):
        tf.gfile.MakeDirs(FLAGS.tfrecords_dir)

    testing_tfrecords = glob.glob(os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'test'))

    if not testing_tfrecords:
        print('[INFO    ]\tNo test tfrecords found. Downloading them in %s' %FLAGS.tfrecords_dir)
        tfrecords_downloader.download_and_extract_tfrecords(False, True, FLAGS.tfrecords_dir)
    
        
def load_datafiles():
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '%s-*' % 'test')
    data_files = tf.gfile.Glob(tf_record_pattern)

    return data_files

def evaluate():
    """
    Eval fusenet using specified args:
    """

    data_files = load_datafiles()
    images, depths, annots, classes = dataset_loader.inputs(
                                            data_files = data_files,
                                            image_size = FLAGS.image_size,
                                            batch_size = FLAGS.batch_size,
                                            num_epochs = 1,
                                            train = False)

    annot_logits, class_logits = fusenet.build(images, depths, FLAGS.num_annots, FLAGS.num_classes, False)
    total_acc, seg_acc , class_acc = fusenet.accuracy(annot_logits, annots, class_logits, classes)
    
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

    try:
        step = 0
        while not coord.should_stop():
            acc_value = sess.run(seg_acc)
            print('[PROGRESS]\tSegmentation accuracy for current batch: %.3f' % acc_value)
            step += 1

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone evaluating in %d steps.' % step)

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    

def main(_):
    """
    Run fusenet prediction on input tfrecords
    """
    maybe_download_and_extract()
    evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Eval FuseNet on given tfrecords.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory', default = '../Datasets/NYU/tfrecords')
    parser.add_argument('--checkpoint_path', help = 'Path of checkpoint to restore', default = '../Datasets/NYU/checkpoints-0')
    parser.add_argument('--num_annots', help = 'Number of segmentation labels', type = int, default = 41)
    parser.add_argument('--num_classes', help = 'Number of Classification labels', type = int, default = 11)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 4)
    parser.add_argument('--visualize', help = 'Visualize predicted annotations', type = bool, default = False)
    
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()