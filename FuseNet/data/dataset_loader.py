# ============================================================== #
#                        Dataset Loader                          #
#                                                                #
#                                                                #
# Processing occurs on a single image at a time. Images are      #
# read and preprocessed in parallel across multiple threads. A   #
# Batch is then formed from these data to be used for training   #
# or evaluation                                                  #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf


IMAGE_FORMAT = 'PNG'


def inputs(data_files, train, batch_size, image_size, 
                     num_epochs, num_preprocess_threads = 4):
    """
    Generate shuffled batches from dataset images:
    ----------
    Args:
        data_files: string, array of shared tensor-records
        train: boolean, is in training mode (shuffle data)
        batch_size: integer, number of examples in batch
        image_size: integer, size used to resize loaded image (w & h)
        num_epochs: integer, number of epochs
        num_preprocess_threads: integer, total number of preprocessing threads

    Returns:
        images: Colored images. 4D tensor of size [batch_size, image_size, image_size, 3]
        depths: Depth images. 4D tensor of size [batch_size, image_size, image_size, 1]
        images: Annotations images. 4D tensor of size [batch_size, image_size, image_size, 1]
        classes: 1-D integer Tensor of [batch_size]
    """

    images, depths, annots, classes = batch_inputs(data_files = data_files, 
                    batch_size = batch_size, image_size = image_size,
                    train = train, num_epochs = num_epochs,
                    num_preprocess_threads = num_preprocess_threads)

    return images, depths, annots, classes


def image_preprocessing(image, image_size, is_color, scope = None):
    """
    Process & resized one image:
    ----------
    Args:
        image: 3-D float Tensor
        image_size: integer
        is_color: is color image (3 channels)
        scope: optional scope

    Returns:
        image: resized 3-D float Tensor
    """

    with tf.name_scope(scope, 'process_image', [image, image_size, image_size]):
        image = tf.expand_dims(image, 0)

        if is_color:
            image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
        else:
            image = tf.image.resize_nearest_neighbor(image, [image_size, image_size], align_corners=False)

        image = tf.squeeze(image, [0])

    return image


def parse_example_proto(example_serialized, image_format):
    """
    Parses an Example proto containing a training example of an image
    and decode image content according to their corresponding format:
    ----------
    Args:
        example_serialized: scalar Tensor tf.string containing a serialized
        image_format: string, image format used for decoding

    Returns:
        color_image: Tensor decoded color image
        depth_image: Tensor decoded depth image
        annot_image: Tensor decoded annotation image
        clss: Tensor image class id
    """

    feature_map = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], dtype = tf.string, default_value = ''),
        'image/class': tf.FixedLenFeature([1], dtype = tf.int64, default_value = -1),
        'image/encoded/color': tf.FixedLenFeature([], dtype = tf.string, default_value = ''),
        'image/encoded/depth': tf.FixedLenFeature([], dtype = tf.string, default_value = ''),
        'image/encoded/annot': tf.FixedLenFeature([], dtype = tf.string, default_value = '')                           
    }

    features     = tf.parse_single_example(example_serialized, feature_map)
    height       = tf.cast(features['image/height'], dtype = tf.int32)
    width        = tf.cast(features['image/width'], dtype = tf.int32)
    clss         = tf.cast(features['image/class'], dtype = tf.int32)

    if image_format.lower().endswith(('png')):
      color_image = tf.image.decode_png(features['image/encoded/color'])
      depth_image = tf.image.decode_png(features['image/encoded/depth'])
      annot_image = tf.image.decode_png(features['image/encoded/annot'])
    else:
      color_image = tf.image.decode_jpeg(features['image/encoded/color'])
      depth_image = tf.image.decode_jpeg(features['image/encoded/depth'])
      annot_image = tf.image.decode_jpeg(features['image/encoded/annot'])

    color_shape = tf.pack([height, width, 3])
    annot_shape = tf.pack([height, width, 1])

    color_image = tf.reshape(color_image, color_shape)
    depth_image = tf.reshape(depth_image, annot_shape)
    annot_image = tf.reshape(annot_image, annot_shape)

    return color_image, depth_image, annot_image, clss


def batch_inputs(data_files, batch_size, image_size, train, num_epochs, num_preprocess_threads):
    """
    Contruct batches of training or evaluation examples from the image dataset:
    ----------
    Args:
        data_files: string, array of shared tensor-records
        batch_size: integer, size of each batch
        image_size: integer, size used to resize loaded image (w & h)
        num_epochs: integer, number of epochs
        num_preprocess_threads: integer, total number of preprocessing threads

    Returns:
        images: 4-D float Tensor of a batch of resized color images
        depths: 4-D float Tensor of a batch of resized depth images
        annots: 4-D float Tensor of a batch of resized annotation images
        classes: 4-D float Tensor of a batch of image class ids
    """

    with tf.name_scope('batch_processing'):
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle = True, num_epochs = num_epochs)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle = False, num_epochs = num_epochs)

        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

        color_image, depth_image, annot_image, clss = parse_example_proto(example_serialized, IMAGE_FORMAT)
        color_image = image_preprocessing(color_image, image_size, is_color = True)
        depth_image = image_preprocessing(depth_image, image_size, is_color = False)
        annot_image = image_preprocessing(annot_image, image_size, is_color = False)

        color_image = tf.cast(color_image, tf.float32)
        depth_image = tf.cast(depth_image, tf.float32)
        annot_image = tf.cast(annot_image, tf.float32)
        clss = tf.cast(clss, tf.int64)

        # color_image = tf.cast(color_image, tf.uint8)
        # depth_image = tf.cast(depth_image, tf.uint8)
        # annot_image = tf.cast(annot_image, tf.uint8)
        # clss = tf.cast(clss, tf.int64)

        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size

        images, depths, annots, classes = tf.train.shuffle_batch(
                [color_image, depth_image, annot_image, clss],
                batch_size = batch_size, num_threads = num_preprocess_threads,
                capacity = capacity,
                min_after_dequeue = min_after_dequeue)

        classes = tf.reshape(classes, [batch_size])

        return images, depths, annots, classes
