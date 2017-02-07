# ============================================================== #
#                    TFRecords Downloader                        #
#                                                                #
#                                                                #
# Download tfrecords for NYU Dataset                             #
# ============================================================== #

import tensorflow as tf
import os
import wget
import tarfile


def download_and_extract_tfrecords(download_training_records, download_testing_records, output_dir):
    """
    Downloads and extracts tfrecords
    ----------
    Args:
        download_training_records: bool, indicates if training records should be downloaded
        download_testing_records: bool, indicates if testing records should be downloaded
        output_dir: directory where tfrecords are saved
    """

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    urls = []

    if download_training_records:
        urls.append('https://transfer.sh/gwDSU/tfrecords-train-40-10.tar.gz')

    if download_testing_records:
        urls.append('https://transfer.sh/16lHPv/tfrecords-test-40-10.tar.gz')
    
    if len(urls) > 0:
        for url in urls:
            filepath = os.path.join(output_dir, 'tmp')
            wget.download(url, out = filepath)

            tar = tarfile.open(filepath)
            tar.extractall(path = output_dir)
            tar.close()

            os.remove(filepath)
        print('[INFO    ]\tTfrecords downloaded successfully')
