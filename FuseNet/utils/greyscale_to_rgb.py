import csv
import os
import numpy as np
import argparse
from PIL import Image

FLAGS = None

parser = argparse.ArgumentParser(description = 'Convert Grayscale image to RGB image')
parser.add_argument('--input_path', help = 'Path to input image')
parser.add_argument('--input_dir', help = 'Input directory, all files are converted')
parser.add_argument('--cmap_path', help = 'Path to cmap file', default='../../Raw/cmap.txt')
FLAGS, unparsed = parser.parse_known_args()

def one_hot(input):
    output = np.zeros((input.size, 256))
    output[np.arange(input.size), input] = 1
    return output

def get_cmap():
    with open(FLAGS.cmap_path, 'r') as cmap_file:
        r = csv.reader(cmap_file, delimiter='\t')
        values = []
        for row in r:
            values.append([float(x) for x in row])

    return np.array(values)

def convert_image(input_path):    
    with open(input_path, 'r') as input_image:
        grey_image = np.asarray(Image.open(input_image))
    
        input_shape = grey_image.shape
        grey_image = grey_image.reshape([-1])
        grey_image = one_hot(grey_image)
        rgb_image = grey_image.dot(cmap)
        rgb_image = rgb_image * 256
        rgb_image = rgb_image.astype(np.uint8)
        output_shape = input_shape + (3,)
        rgb_image = rgb_image.reshape(output_shape)
        output = Image.fromarray(rgb_image)

    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]

    output_filename = name + '_rgb' + extension

    output_path = os.path.join(directory, output_filename)
    
    output.save(output_path)


cmap = get_cmap()

if FLAGS.input_path is not None:
    convert_image(FLAGS.input_path)

if FLAGS.input_dir is not None:
    for f in os.listdir(FLAGS.input_dir):
        if f.endswith('png') or f.endswith('jpg') or f.endswith('jpeg'):
            file_path = os.path.join(FLAGS.input_dir, f)
            convert_image(file_path)
   
