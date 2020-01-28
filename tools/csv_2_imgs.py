from __future__ import absolute_import, print_function, division, unicode_literals

import csv
import skimage.io
import os
import click
import numpy as np


@click.command()
@click.argument('csv_file', type=click.STRING)
@click.option('--output_fold', type=click.STRING, default='', help='path to the output folder with images')
def gen_imgs_from_csv(csv_file, output_fold=''):
    labels = []
    images = []
    with open(csv_file) as csv_file:
        scv_reader = csv.reader(csv_file, delimiter=',')
        next(scv_reader)
        for data in scv_reader:
            label = data[:1]
            image = data[1:]
            image = np.array(image, dtype=int)
            image = image.reshape(28, 28)

            labels.append(label)
            images.append(image)

    return images, labels


if __name__ == '__main__':
    gen_imgs_from_csv()
