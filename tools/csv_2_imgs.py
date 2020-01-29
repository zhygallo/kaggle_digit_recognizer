from __future__ import absolute_import, print_function, division, unicode_literals

import csv
import skimage.io
import os
import click
import numpy as np


@click.command()
@click.argument('csv_file', type=click.STRING)
@click.argument('data_mode', type=click.STRING)
@click.option('--output_fold', type=click.STRING, default='', help='path to the output folder with images')
def gen_imgs_from_csv(csv_file, data_mode, output_fold=''):
    labels = []
    images = []
    with open(csv_file) as csv_file:
        scv_reader = csv.reader(csv_file, delimiter=',')
        next(scv_reader)
        img_id = 0
        for data in scv_reader:
            if data_mode == 'train':
                label = data[0]
                image = data[1:]
            else:
                label = 'images'
                image = data

            image = np.array(image, dtype=int)
            image = image.reshape(28, 28)

            if output_fold != '':
                if not os.path.isdir(output_fold):
                    os.makedirs(output_fold, exist_ok=True)
                full_fold_path = os.path.join(output_fold, data_mode, label)
                if not os.path.isdir(full_fold_path):
                    os.makedirs(full_fold_path, exist_ok=True)
                skimage.io.imsave(os.path.join(full_fold_path, str(img_id).zfill(5) + '.png'),
                                  image.astype('uint8'))
                img_id += 1

            labels.append(label)
            images.append(image)

    return images, labels


if __name__ == '__main__':
    gen_imgs_from_csv()
