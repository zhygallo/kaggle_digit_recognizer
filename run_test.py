from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
import click
import csv
import os


@click.command()
@click.argument('input_img_folder', type=click.STRING)
@click.argument('input_model_file', type=click.STRING)
@click.option('--test_img_shape', nargs=2, type=click.INT, default=(28, 28))
@click.option('--batch_size', type=click.INT, default=16)
@click.option('--output_folder', type=click.STRING, default='output')
def main(input_img_folder, input_model_file, test_img_shape, batch_size, output_folder):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    datagen_config = {'target_size': test_img_shape,
                      'batch_size': batch_size,
                      'class_mode': 'categorical',
                      'color_mode': 'grayscale',
                      'shuffle': False}

    train_generator = test_datagen.flow_from_directory(input_img_folder, **datagen_config)

    model = tf.keras.models.load_model(input_model_file, compile=False)

    pred_probs = model.predict_generator(train_generator,
                                         steps=train_generator.n // train_generator.batch_size,
                                         verbose=1)

    predictions = pred_probs.argmax(axis=-1)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'result.csv'), 'w+', newline='\n') as csvfile:
        header = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for idx, pred in enumerate(predictions):
            writer.writerow({'ImageId': str(idx+1), 'Label': str(pred)})


if __name__ == "__main__":
    main()
