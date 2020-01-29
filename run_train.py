from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
import click
import os


@click.command()
@click.argument('input_folder', dtype=click.STRING)
@click.option('--train_img_shape', nargs=2, dtype=click.INT, default=(28, 28))
@click.option('--num_epochs', dtype=click.INT, default=20)
@click.option('--batch_size', dtype=click.INT, default=16)
@click.option('--learn_rate', dtype=click.FLOAT, default=1e-4)
@click.option('--output_folder', dtype=click.STRING, default='output')
def main(input_folder, train_img_shape, num_epochs, batch_size, learn_rate, output_folder):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                    validation_split=0.2)

    datagen_config = {'target_size': train_img_shape,
                      'batch_size': batch_size,
                      'class_mode': 'categorical',
                      'color_mode': 'grayscale'}

    train_generator = train_datagen.flow_from_directory(input_folder, stubset='training', **datagen_config)
    validation_generator = train_datagen.flow_from_directory(input_folder, stubset='validation', **datagen_config)

    input_shape = (train_img_shape[0], train_img_shape[1], 1)
    model = tf.keras.Sequential(
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    )

    model.compile(optimizer=tf.optimizers.Adam(learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    save_best_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(output_folder, 'model.h5'),
                                                            monitor='val_loss',
                                                            save_best_only=True)
    model.fit(train_generator,
              epochs=num_epochs,
              steps_per_epoch=train_generator.n // train_generator.batch_size,
              callbacks=[save_best_callback],
              validation_data=validation_generator,
              validation_steps=validation_generator.n)


if __name__ == '__main__':
    main()
