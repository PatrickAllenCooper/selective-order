#!/usr/bin/env python
"""Intended to establish a reasonable baseline for comparison via selective injection within mnist."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import foolbox as fb
import foolbox.attacks as fa
import eagerpy as ep
import numpy as np
from datetime import datetime

__author__ = "Patrick Cooper"
__email__ = "patrick.allen.cooper@gmail.com"
__status__ = "Research"
__copyright__ = "Copyright 2020, Patrick Cooper, All rights reserved."
__source_url__ = "paper@journal.com"

tfds.disable_progress_bar()
tf.enable_v2_behavior()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

log_dir = "logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

baseline_model = build_model()

baseline_model.fit(
    ds_train,
    epochs=1,
    validation_data=ds_test,
    callbacks=[tensorboard_callback]
)


for images, labels in ds_train.take(1):  # only take first element of dataset
    images_ex = ep.astensors(images)
    labels_ex = ep.astensors(labels)

fmodel = fb.TensorFlowModel(baseline_model, bounds=(0, 1))

attacks = [
    fa.FGSM(),
    fa.LinfPGD(),
    fa.LinfBasicIterativeAttack(),
    fa.LinfAdditiveUniformNoiseAttack(),
    fa.LinfDeepFoolAttack(),
]

epsilons = [
    0.0,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.1,
    0.3,
    0.5,
    1.0,
]

print("epsilons")
print(epsilons)
print("")

attack = fa.FGSM()
epsilon = 0.005
_, adv, success = attack(fmodel, images, labels, epsilons=epsilon)

"""
attack_success = np.zeros((len(attacks), len(epsilons), len(images)), dtype=np.bool)
for i, attack in enumerate(attacks):
    _, adv, success = attack(fmodel, images, labels, epsilons=epsilons)
    assert success.shape == (len(epsilons), len(images))
    success_ = success.numpy()
    assert success_.dtype == np.bool
    attack_success[i] = success_
    print(attack)
    print("  ", 1.0 - success_.mean(axis=-1).round(2))


robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
print("")
print("-" * 79)
print("")
print("worst case (best attack per-sample)")
print("  ", robust_accuracy.round(2))

"""

model2 = build_model()

model2.fit(
    adv,
    labels,
    epochs=1,
    validation_data=ds_test,
    callbacks=[tensorboard_callback]
)

model3 = build_model()

#model3.set_weights(model2.get_weights())

model3.fit(
    images,
    labels,
    epochs=5,
    validation_data=ds_test,
    callbacks=[tensorboard_callback]
)

print("Code is the result of research performed by " + __author__ + " for the paper " + __source_url__ + ". For more"
                                                                    " information please contact " + __email__ + ".")
