#!/usr/bin/env python
"""Intended to establish a reasonable baseline for comparison via selective injection within mnist."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import pandas as pd
import foolbox as fb
import foolbox.attacks as fa
import eagerpy as ep
import numpy as np
import scipy as sp
import ast
import os.path as os
import matplotlib.pyplot as plt
from datetime import datetime

__author__ = "Patrick Cooper"
__email__ = "patrick.allen.cooper@gmail.com"
__status__ = "Research"
__copyright__ = "Copyright 2020, Patrick Cooper, All rights reserved."
__source_url__ = "paper@journal.com"

tfds.disable_progress_bar()
tf.enable_v2_behavior()

APPLY_TRANSFER = True
BASE_SCALAR = 1
CONFIGURATION_DIRECTORY = "mnist_configuration.txt"
SHOW_DISTRIBUTION_GRAPH = True

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


# copies model_a to model_b
def apply_transfer(model_a, model_b, transfer):
    if APPLY_TRANSFER:
        if transfer == "Direct Copy Weights":
            model_b.set_weights(model_a.get_weights())

    return model_b


# nest one model within another
def embed_models(epochs_a, epochs_b, attack, epsilon, transfer):

    # Apply selected method
    _, adv, success = attack(fmodel, images, labels, epsilons=epsilon)

    adversarial_learner = build_model()
    adversarial_learner.fit(
        adv,
        labels,
        epochs=epochs_a,
        validation_data=ds_test,
        callbacks=[]
    )

    standard_learner = build_model()
    modified_standard_learner = apply_transfer(adversarial_learner, standard_learner, transfer)
    train_history = modified_standard_learner.fit(
        images,
        labels,
        epochs=epochs_b,
        validation_data=ds_test,
        callbacks=[tensorboard_callback]
    )

    epoch_results = np.array([[12, 13, 14, 15, 16]])
    
    return epoch_results


# injects weights of adversarial models into  as given by distribution
def epoch_cycle(attack, epsilon, transfer, distribution):
    # display epoch cycle distribution used
    # shape
    a = 5.0
    n = 1000
    count, bins, ignored = plt.hist(distribution, bins=2)
    x = np.linspace(0, 1, 100)
    y = a * x ** (a - 1.)
    normed_y = n * np.diff(bins)[0] * y
    plt.title('Epoch Proportion')
    plt.xlabel('Epoch Set Number')
    plt.ylabel('Number Of Standard Training Injections')
    plt.legend()
    if SHOW_DISTRIBUTION_GRAPH:
        plt.show()

    count = count.astype(int)
    for x_entries in range(count.size):
        for amount_of_entries in count:
            results_list = embed_models(amount_of_entries, BASE_SCALAR, attack, epsilon, transfer)

    return results_list


def unroll_print(results):
        data = pd.DataFrame(results, columns=['Epoch Number', 'Loss', 'Accuracy', 'Validation Loss',
                                                        'Validation Accuracy'])

        excel_log = "excel_log\\spreadsheets\\Result_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".xlsx"
        data.to_excel(excel_log, index=False)


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

transfer_methods = [
    "Transfer With Last Layer",
    "Transfer With No Last Layer",
    "Direct Copy Weights"
]

# TODO: Add distribution arguments
distributions = [
    np.random.poisson(lam=1, size=100),
    """
    np.random.power(),
    np.random.beta(),
    np.random.dirichlet(),
    np.random.f(),
    np.random.gamma(),
    np.random.geometric(),
    np.random.logistic(),
    np.random.multinomial(),
    np.random.normal(),
    np.random.pareto(),
    np.random.lognormal(),
    np.random.uniform(),
    """
]


print("epsilons")
print(epsilons)
print("")

# Evaluation of adversarial methods used
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

results = np.empty([3, 5])

if os.isfile(CONFIGURATION_DIRECTORY):
    file = open(CONFIGURATION_DIRECTORY, "r")
    contents = file.read()
    config = ast.literal_eval(contents)
    attack, epsilon, transfer, distribution = (config[key] for key in ['Attack', 'Epsilon', 'Transfer_Methods',
                                                                       'Distribution'])
    data = epoch_cycle(attack, epsilon, transfer, distribution)
    unroll_print(data)
    for i in enumerate(data):
        datum = data[i]
        results.append(pd.DataFrame(datum, columns=['Epoch Number', 'Loss', 'Accuracy', 'Validation Loss',
                                                    'Validation Accuracy']))
    file.close()

else:
    # grid search attacks, epsilons, transfer methods, distributions
    # evaluates based on test acc
    for attack in attacks:
        for epsilon in epsilons:
            for transfer in transfer_methods:
                for distribution in distributions:
                    # solution for best available data
                    data = epoch_cycle(attack, epsilon, transfer, distribution)
                    unroll_print(data)

# TODO: Generalize capability of
# TODO: Introduce capacity for composite transfer modes. ltg
# TODO: Add shap explainers, only for best model. ltg.
print("Code is the result of research performed by " + __author__ + " for the paper " + __source_url__ + ". For more"
                                                                    " information please contact " + __email__ + ".")
