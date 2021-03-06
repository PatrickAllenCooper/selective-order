#!/usr/bin/env python
"""Intended to establish a reasonable baseline for comparison via selective injection within cifar10."""

import foolbox as fb
import foolbox.attacks as fa
import eagerpy as ep
import numpy as np
import ast
import xlsxwriter
import os.path as os
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

__author__ = "Patrick Cooper"
__email__ = "patrick.allen.cooper@gmail.com"
__status__ = "Research"
__copyright__ = "Copyright 2020, Patrick Cooper, All rights reserved."
__source_url__ = "paper@journal.com"

tfds.disable_progress_bar()
tf.enable_v2_behavior()

NUM_CLASSES = 10
APPLY_TRANSFER = True
NUMBER_OF_SAMPLES = 5
BASE_SCALAR = 1
CONFIGURATION_DIRECTORY = "cifar10_configuration"
SHOW_DISTRIBUTION_GRAPH = False
BINNED_CYCLES = 3
PERFORM_GS = True

# Baseline model parameters
PERFORM_GS_OPT = False
BASELINE_EPOCHS = 5

log_dir = os.join('logs', 'scalars', datetime.now().strftime("%Y%m%d-%H%M%S"))
excel_log = os.join('excel_log', 'spreadsheets', 'Result_CIFAR10_' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".xlsx")

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
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
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES),
        tf.keras.layers.Activation('softmax')
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
    history = modified_standard_learner.fit(
        images,
        labels,
        epochs=epochs_b,
        validation_data=ds_test,
        callbacks=[tensorboard_callback]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epoch_results = np.array([[str(loss), str(acc), str(val_loss), str(val_acc)]])

    return epoch_results


# injects weights of adversarial models into  as given by distribution
def epoch_cycle(attack, epsilon,
                transfer, distribution,
                attacks_name, epsilon_name,
                transfer_method_name, distributions_name):
    a = 5.0
    n = 1000
    count, bins, ignored = plt.hist(distribution, bins=BINNED_CYCLES)
    x = np.linspace(0, 1, NUMBER_OF_SAMPLES)
    y = a * x ** (a - 1.)
    normed_y = n * np.diff(bins)[0] * y
    plt.title('Epoch Proportion')
    plt.xlabel('Epoch Set Number')
    plt.ylabel('Number Of Standard Training Injections')
    plt.legend()
    if SHOW_DISTRIBUTION_GRAPH:
        plt.show()

    agg_results = []

    count = np.array(count.astype(int))
    for x_entries in count:
        for amount_of_entries in range(x_entries):
            results_list = embed_models(amount_of_entries, BASE_SCALAR, attack, epsilon, transfer)
            results_list = np.insert(results_list, 0, x_entries, axis=1)
            results_list = np.insert(results_list, 0, distributions_name, axis=1)
            results_list = np.insert(results_list, 0, transfer_method_name, axis=1)
            results_list = np.insert(results_list, 0, epsilon_name, axis=1)
            results_list = np.insert(results_list, 0, attacks_name, axis=1).flatten()
            agg_results.append(results_list)

    return agg_results


def unroll_print(data):
    columns = ['Attack_Name', 'Epsilon_Name',
               'Transfer_Name', 'Distribution_Name',
               'Epoch Number', 'Loss', 'Accuracy',
               'Validation Loss', 'Validation Accuracy']
    log = []
    for distribution in data:
        for transfer_method in distribution:
            transfer_method = transfer_method.tolist()
            log.append(transfer_method)

    with xlsxwriter.Workbook(excel_log) as workbook:
        worksheet = workbook.add_worksheet()

        for row_num, record in enumerate(log):
            worksheet.write_row(row_num, 0, record)


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

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

baseline_model = build_model()

baseline_model.fit(
    ds_train,
    epochs=15,
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

attacks_names = [
    "FGSM",
    "LinfPGD",
    "LinfBasicIterativeAttack",
    "LinfAdditiveUniformNoiseAttack",
    "LinfDeepFoolAttack"
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
    #"Transfer With Last Layer",
    #"Transfer With No Last Layer",
    "Direct Copy Weights"
]

# TODO: Add distribution arguments, only independent argument ought to be distribution center
distributions = [
    np.random.poisson(lam=1, size=NUMBER_OF_SAMPLES),
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

distributions_names = [
    "poisson",
    "poisson"
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

if os.isfile(CONFIGURATION_DIRECTORY) and PERFORM_GS is False:
    file = open(CONFIGURATION_DIRECTORY, "r")
    contents = file.read()
    config = ast.literal_eval(contents)
    attack, epsilon, transfer, distribution = (config[key] for key in ['Attack', 'Epsilon', 'Transfer_Methods',
                                                                       'Distribution'])
    print("Currently starting attack " + attack + " with epsilon " + epsilon + " and transfer method " + transfer + ".")
    attack = attacks[int(attack)]
    distribution = distributions[int(distribution)]
    data = epoch_cycle(attack, epsilon, transfer, distribution)
    unroll_print(data)
    file.close()

else:
    data = []
    for a_idx, attack in enumerate(attacks):
        for e_idx, epsilon in enumerate(epsilons):
            for t_idx, transfer in enumerate(transfer_methods):
                print("Currently starting attack " + str(attacks_names[a_idx]) + " with epsilon " + str(epsilon) +
                      " and transfer method " + transfer + ".")
                for d_idx, distribution in enumerate(distributions):
                    data.append(epoch_cycle(attack, epsilon, transfer, distribution, attacks_names[a_idx],
                                            str(epsilon), transfer_methods, distributions_names[d_idx]))

    unroll_print(data)

print("Code is the result of research performed by " + __author__ + " for the paper " + __source_url__ + ". For more"
                                                                    " information please contact " + __email__ + ".")