"""A small CNN whose depth is the length of ``channels``.

Every depth uses the same head and the same number of pooling stages.
The pool count is chosen from the image size and ``min_spatial`` only, then
spread across the convolutional blocks. On a 48×48 input with
``min_spatial`` 12, two pooling stages reduce the map to 12×12 whether the
network has two convolutional layers or four. Extra layers add convolution
without extra downsampling, which keeps the depth comparison from also being
a resolution comparison.

Global average pooling then makes the classifier input depend on the last
channel count rather than on the feature-map area.
"""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def max_pool_count(image_size: int, min_spatial: int) -> int:
    """How many 2×2 pools keep each side at least ``min_spatial`` pixels."""
    if image_size < 1 or min_spatial < 1:
        raise ValueError("image_size and min_spatial must be positive.")
    if image_size < min_spatial:
        raise ValueError(
            f"image_size ({image_size}) is smaller than min_spatial ({min_spatial})."
        )
    spatial = image_size
    pools = 0
    while spatial // 2 >= min_spatial:
        spatial //= 2
        pools += 1
    return pools


def pooling_schedule(image_size: int, num_layers: int, min_spatial: int) -> list[int]:
    """Return how many 2×2 pools follow each convolutional block.

    The total always equals :func:`max_pool_count`. When there are more blocks
    than pools, the pools are spaced through the network. When there are fewer
    blocks, the extra pools are stacked after the last block so the final
    resolution does not depend on depth.
    """
    if num_layers < 1:
        raise ValueError("The model needs at least one convolutional layer.")
    total = max_pool_count(image_size, min_spatial)
    if total == 0:
        return [0] * num_layers
    if num_layers <= total:
        schedule = [1] * (num_layers - 1)
        schedule.append(total - (num_layers - 1))
        return schedule
    schedule = [0] * num_layers
    for pool_number in range(1, total + 1):
        layer_index = math.ceil(pool_number * num_layers / total) - 1
        schedule[layer_index] += 1
    if sum(schedule) != total:
        raise RuntimeError("The pooling schedule does not match the planned pool count.")
    return schedule


def final_spatial_size(image_size: int, num_layers: int, min_spatial: int) -> int:
    """Feature-map size after the convolutional stack, before global pooling."""
    spatial = image_size
    for pools in pooling_schedule(image_size, num_layers, min_spatial):
        spatial //= 2**pools
    return spatial


def build_cnn(
    image_size: int,
    num_classes: int,
    channels: list[int],
    kernel_size: int,
    dropout: float,
    dense_units: int,
    min_spatial: int,
) -> keras.Model:
    """Build the traffic-sign CNN described in this module."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    schedule = pooling_schedule(image_size, len(channels), min_spatial)
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    features = inputs
    for index, filters in enumerate(channels):
        block = index + 1
        # Batch normalization supplies the shift, so the convolution has no bias.
        features = layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            use_bias=False,
            name=f"conv_{block}",
        )(features)
        features = layers.BatchNormalization(name=f"bn_{block}")(features)
        features = layers.Activation("relu", name=f"relu_{block}")(features)
        for pool_index in range(schedule[index]):
            suffix = "" if schedule[index] == 1 else f"_{pool_index + 1}"
            features = layers.MaxPooling2D(pool_size=2, name=f"pool_{block}{suffix}")(features)
    features = layers.GlobalAveragePooling2D(name="average_pool")(features)
    features = layers.Dropout(dropout, name="dropout")(features)
    features = layers.Dense(dense_units, activation="relu", name="dense")(features)
    features = layers.Dropout(dropout, name="dropout_dense")(features)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(features)
    return keras.Model(inputs, outputs, name="traffic_sign_cnn")


def build_model_from_config(config: dict) -> keras.Model:
    """Build the CNN described by a resolved experiment configuration."""
    model_config = config["model"]
    return build_cnn(
        image_size=config["data"]["image_size"],
        num_classes=config["data"]["num_classes"],
        channels=list(model_config["channels"]),
        kernel_size=model_config["kernel_size"],
        dropout=model_config["dropout"],
        dense_units=model_config["dense_units"],
        min_spatial=model_config["min_spatial"],
    )


def parameter_counts(model: keras.Model) -> dict:
    """Return trainable and total parameter counts."""
    trainable = int(sum(int(tf.size(weight)) for weight in model.trainable_weights))
    total = int(sum(int(tf.size(weight)) for weight in model.weights))
    return {"trainable_parameters": trainable, "total_parameters": total}
