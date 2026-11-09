"""Keep the suite on CPU and limit TensorFlow's startup noise."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    # Tests still pass when TensorFlow has already chosen a device.
    pass
