"""CNN experiments on the German Traffic Sign Recognition Benchmark."""

import logging
import os

__version__ = "0.1.0"

# Quiet TensorFlow's C++ logger unless the environment already overrides it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Scripts attach their own handlers. A null handler keeps library imports quiet in tests.
_package_logger = logging.getLogger("traffic_signs")
if not _package_logger.handlers:
    _package_logger.addHandler(logging.NullHandler())
_package_logger.propagate = False
