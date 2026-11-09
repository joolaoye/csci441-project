"""CNN depth, output shape, and the shared pooling schedule."""

import numpy as np
from tensorflow.keras import layers

from traffic_signs.models.cnn import (
    build_cnn,
    final_spatial_size,
    max_pool_count,
    pooling_schedule,
)


def test_pool_count_depends_on_image_size_not_depth():
    assert max_pool_count(48, 12) == 2
    assert pooling_schedule(48, 2, 12) == [1, 1]
    assert pooling_schedule(48, 3, 12) == [0, 1, 1]
    assert pooling_schedule(48, 4, 12) == [0, 1, 0, 1]
    assert final_spatial_size(48, 2, 12) == 12
    assert final_spatial_size(48, 3, 12) == 12
    assert final_spatial_size(48, 4, 12) == 12


def test_extra_pools_stack_when_there_are_fewer_layers_than_pools():
    assert pooling_schedule(32, 2, 4) == [1, 2]
    assert final_spatial_size(32, 2, 4) == 4


def test_output_shape_and_softmax_for_several_depths():
    for channels in ([4, 8], [4, 8, 8], [4, 8, 8, 8]):
        model = build_cnn(
            image_size=48,
            num_classes=43,
            channels=channels,
            kernel_size=3,
            dropout=0.2,
            dense_units=16,
            min_spatial=12,
        )
        output = model(np.zeros((2, 48, 48, 3), dtype=np.float32), training=False).numpy()
        assert output.shape == (2, 43)
        np.testing.assert_allclose(output.sum(axis=-1), 1.0, atol=1e-5)
        convolutions = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
        pools = [layer for layer in model.layers if isinstance(layer, layers.MaxPooling2D)]
        assert len(convolutions) == len(channels)
        assert all(layer.padding == "same" for layer in convolutions)
        assert len(pools) == 2


def test_stacked_pools_are_real_layers():
    model = build_cnn(
        image_size=32,
        num_classes=2,
        channels=[2, 2],
        kernel_size=3,
        dropout=0.0,
        dense_units=4,
        min_spatial=4,
    )
    pools = [layer for layer in model.layers if isinstance(layer, layers.MaxPooling2D)]
    assert len(pools) == 3
    output = model(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False).numpy()
    assert output.shape == (1, 2)
