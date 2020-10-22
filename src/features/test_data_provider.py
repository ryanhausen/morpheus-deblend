import src.features.data_provider as dp

import gin


def test_get_dataset():
    batch_size = 5
    expected_x_shape = (batch_size, 512, 512, 1)
    expected_y_shape = (batch_size, 512, 512, 5)

    (train, _, _, _) = dp.get_dataset(batch_size)

    x_train, y_train = next(iter(train))

    assert x_train.shape == expected_x_shape
    assert y_train.shape == expected_y_shape


if __name__ == "__main__":
    test_get_dataset()
