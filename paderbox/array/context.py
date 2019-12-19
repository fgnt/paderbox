import numpy as np

from paderbox.array.rearrange import tbf_to_tbchw


def stack_context(X, left_context=0, right_context=0, step_width=1):
    """ Stack TxBxF format with left and right context.

    There is a notebook, which illustrates this feature with many details in
    the example notebooks repository.

    :param X: Data with TxBxF format.
    :param left_context: Length of left context.
    :param right_context: Length of right context.
    :param step_width: Step width.
    :return: Stacked features with symmetric padding and head and tail.
    """
    X_stacked = tbf_to_tbchw(
        X,
        left_context=left_context,
        right_context=right_context,
        step_width=step_width
    )[:, :, 0, :].transpose((0, 1, 3, 2))

    T, B, F, W = X_stacked.shape
    X_stacked = X_stacked.reshape(T, B, F * W)

    return X_stacked


def unstack_context(X, mode, left_context=0, right_context=0, step_width=1):
    """ Unstacks stacked features.

    This only works in special cases. Right now, only mode='center'
    is supported. It will return just the center frame and drop the remaining
    parts.

    Other options are related to combining overlapping context frames.

    :param X: Stacked features (or output of your network)
    :param X: mode
    :param left_context: Length of left context.
    :param right_context: Length of right context.
    :param step_width: Step width.
    :return: Data with TxBxF format.
    """

    assert step_width == 1
    context_length = left_context + 1 + right_context
    assert X.shape[2] % context_length == 0
    F = X.shape[2] // context_length

    if mode == 'center':
        return X[:, :, left_context * F:(left_context + 1) * F]
    else:
        NotImplementedError(
            'All other unstack methods are not yet implemented.'
        )


def add_context(data, left_context=0, right_context=0, step=1,
                cnn_features=False, deltas_as_channel=False,
                num_deltas=2, sequence_output=True):
    if cnn_features:
        data = tbf_to_tbchw(data, left_context, right_context, step,
                            pad_mode='constant',
                            pad_kwargs=dict(constant_values=(0,)))
        if deltas_as_channel:
            feature_size = data.shape[3] // (1 + num_deltas)
            data = np.concatenate(
                [data[:, :, :, i * feature_size:(i + 1) * feature_size, :]
                 for i in range(1 + num_deltas)], axis=2)
    else:
        data = stack_context(data, left_context=left_context,
                             right_context=right_context, step_width=step)
        if not sequence_output:
            data = np.concatenate(
                [data[:, i, ...].reshape((-1, data.shape[-1])) for
                 i in range(data.shape[1])], axis=0)
    return data
