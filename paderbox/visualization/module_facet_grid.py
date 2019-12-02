import itertools
import numpy as np
import matplotlib.pyplot as plt
import types


def facet_grid(
        data_list, function_list, kwargs_list=(), colwrap=2, scale=1,
        title_list=(), return_axis=False,
        sharex=False, sharey=False,
):

    if isinstance(function_list, types.FunctionType):
        function_list = [function_list]

    assert len(function_list) == len(data_list) or \
           len(function_list) == 1 or \
           len(data_list) == 1
    assert len(kwargs_list) == 0 or \
           len(kwargs_list) == 1 or \
           len(kwargs_list) == len(data_list) or \
           len(kwargs_list) == len(function_list) or \
           len(function_list) == len(data_list)
    assert len(title_list) == len(data_list) or \
           len(title_list) == 0

    number_of_plots = max(len(function_list), len(data_list), len(kwargs_list))
    number_of_rows = int(np.ceil(number_of_plots / colwrap))

    if len(kwargs_list) == 0:
        kwargs_list = len(data_list) * [{}]
    if len(kwargs_list) == 1:
        kwargs_list = len(data_list) * kwargs_list
    if len(kwargs_list) == 1:
        kwargs_list = len(function_list) * kwargs_list

    figure, axis = plt.subplots(
        number_of_rows, colwrap, sharex=sharex, sharey=sharey,
    )

    figure.set_figwidth(figure.get_figwidth() * scale * colwrap)
    figure.set_figheight(figure.get_figheight() * scale * number_of_rows)

    if len(data_list) == 1:
        data_list = list(itertools.repeat(data_list[0], number_of_plots))
    if len(function_list) == 1:
        function_list = list(
            itertools.repeat(function_list[0], number_of_plots))

    for index in range(len(data_list)):
        if data_list[index] is None:
            function_list[index](ax=axis.flatten()[index],
                                 **kwargs_list[index])
        else:
            function_list[index](data_list[index],
                                 ax=axis.flatten()[index],
                                 **kwargs_list[index])

    for index in range(number_of_plots, number_of_rows * colwrap):
        axis.flatten()[index].axis('off')

    if len(title_list) > 0:
        for idx, title in enumerate(title_list):
            axis.flatten()[idx].set_title(title)

    figure.tight_layout()
    if return_axis:
        return figure, axis
    else:
        return figure


def facet_grid_zero_space_time_frequency_plot(
        data_list, function_list, kwargs_list=(), colwrap=2, scale=1,
        title_list=(), return_axis=False, wspace=0.01, hspace=0.01):
    """
    This Function is a wrapper around facet_grid. It assumes,
    that function_list is a list of one plot function. This
    function has a parameter colorbar, like paderbox.visualisation.plot.mask.
    All inner xlabel, xticklabel, ylabel, yticklabel and title will be removed.
    The space between the figures will be set zu 0.01
    """

    number_of_plots = len(data_list)

    if len(kwargs_list) == 0:
        kwargs_list = [{}] * len(data_list)
    elif len(kwargs_list) == 1:
        kwargs_list *= number_of_plots

    kwargs_list = [{'colorbar': (i % colwrap == colwrap - 1), **kwargs}
                   for i, kwargs in zip(range(len(data_list)), kwargs_list)]

    figure, axis = facet_grid(
        data_list, function_list, title_list=title_list, scale=scale,
        kwargs_list=kwargs_list, colwrap=colwrap, return_axis=True)

    try:
        [ax.xaxis.set_ticklabels([]) for ax in axis[:-1, :].ravel()]
        [ax.set_xlabel('') for ax in axis[:-1, :].ravel()]
        [ax.yaxis.set_ticklabels([]) for ax in axis[:, 1:].ravel()]
        [ax.set_ylabel('') for ax in axis[:, 1:].ravel()]
        [ax.set_title('') for ax in axis[1:, :].ravel()]
    except IndexError:
        try:
            [ax.yaxis.set_ticklabels([]) for ax in axis[1:].ravel()]
            [ax.set_ylabel('') for ax in axis[1:].ravel()]
        except IndexError:
            pass

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    if return_axis:
        return figure, axis
    else:
        return figure
