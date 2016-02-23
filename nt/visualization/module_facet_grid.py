import seaborn as sns
import itertools
import numpy as np
import matplotlib.pyplot as plt


def facet_grid(data_list, function_list, kwargs_list=(), colwrap=2, scale=1,
               title_list=()):
    assert len(function_list) == len(data_list) or \
           len(function_list) == 1 or \
           len(data_list) == 1
    assert len(kwargs_list) == 0 or \
           len(kwargs_list) == 1 or \
           len(kwargs_list) == len(data_list) or \
           len(kwargs_list) == len(function_list)
    assert len(title_list) == len(data_list) or \
           len(title_list) == 0

    number_of_plots = max(len(function_list), len(data_list))
    number_of_rows = int(np.ceil(number_of_plots / colwrap))

    if len(kwargs_list) == 0:
        kwargs_list = len(data_list) * [{}]
    if len(kwargs_list) == 1:
        kwargs_list = len(data_list) * kwargs_list
    if len(kwargs_list) == 1:
        kwargs_list = len(function_list) * kwargs_list

    figure, axis = plt.subplots(number_of_rows, colwrap)
    figure.set_figwidth(figure.get_figwidth() * scale * colwrap)
    figure.set_figheight(figure.get_figheight() * scale * number_of_rows)

    if len(data_list) == 1:
        data_list = list(itertools.repeat(data_list[0], len(function_list)))
    if len(function_list) == 1:
        function_list = list(
            itertools.repeat(function_list[0], len(data_list)))

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
    return figure
