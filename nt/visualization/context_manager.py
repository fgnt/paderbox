import seaborn as sns
import matplotlib.pyplot as plt


def create_context_manager(
    seaborn_axes_style='whitegrid',
    seaborn_plotting_context='notebook',
    font_scale=1.5,
    line_width=2,
    figure_size=[8.0, 6.0],
    extra_rc={},
):
    """ Helper to create a plotting style with auto completion.

    :param seaborn_axes_style:
        One of the seaborn axes style presets. You can choose from
        `darkgrid``, ``whitegrid``, ``dark``, ``white``, and ``ticks``
        and find further details in the
        `seaborn tutorial <http://stanford.edu/~mwaskom/software/seaborn-dev/tutorial/aesthetics.html#styling-figures-with-axes-style-and-set-style>`_.
    :param seaborn_plotting_context:
        One of the seaborn plotting contexts. You can choose from
        ``notebook``, ``paper``, ``talk``, and ``poster``.
        Further details are here in the
        `seaborn tutorial <http://stanford.edu/~mwaskom/software/seaborn-dev/generated/seaborn.plotting_context.html#seaborn-plotting-context>`_.
    :param font_scale:
        The font scale scales all fonts at the same time relative
        to the seaborn plotting context default.
        A factor of ``2.0`` is already quite large.
    :param line_width:
        Line width in line plots.
    :param figure_size:
        Figure size in inches: width, height as a tuple
    :param extra_rc:
        Any other matplotlib rc parameter.
        You can get a list of all rc parameters by calling
        ``plt.rcParamsDefault``.
        If you frequently change a specific parameter, you are encouraged to
        add this parameter to the list of named parameters just as has been
        done with ``figure_size``.
    :return: A context manager to be used around your plots.
    """
    axes_style = sns.axes_style(seaborn_axes_style)

    rc_parameters = {
        'lines.linewidth': line_width,
        'figure.figsize': figure_size
    }
    rc_parameters.update(extra_rc)

    plotting_context = sns.plotting_context(
        seaborn_plotting_context,
        font_scale=font_scale,
    )

    final = dict(axes_style, **plotting_context)
    final.update(rc_parameters)

    return plt.rc_context(final)


default_context_manager = create_context_manager
