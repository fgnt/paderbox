import seaborn as sns
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import subprocess
import platform
import os
from os import path
from nt.visualization.new_cm import cmaps
from nt.utils import mkdir_p


mpl_ge_150 = LooseVersion(mpl.__version__) >= '1.5.0'


class DollarFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        if len(self.locs) == 0:
            return ''
        else:
            if not type(x) is str:
                s = self.pprint_val(x)
                return '\$' + self.fix_minus(s) + '\$'
            else:
                return x


class LatexContextManager(object):
    """ Context manager used for plotting which exports and calls Inkscape.

    """
    def __init__(
            self,
            filename,
            generate=None,  # eps recomented (alternative pdf), because pdf is in Inkscape 0.91 r not working
            build_folder=None,
            figure_size=[8.0, 6.0],
            formatter=DollarFormatter,
            format_x=True,
            format_y=True,
            palette=cmaps['upb'].colors[1:],
            extra_rc=None
    ):
        assert filename.endswith('.svg')
        self.filename = filename
        self.formatter = formatter
        self.figure_size = figure_size
        self.format_x = format_x
        self.format_y = format_y
        self.palette = palette
        self.generate = generate
        self.build_folder = build_folder if build_folder is not None else path.dirname(filename)
        if extra_rc is None:
            self.extra_rc = dict()
        else:
            self.extra_rc = extra_rc

    def __enter__(self):
        extra_rc = {
            'svg.fonttype': 'none',
            'text.usetex': False,
            'text.latex.unicode': False,
            'axes.unicode_minus': False
        }
        extra_rc.update(self.extra_rc)
        return context_manager(
            font_scale=2.5,
            extra_rc=extra_rc,
            figure_size=self.figure_size,
            palette=self.palette,
        )

    def __exit__(self, type, value, tb):
        figure = plt.gcf()
        for ax in figure.get_axes():
            if self.format_x:
                ax.xaxis.set_major_formatter(self.formatter())
            if self.format_y:
                ax.yaxis.set_major_formatter(self.formatter())
        if self.filename is not None:
            try:
                plt.savefig(self.filename)

                if platform.system() == 'Darwin':  # OS X
                    inkscape_path = ('/Applications/Inkscape.app/' +
                                     'Contents/Resources/bin/inkscape')
                else:
                    inkscape_path = 'inkscape'

                if self.generate:
                    try:
                        # inkscape --help
                        # -z, --without-gui  Do not use X server (only process
                        #                    files from console)

                        # inkscape -z --export-area-page fig.svg --export-eps=fig.eps --export-latex
                        build_file = os.path.splitext(path.join(self.build_folder, path.basename(self.filename)))[0]\
                                     + '.' + self.generate

                        cmd = [
                            inkscape_path, '-z', '--export-area-page',  # '--export-area-drawing',
                            os.path.realpath(self.filename),
                            '--export-{}={}'.format(
                                self.generate,
                                build_file),
                            '--export-latex'
                        ]
                        subprocess.run(cmd)
                    except:
                        print('Could not perform Inkscape export: {}.'.format(' '.join(cmd)))
            except FileNotFoundError:
                if not path.exists(path.dirname(self.filename)):
                    print('The folder {} does not exist.'.format(path.realpath(path.dirname(self.filename))))

                print('Could not save file {}.'.format(self.filename))
            except:
                print('Could not save file {}.'.format(self.filename))


def context_manager(
    seaborn_axes_style='whitegrid',
    seaborn_plotting_context='notebook',
    font_scale=1.0,
    line_width=3,
    figure_size=(8.0, 6.0),
    palette='muted',
    extra_rc=None,
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

    colors = sns.palettes.color_palette(sns.color_palette(palette))

    if mpl_ge_150:
        from cycler import cycler
        mul = len(colors)
        colors = 4*colors
        cyl = cycler('color', colors) + cycler(
            'linestyle', [*mul*['-'], *mul*['--'], *mul*[':'], *mul*['-.']])
        rc_parameters.update({
            'axes.prop_cycle': cyl
        })
    else:
        rc_parameters.update({
            'axes.color_cycle': list(colors)
        })

    rc_parameters.update({
        'patch.facecolor': colors[0]
    })

    plotting_context = sns.plotting_context(
        seaborn_plotting_context,
        font_scale=font_scale,
    )

    if extra_rc is None:
        extra_rc = dict()
    rc_parameters.update(extra_rc)

    final = dict(axes_style, **plotting_context)
    final.update(rc_parameters)

    return plt.rc_context(final)
