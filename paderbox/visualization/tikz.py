import pathlib
import re
import tempfile
import textwrap
from pathlib import Path

from bs4 import BeautifulSoup

import IPython.display
from IPython.display import SVG, IFrame, Code, display
from IPython.core.magic import register_cell_magic

import fire

from paderbox.utils.process_caller import run_process


class FloatNumberUnit:
    """
    >>> FloatNumberUnit('23') * 2
    46.0
    >>> FloatNumberUnit('23pt') * 2
    46.0pt
    """
    _unit = ''

    def __init__(self, value: str):
        if isinstance(value, str) and value.endswith('pt'):
            self._unit = 'pt'
            self._value = float(value[:-len(self._unit)])
        else:
            self._value = float(value)

    def __mul__(self, other):
        other = float(other)
        return FloatNumberUnit(
            self.value_unit_to_str(self._value * other, self._unit))

    @staticmethod
    def value_unit_to_str(value, unit):
        return f'{value}{unit}'

    def __repr__(self):
        return self.value_unit_to_str(self._value, self._unit)


def resize_svg_code(
        code,
        factor,
        debug=False,
        method='bs4',
):
    if method == 'bs4':
        code_bs = BeautifulSoup(code, "lxml")
        code_bs.svg["height"] = FloatNumberUnit(code_bs.svg["height"]) * factor
        code_bs.svg["width"] = FloatNumberUnit(code_bs.svg["width"]) * factor
        return str(code_bs.svg)

    elif method == 'regex':
        m = re.fullmatch('(.*svg width=")(\d*)(pt" height=")(\d*)(pt".*)',
                         code,
                         re.DOTALL)
        g = list(m.groups())

        assert len(g) == 5

        #     print(m)
        if debug:
            print(g[1], g[3])

        g[1] = str(int(g[1]) * factor)
        g[3] = str(int(g[3]) * factor)

        if debug:
            print(g[1], g[3])

        n = ''.join(g)
        return n
    else:
        raise ValueError(method)


class Tikz2svg:
    """
    See
        git@ntgit:python/toolbox/notebooks
        toolbox_examples/visualization/Creating_tikz_images.ipynb
    for an example.

    Call `Tikz2svg.register_cell_magic()` to register tikz magic.
    Mark a cell with `%%tikz2svg --as-tex=True` to use it.
    Inside the cell youcan write prue latex code.

    """

    @staticmethod
    def fill_template(code, tikzpicture_args):
        code = textwrap.indent(code, ' ' * 4)
        return (
            '\\documentclass[tikz,border=5]{standalone}\n'
            '\\usepackage[expproduct=times]{siunitx}\n'
            '\\begin{document}\n'
            '\\begin{tikzpicture}['
            f'{tikzpicture_args}'
            ']\n'
            f'{code}'
            '\\end{tikzpicture}\n'
            '\\end{document}\n'
        )

    @classmethod
    def register_cell_magic(cls):

        @register_cell_magic
        def tikz2svg(line, cell):
            image = None

            def main(
                    x='1em',
                    y='1em',
                    scale='1',
                    svg_scale=2,
                    as_tex=False,
                    raw=False,
            ):
                nonlocal image
                if as_tex:
                    image = IPython.display.Code(cls.tikz_to_tex(
                        code=cell,
                        x=x,
                        y=y,
                        scale=scale,
                        raw=raw,
                    ), language='latex')
                else:
                    svg = cls.tikz_to_plot(
                        code=cell,
                        x=x,
                        y=y,
                        scale=scale,
                        svg_scale=svg_scale,
                        raw=raw,
                    )
                    image = svg

            fire.Fire(main, line)

            return image

    @classmethod
    def tikz_to_tex(
            cls,
            code,
            x='1em',
            y='1em',
            scale='1',
            raw=False,
            **kwargs,
    ):
        kwargs['x'] = x
        kwargs['y'] = y
        kwargs['scale'] = scale
        tikzpicture_args = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
        if not raw:
            code = cls.fill_template(
                code=code,
                tikzpicture_args=tikzpicture_args,
            )
        return code

    @classmethod
    def tikz_to_plot(
            cls,
            code,
            x='1em',
            y='1em',
            scale='1',
            svg_scale=2,
            raw=False,
            **kwargs,
    ):
        code = cls.tikz_to_tex(code, x=x, y=y, scale=scale, raw=raw, **kwargs)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                main_tex = (tmpdir / 'main.tex')
                main_tex.write_text(code)

                run_process([
                    'pdflatex',
                    'main.tex'
                ], cwd=tmpdir)

                run_process([
                    'inkscape',
                    '--without-gui',
                    '--file=main.pdf',
                    '--export-plain-svg=main.svg',
                ], cwd=tmpdir)
                res_svg = Path(tmpdir / 'main.svg').read_text()
            return SVG(
                resize_svg_code(res_svg, factor=svg_scale, method='bs4'))
        except Exception as e:
            display(Code(code, language='tex'))
            raise


def plot_to_tikz_embedded_png(
        name, png_path=pathlib.Path('.'), tikz_path=pathlib.Path('.'),
        tex_image_prefix='images', width='\\plotwidth', xlabel='', ylabel='',
        colors='', additional_tikz_content='', dpi=300):
    """
    Creates a tikz file that has a png image as background that is created using
    the current matplotlib figure.

    Example:
        >>> from matplotlib import pyplot as plt
        >>> import numpy as np
        >>> data = np.random.randn(2, 100)
        >>> _ = plt.scatter(*data)
        >>> plot_to_tikz_embedded_png('test')

    Args:
        name: name of the figure (png and tikz file)
        png_path: path to place the png file
        tikz_path: path to place the tikz file
        tex_image_prefix: prefix used in the tikz file to reference the image
            folder
        width: width of the plot. Can be a string as e.g. `.5\columnwidth`
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        colors: Color definitions are placed at the beginning of the tikzpicture
        additional_tikz_content: optional content. Is inserted behind the
            embedded png and before '\end{axis}'.
        dpi: Dots per inch for the png export
    """
    from matplotlib import pyplot as plt

    # Make sure no axis is drawn and no whitespace appears
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # Create png file
    plt.savefig(str(png_path / f'{name}.png'), bbox_inches='tight',
                pad_inches=0, dpi=dpi)

    # Create tikz
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    with (tikz_path / f'{name}.tikz').open('w', encoding='utf-8') as f:
        f.write(f"""
    \\begin{{tikzpicture}}[trim axis left, trim axis right]
            {colors}
            \\begin{{axis}}[
            xmin={xmin},xmax={xmax},ymin={ymin},ymax={ymax},
            axis on top,
            width={width}, scale only axis,
            ylabel style={{at={{(axis description cs:0.075,.5) }} }},
            xlabel={xlabel},
            ylabel={ylabel}]
            \\addplot graphics[xmin={xmin},ymin={ymin},xmax={xmax},ymax={ymax}] {{{tex_image_prefix}/{name}.png}};
            {additional_tikz_content}
        \\end{{axis}}
    \\end{{tikzpicture}}
    """)
