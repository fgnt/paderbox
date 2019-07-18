import tempfile
from pathlib import Path
from paderbox.utils.process_caller import run_process


class PDF(object):
    def __init__(self, pdf, backend='iframe', size=(1000, 400)):
        """

        :param pdf: path to pdf
        :param backend:  may either be svg or iframe,
                    where svg works better embedded, but may show a faulty
                    representation of the pdf
        :param size:
        """
        self.pdf = pdf
        self.size = size
        assert backend in ['svg', 'iframe']
        self.backend = backend

    def _repr_html_(self):
        if self.backend == 'svg':
            return self._repr_svg_()
        elif self.backend == 'iframe':
            return '<iframe src="{0}" width={1[0]} height={1[1]}></iframe>'.\
                format(self.pdf, self.size)
        else:
            raise ValueError(self.backend)

    def _repr_latex_(self):
        return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

    def _repr_svg_(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            run_process([
                'inkscape',
                '--without-gui',
                f'--file={Path(self.pdf).expanduser().resolve()}',
                '--export-plain-svg=main.svg',
            ], cwd=tmpdir)
            return Path(tmpdir / 'main.svg').read_text()