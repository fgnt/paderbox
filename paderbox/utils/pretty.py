import dataclasses
import sys
import io
import IPython.lib.pretty
import numpy as np

if sys.version_info >= (3, 8):
    from IPython.lib.pretty import CallExpression
else:
    # CallExpression was added in ipython 8, which dropped support for python 3.7
    # The following is a copy of the CallExpression class from IPython 8.
    class CallExpression:
        """ Object which emits a line-wrapped call expression in the form `__name(*args, **kwargs)` """

        def __init__(__self, __name, *args, **kwargs):
            # dunders are to avoid clashes with kwargs, as python's name manging
            # will kick in.
            self = __self
            self.name = __name
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def factory(cls, name):
            def inner(*args, **kwargs):
                return cls(name, *args, **kwargs)

            return inner

        def _repr_pretty_(self, p, cycle):
            # dunders are to avoid clashes with kwargs, as python's name manging
            # will kick in.

            started = False

            def new_item():
                nonlocal started
                if started:
                    p.text(",")
                    p.breakable()
                started = True

            prefix = self.name + "("
            with p.group(len(prefix), prefix, ")"):
                for arg in self.args:
                    new_item()
                    p.pretty(arg)
                for arg_name, arg in self.kwargs.items():
                    new_item()
                    arg_prefix = arg_name + "="
                    with p.group(len(arg_prefix), arg_prefix):
                        p.pretty(arg)


class _MyRepresentationPrinter(IPython.lib.pretty.RepresentationPrinter):
    def __init__(
            self,
            *args,
            max_array_length,
            np_suppress_small,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        def _ipy_pprint_ndarray(obj, p, cycle):
            if obj.size <= max_array_length:
                # Use repr or str?
                # repr -> array([...])
                # str -> [...]
                p.text(np.array_repr(
                    obj, suppress_small=np_suppress_small))
            else:
                p.text(
                    # f'{obj.__class__.__name__}'  # ndarray
                    f'{obj.__class__.__name__.replace("nd", "")}'
                    f'(shape={obj.shape}, dtype={obj.dtype})'
                )

        def _ipy_pprint_tensor(obj, p, cycle):
            # Following the indroduction in
            # https://github.com/pytorch/pytorch/issues/15683
            # it is unlikely, that torch will introduce a suppress.
            if obj.numel() <= max_array_length:
                p.text(repr(obj))
            else:
                p.text(
                    # f'{obj.__class__.__name__}'  # Tensor
                    f'{obj.__class__.__name__.lower()}'
                    f'('
                    f'shape={tuple(obj.shape)}, '
                    f'dtype={str(obj.dtype).replace("torch.", "")}'
                    f')'
                )

        self.type_pprinters[np.ndarray] = _ipy_pprint_ndarray
        self.deferred_pprinters[('torch', 'Tensor')] = _ipy_pprint_tensor
        self.type_pprinters[type({}.keys())] = \
            IPython.lib.pretty._seq_pprinter_factory('dict_keys(', ')')
        self.type_pprinters[type({}.values())] = \
            IPython.lib.pretty._seq_pprinter_factory('dict_values(', ')')
        self.type_pprinters[type({}.items())] = \
            IPython.lib.pretty._seq_pprinter_factory('dict_items(', ')')

    @property
    def max_seq_length(self):
        if isinstance(self._max_seq_length, (tuple, list)):
            if self.depth >= len(self._max_seq_length):
                return self._max_seq_length[-1]
            else:
                return self._max_seq_length[self.depth]
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value):
        self._max_seq_length = value

    depth = -1

    def _enumerate(self, seq):
        """
        >>> nested = {l: [tuple(range(3))]*4 for l in 'abcd'}
        >>> pprint(nested)
        {'a': [(0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)],
         'b': [(0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)],
         'c': [(0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)],
         'd': [(0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)]}
        >>> pprint(nested, max_seq_length=[3, 2])
        {'a': [(0, 1, ...), (0, 1, ...), ...],
         'b': [(0, 1, ...), (0, 1, ...), ...],
         'c': [(0, 1, ...), (0, 1, ...), ...],
         ...}
        >>> pprint(nested, max_seq_length=[3, 2, 1])
        {'a': [(0, ...), (0, ...), ...],
         'b': [(0, ...), (0, ...), ...],
         'c': [(0, ...), (0, ...), ...],
         ...}
        >>> pprint(nested, max_seq_length=2)
        {'a': [(0, 1, ...), (0, 1, ...), ...],
         'b': [(0, 1, ...), (0, 1, ...), ...],
         ...}
        """
        self.depth += 1
        yield from super()._enumerate(seq)
        self.depth -= 1

    @staticmethod
    def _dataclass_repr_pretty_(self, p, cycle):
        """
        >>> @dataclasses.dataclass
        ... class PointClsWithALongName:
        ...     x: int
        ...     y: int
        >>> pprint(PointClsWithALongName(1, 2), max_width=len('PointClsWithALongName') - 5)
        PointClsWithALongName(x=1,
                              y=2)
        >>> pprint(PointClsWithALongName(1, 2), max_width=len('PointClsWithALongName') + 9)
        PointClsWithALongName(x=1,
                              y=2)
        >>> pprint(PointClsWithALongName(1, 2), max_width=len('PointClsWithALongName') + 10)
        PointClsWithALongName(x=1, y=2)

        >>> @dataclasses.dataclass
        ... class PrettyPoint:
        ...     x: int
        ...     y: int
        ...     def _repr_pretty_(self, p, cycle):
        ...         p.text(f'CustomRepr(x={self.x}, y={self.y})')
        >>> pprint(PrettyPoint(1, 2))
        CustomRepr(x=1, y=2)

        """
        p.pretty(CallExpression.factory(
            self.__class__.__name__
        )(
            **{k: getattr(self, k) for k in self.__dataclass_fields__.keys()}
        ))

    def _in_deferred_types(self, cls):
        if '_repr_pretty_' not in cls.__dict__ and dataclasses.is_dataclass(cls):
            return self._dataclass_repr_pretty_
        else:
            return super()._in_deferred_types(cls)


def pprint(
        obj,
        *objs,
        verbose=False,
        max_width=79,
        newline='\n',
        max_seq_length=IPython.lib.pretty.MAX_SEQ_LENGTH,
        max_array_length=50,
        np_suppress_small=True,
):
    """
    Copy of IPython.lib.pretty.pprint.
    Differences:
     - Shortens the __repr__ of large np.ndarray and torch.Tensor
     - Support multiple objects (Causes bad readable error in original)
     - Support list/tuple for max_seq_length, where
       max(depth, len(max_seq_length)-1) is used as index.


    >>> pprint([np.array([1.]), np.array([1.]*100)])
    [array([1.]), array(shape=(100,), dtype=float64)]
    >>> print([np.array([1])])
    [array([1])]

    >>> import torch
    >>> pprint([torch.tensor([1]), torch.tensor([1]*100)])
    [tensor([1]), tensor(shape=(100,), dtype=int64)]
    >>> print([torch.tensor([1])])
    [tensor([1])]

    >>> d = {'a'*10: 1_000_000, 'b'*10: 2_000_000}
    >>> pprint(d.keys(), max_width=30)
    dict_keys('aaaaaaaaaa',
              'bbbbbbbbbb')
    >>> pprint(d.values(), max_width=20)
    dict_values(1000000,
                2000000)
    >>> pprint(d.items(), max_width=30)
    dict_items(('aaaaaaaaaa',
                1000000),
               ('bbbbbbbbbb',
                2000000))
    >>> print(d.keys())
    dict_keys(['aaaaaaaaaa', 'bbbbbbbbbb'])
    >>> print(d.values())
    dict_values([1000000, 2000000])
    >>> print(d.items())
    dict_items([('aaaaaaaaaa', 1000000), ('bbbbbbbbbb', 2000000)])



    """

    printer = _MyRepresentationPrinter(
        sys.stdout, verbose, max_width, newline,
        max_seq_length=max_seq_length,
        max_array_length=max_array_length,
        np_suppress_small=np_suppress_small,
    )

    if len(objs):
        printer.pretty((obj, *objs))
    else:
        printer.pretty(obj)

    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()


def pretty(
        obj,
        *objs,
        verbose=False,
        max_width=79,
        newline='\n',
        max_seq_length=IPython.lib.pretty.MAX_SEQ_LENGTH,
        max_array_length=50,
        np_suppress_small=True,
):
    """
    Copy of IPython.lib.pretty.pretty.
    Differences:
     - Shortens the __repr__ of large np.ndarray and torch.Tensor
     - Support multiple objects (Causes bad readable error in original)

    Pretty print the object's representation.
    """
    stream = io.StringIO()
    printer = _MyRepresentationPrinter(
        stream, verbose, max_width, newline,
        max_seq_length=max_seq_length,
        max_array_length=max_array_length,
        np_suppress_small=np_suppress_small,
    )
    if len(objs):
        printer.pretty((obj, *objs))
    else:
        printer.pretty(obj)
    printer.flush()
    return stream.getvalue()


if __name__ == '__main__':
    import tempfile
    import contextlib
    import shutil
    import subprocess
    import shlex
    from pathlib import Path

    import paderbox as pb


    def cli_pprint(file, max_seq_length=[10, 5, 2], max_width=None, unsafe=False):
        """Load a file and pretty print it. With max_seq_length you can
        control the length of the printed sequences.

        Args:
            file: e.g. a json file
            max_seq_length: An integer or a list of integers.
                The last entry is used for all larger depths.
            max_width:
        """
        data = pb.io.load(file, unsafe=unsafe)

        if max_width is None:
            max_width = shutil.get_terminal_size((79, 20)).columns

        pprint(data, max_seq_length=max_seq_length, max_width=max_width)

    def cli_diff(file1, file2, max_seq_length=[10, 5, 2], max_width=None):
        """Load two files, prettify them and forward to icdiff.
        icdiff shows a colored side by side diff in the terminal.

        Args:
            file1: e.g. a json file
            file2: e.g. a json file
            max_seq_length: An integer or a list of integers.
                The last entry is used for all larger depths.
            max_width:
        """
        if max_width is None:
            max_width = shutil.get_terminal_size((79, 20)).columns // 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            json1 = Path(file1)
            json2 = Path(file2)
            # Add prefix (0_, 1_) to ensure different file names.
            f1 = tmp_dir / ('0_' + json1.name)
            f2 = tmp_dir / ('1_' + json2.name)

            with open(f1, 'w') as fd:
                with contextlib.redirect_stdout(fd):
                    cli_pprint(json1, max_seq_length=max_seq_length, max_width=max_width)
            with open(f2, 'w') as fd:
                with contextlib.redirect_stdout(fd):
                    cli_pprint(json2, max_seq_length=max_seq_length, max_width=max_width)

            subprocess.run(
                f'icdiff {shlex.quote(str(f1))} {shlex.quote(str(f2))}',
                shell=True)

    import fire
    fire.Fire({
        'pprint': cli_pprint,
        'pp': cli_pprint,
        'diff': cli_diff,
    })
