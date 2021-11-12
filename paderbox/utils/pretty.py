import sys
import io
import IPython.lib.pretty
import numpy as np


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
