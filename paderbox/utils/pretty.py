import sys
import IPython.lib.pretty
import numpy as np


def pprint(obj, verbose=False, max_width=79, newline='\n',
           max_seq_length=IPython.lib.pretty.MAX_SEQ_LENGTH,
           max_array_length=50,
           np_suppress_small=True,
           ):
    """
    Copy of IPython.lib.pretty.pprint.
    Modifies the __repr__ of np.ndarray and torch.Tensor compared to the
    original.

    >>> pprint([np.array([1]), np.array([1]*100)])
    [array([1]), array(shape=(100,), dtype=int64)]
    >>> print([np.array([1])])
    [array([1])]

    >>> import torch
    >>> pprint([torch.tensor([1]), torch.tensor([1]*100)])
    [tensor([1]), tensor(shape=(100,), dtype=int64)]
    >>> print([torch.tensor([1])])
    [tensor([1])]
    """

    class MyRepresentationPrinter(IPython.lib.pretty.RepresentationPrinter):
        def __init__(self, *args, **kwargs):
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

    printer = MyRepresentationPrinter(sys.stdout, verbose, max_width, newline,
                                      max_seq_length=max_seq_length)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()
