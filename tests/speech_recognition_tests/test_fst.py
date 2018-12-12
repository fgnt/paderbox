import tempfile
import unittest

from paderbox.TODO.kaldi.fst import build_monophone_fst, draw


class TestFST(unittest.TestCase):
    def test_build_monophone_fst(self):

        monophones = { 1: { -1: {'transitions': ((0, 0.5),)},
                             0: {'transitions': ((0, 0.5), (1, 0.5)),
                                 'observations': ((1, 0.5), (2, 0.5))},
                             1: {'transitions': ((1, 0.5), (2, 0.5)),
                                 'observations': ((3, 0.5), (4, 0.5))},
                             2: {'transitions': ((2, 0.5), (-1, 0.5)),
                                 'observations': ((5, 0.5), (6, 0.5))}},
                       2: { -1: {'transitions': ((0, 0.5),)},
                             0: {'transitions': ((0, 0.5), (1, 0.5)),
                                 'observations': ((7, 0.5), (8, 0.5))},
                             1: {'transitions': ((1, 0.5), (2, 0.5)),
                                 'observations': ((9, 0.5), (10, 0.5))},
                             2: {'transitions': ((2, 0.5), (-1, 0.5)),
                                 'observations': ((11, 0.5), (12, 0.5))}}}

        monophone_fst = build_monophone_fst(monophones, 0)
        with tempfile.TemporaryDirectory() as working_dir:
            fst_file = working_dir + '/monophone.fst'
            pdf_file = working_dir + '/monophone.pdf'
            monophone_fst.write_fst(fst_file)
            draw(None, None, fst_file, pdf_file)