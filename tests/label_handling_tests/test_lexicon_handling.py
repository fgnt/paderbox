import unittest

import paderbox.testing as tc
from paderbox.label_handling.lexicon import get_lexicon_from_word_list, Lexicon


class TestLexicon(unittest.TestCase):
    def test_get(self):
        lex = get_lexicon_from_word_list(['abc'])
        tc.assert_equal(lex['abc'], ['a', 'b', 'c'])
        tc.assert_equal(lex.words(), ['abc'])
        tc.assert_equal(lex.tokens(), ['a', 'b', 'c'])

    def test_list(self):
        lex = Lexicon(["a", "bc", "d"])
        expected = {"a": ["a", ], "bc": ["bc", ], "d": ["d", ]}
        tc.assert_equal(lex, expected)

    def test_rm_variants(self):
        lex = Lexicon({"a": [(["a", ], 1.), (["a", "a"], 0.5)]})
        expected = {"a": ["a", ]}
        tc.assert_equal(lex.no_variants(), expected)

    def test_disambiguate(self):
        lex = Lexicon({"a": ["a", ],
                       "ab": ["a", "b", ],
                       "ab!": ["a", "b", ]})
        n = lex.disambiguate()
        tc.assert_equal(n, 2)
        tc.assert_equal(lex['a'], ["a", "#1"])

    def test_clean(self):
        lex = Lexicon({"a": ["a", ],
                       "ab": ["a", "b", ],
                       "ab!": ["a", "b", ]})
        lex_cleaned = lex.cleaned(["a"])
        tc.assert_equal(lex_cleaned, Lexicon({"a": ["a", ]}))
