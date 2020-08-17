import tempfile
from pathlib import Path

import numpy as np
import datetime

from paderbox.io.new_subdir import get_new_subdir, NameGenerator


def test_index():
    # Do all get_new_subdir tests in one TemporaryDirectory to ensure, that
    # they correctly read the files from the folder.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        new = get_new_subdir(tmp_dir)
        assert new.name == '1'
        
        new = get_new_subdir(tmp_dir)
        assert new.name == '2'

        new = get_new_subdir(tmp_dir, prefix='prefix')
        assert new.name == 'prefix_1'

        new = get_new_subdir(tmp_dir, prefix='prefix')
        assert new.name == 'prefix_2'

        new = get_new_subdir(tmp_dir, suffix='suffix')
        assert new.name == '1_suffix'

        new = get_new_subdir(tmp_dir, suffix='suffix')
        assert new.name == '2_suffix'

        new = get_new_subdir(tmp_dir, prefix='prefix', suffix='suffix')
        assert new.name == 'prefix_1_suffix'

        new = get_new_subdir(tmp_dir, prefix='prefix', suffix='suffix')
        assert new.name == 'prefix_2_suffix'

        new = get_new_subdir(tmp_dir, id_naming='time')
        # depends on the the current time -> no assert

        rng = np.random.RandomState(0)
        new = get_new_subdir(tmp_dir, id_naming=NameGenerator(rng=rng))
        assert new.name == 'helpful_tomato_finch'

        rng = np.random.RandomState(0)
        new = get_new_subdir(tmp_dir, id_naming=NameGenerator(rng=rng))
        assert new.name == 'colourful_apricot_piranha'

        rng = np.random.RandomState(0)
        new = get_new_subdir(tmp_dir, id_naming=NameGenerator(('adjectives', 'animals', range(10)), rng=rng))
        assert new.name == 'helpful_bonobo_5'

        rng = np.random.RandomState(0)
        new = get_new_subdir(tmp_dir, id_naming=NameGenerator(('adjectives', 'animals', range(10)), rng=rng))
        assert new.name == 'colourful_tern_3'

        assert NameGenerator(rng=rng).possibilities() == 28_406_196
        assert NameGenerator(('adjectives', 'animals', range(10))).possibilities() == 5_462_730

        # Example, how you get time stamp and word combination
        # Note, you shouldn't use here the rng argument from NameGenerator.
        def id_naming():
            time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            return time + NameGenerator()()

        new = get_new_subdir(tmp_dir, id_naming=id_naming)
