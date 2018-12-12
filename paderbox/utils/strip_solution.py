"""
Run this either with
$ python -m paderbox.utils.strip_solution name_template.ipynb name_solution.ipynb
or directly with
$ paderbox.strip_solution name_template.ipynb name_solution.ipynb
"""
from pathlib import Path
import re
import fire

MAGIC_WORD = '# REPLACE '


def replace(old_path, new_path):
    """Helps to remove the solution from a Jupyter notebook.

    For example, assume the following line is in a notebook:
        x = 10  # REPLACE x = ???

    The result will then be:
        x = ???
    """
    old_path = Path(old_path)
    new_path = Path(new_path)

    assert old_path.name.endswith('_solution.ipynb'), old_path
    assert new_path.name.endswith('_template.ipynb'), new_path
    assert old_path.is_file(), f'{old_path} is not a file.'
    assert not new_path.is_file(), f'{new_path} already exists.'

    replacements = 0

    with old_path.open() as old, new_path.open('w') as new:
        for line in old:
            if MAGIC_WORD in line:
                solution, template = line.split(MAGIC_WORD)
                whitespace = re.search('(\ *"\ *)', line).group(1)
                new.write(whitespace + template)
                replacements += 1
            else:
                new.write(line)

    print(f'Replaced {replacements} lines of code.')


def entry_point():
    fire.Fire(replace)


if __name__ == '__main__':
    fire.Fire(replace)
