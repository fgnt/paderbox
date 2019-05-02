"""
Run this either with
$ python -m paderbox.utils.strip_solution name_template.ipynb name_solution.ipynb
or directly with
$ paderbox.strip_solution name_template.ipynb name_solution.ipynb
"""
import re
from pathlib import Path

import fire
import nbformat


CODE_MAGIC_WORD = '# REPLACE'
LATEX_MAGIC_WORD = '% REPLACE'


def code_replace(source, cell_type='code'):
    """

    Args:
        source: solutio code
        cell_type: 'code', 'markdown' or 'raw'

    Returns:
        template code

    >>> code_replace('return 1. / (1 + np.exp(-x))  # REPLACE return ???')
    'return ???'
    >>> print(code_replace(
    ... '''b = 1
    ... a = 1. / (1 + np.exp(-x))  # REPLACE
    ... return 1. / (1 + np.exp(-x))  # REPLACE return ???'''
    ... ))
    b=1
    return ???
    >>> print(code_replace(
    ... '''b = 1
    ... a = 1. / (1 + np.exp(-x))  # REPLACE a =
    ... return 1. / (1 + np.exp(-x))  # REPLACE return ???'''
    ... ))
    b = 1
    a =
    return ???

    """
    if cell_type in ['code', 'markdown']:
        MAGIC_WORD = {
            'code': CODE_MAGIC_WORD,
            'markdown': LATEX_MAGIC_WORD,
        }[cell_type]

        if MAGIC_WORD in source:
            new_source_lines = []
            for line in source.split('\n'):
                if MAGIC_WORD in line:
                    solution, template = line.split(MAGIC_WORD)
                    # Remove leading whitespaces
                    template = template.lstrip(' ')
                    if template == '':
                        continue
                    whitespace = re.search('( *)', line).group(1)
                    new_source_lines.append(whitespace + template.lstrip(' '))
                else:
                    new_source_lines.append(line)
            source = '\n'.join(new_source_lines)
    elif cell_type in ['raw']:
        pass
    else:
        raise TypeError(cell_type, source)
    return source


def nb_replace(old_path, new_path):
    """Remove the solution from a Jupyter notebook.

    For example, assume the following line is in a notebook:
        x = 10  # REPLACE x = ???

    The result will then be:
        x = ???

    The following example
        y = 42  # REPLACE y = # TODO
        q = 328 # REPLACE
        z = y * q # REPLACE
    will result in (Without an replacement the line will be deleted)
        y = # TODO

    And the example
        foo = bar  # COMMENT
    will result in
        # foo = bar
    """
    old_path = Path(old_path)
    new_path = Path(new_path)

    assert old_path != new_path, (old_path, new_path)
    assert old_path.is_file(), f'{old_path} is not a file.'
    assert old_path.name.endswith('_solution.ipynb'), old_path
    assert new_path.name.endswith('_template.ipynb'), new_path
    assert not new_path.is_file(), f'{new_path} already exists.'

    nb = nbformat.read(str(old_path), nbformat.NO_CONVERT)

    replacements = 0
    for i, cell in enumerate(nb['cells']):
        cell_source = code_replace(cell['source'], cell['cell_type'])
        if cell_source != cell['source']:
            replacements += 1
            cell['source'] = cell_source

    print(f'Replaced {replacements} lines of code.')
    nbformat.write(nb, new_path, nbformat.NO_CONVERT)


def entry_point():
    fire.Fire(nb_replace)


if __name__ == '__main__':
    fire.Fire(nb_replace)
