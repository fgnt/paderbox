"""
Run this either with
$ python -m paderbox.utils.strip_solution name_template.ipynb name_solution.ipynb
or directly with
$ paderbox.strip_solution name_template.ipynb name_solution.ipynb
"""
from pathlib import Path
import re
import fire

class FileExistsWarning(ResourceWarning):
    pass


REPLACE_PATTERN = re.compile(r'^(\s*)"(?:(\s*).*\s{0,2})?# (\d*)\s?REPLACE (.*(?:\\n)?",?)$')
COMMENT_PATTERN = re.compile(r'^(\s*)"(\s*)(.*)\s+# COMMENT\s*((?:\\n)?",?)$')
TRAILING_COMMA_PATTERN = re.compile(r',\\n(\s*)]')
TRAILING_COMMA_SUBST = r'\\n\1]'


def remove_trailing_commas(json_like):
    """Remove trailing commas from *json_like* and return the result.

    Source: https://gist.github.com/liftoff/ee7b81659673eca23cd9fc0d8b8e68b7
    Example::
        >>> remove_trailing_commas('{"foo":"bar","baz":["blah",],}')
        '{"foo":"bar","baz":["blah"]}'
    """
    trailing_object_commas_re = re.compile(
        r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    trailing_array_commas_re = re.compile(
        r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    # Fix objects {} first
    objects_fixed = trailing_object_commas_re.sub("}", json_like)
    # Now fix arrays/lists [] and return the result
    return trailing_array_commas_re.sub("]", objects_fixed)


def replace(old_path, new_path):
    """Remove the solution from a Jupyter notebook.

    For example, assume the following line is in a notebook:
        x = 10  # REPLACE x = ???

    The result will then be:
        x = ???

    The following example
        y = 42  # 3 REPLACE y = # TODO
        q = 328
        z = y * q
    will result in
        y = # TODO

    And the example
        foo = bar  # COMMENT
    will result in
        # foo = bar
    """
    old_path = Path(old_path)
    new_path = Path(new_path)

    assert old_path.name.endswith('_solution.ipynb'), old_path
    assert new_path.name.endswith('_template.ipynb'), new_path
    assert old_path.is_file(), f'{old_path} is not a file.'
    assert not new_path.is_file(), f'{new_path} already exists.'

    replacements = 0

    with old_path.open() as old, new_path.open('w') as new:
        replace_lines_count = 0
        for line in old:
            replace_match = REPLACE_PATTERN.match(line)
            comment_match = COMMENT_PATTERN.match(line)
            if replace_match is not None:
                outer_whitespace = replace_match.group(1)
                if replace_match.group(2) is not None:
                    inner_whitespace = replace_match.group(2)
                else:
                    inner_whitespace = ''
                # replace_lines_count: '# n REPLACE ...' lets the script remove
                # the n-1 following lines. Useful for multi-line solutions.
                if replace_match.group(3):
                    replace_lines_count = int(replace_match.group(3)) - 1
                else:
                    replace_lines_count = 0
                template_and_ending = replace_match.group(4)
                new.write(outer_whitespace + '"' + inner_whitespace
                          + template_and_ending + '\n')
                replacements += 1
            elif comment_match is not None:
                outer_whitespace = comment_match.group(1)
                inner_whitespace = comment_match.group(2)
                content = comment_match.group(3)
                ending = comment_match.group(4)
                new.write(outer_whitespace + '"' + inner_whitespace
                          + '# ' + content + ending + '\n')
            else:
                if replace_lines_count == 0:
                    new.write(line)
                else:
                    replace_lines_count -= 1
                    replacements += 1

    # remove obsolete trailing commas. This can happen if a REPLACE line
    # leads to replacing the last line of a cell.

    with new_path.open('r') as new:
        content = remove_trailing_commas(new.read())
    with new_path.open('w') as new:
        new.write(content)

    print(f'Replaced {replacements} lines of code.')


def entry_point():
    fire.Fire(replace)


if __name__ == '__main__':
    fire.Fire(replace)
