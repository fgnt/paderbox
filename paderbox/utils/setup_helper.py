from itertools import product
from copy import deepcopy
from pprint import pprint
from operator import itemgetter


def generate_setups(base_setup, update_setup=None):
    """ Helps to generate sweeps of setups.

    >>> setup = {
    ...     'length': 1,
    ...     'width': [10, 20]
    ... }
    >>> setups = generate_setups(setup)
    >>> pprint(setups)
    [{'length': 1, 'width': 10}, {'length': 1, 'width': 20}]

    >>> setups = generate_setups(setup, {'length': [1, 2]})
    >>> pprint(setups)
    [{'length': 1, 'width': 10},
     {'length': 1, 'width': 20},
     {'length': 2, 'width': 10},
     {'length': 2, 'width': 20}]

    >>> setups = generate_setups(setup, {'width': [11, 22]})
    >>> pprint(setups)
    [{'length': 1, 'width': 11}, {'length': 1, 'width': 22}]

    :param base_setup: Setup dict, where each entry which is a list will be
        be sweeped. Other iterables will not be sweeped, i.e. a tuple remains.
    :param update_setup: Optional update dict to the base setup.
    :return: List of setup dicts.
    """
    def _make_each_value_a_list(dictionary):
        return {key: value if isinstance(value, list) else [value]
                for key, value in dictionary.items()}

    def _get_list_of_setup_combinations(setup):
        dict_of_lists = _make_each_value_a_list(setup)
        list_of_items = list(dict_of_lists.items())
        list_of_items.sort(key=itemgetter(0))
        keys, values = list(zip(*list_of_items))
        result = product(*values)
        list_of_dicts = [dict(zip(keys, r)) for r in result]
        return list_of_dicts

    update_setup = {} if update_setup is None else update_setup
    setup = deepcopy(base_setup)
    setup.update(update_setup)
    return _get_list_of_setup_combinations(setup)
