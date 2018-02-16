import json
from nt.io.json_module import dump_json, Encoder

DELIMITER = '/'
ALLOWED_TYPES = (list, tuple, int, float, str, type(None))


class Options:
    _init_done = False

    def __init__(self, **kwargs):
        self._params = dict()
        self.add_nested_params(kwargs)
        self._init_done = True

    def to_nested_dict(self):
        """Converts structure to a nested dict."""
        nested_dict = dict()
        for k, v in self._params.items():
            if isinstance(v, type(self)):
                nested_dict[k] = v.to_nested_dict()
            else:
                nested_dict[k] = v
        return nested_dict

    def add_nested_params(self, nested_dict):
        """Adds nested parameters from a dict."""
        for k, v in nested_dict.items():
            self.add_param(k, v)
        return self

    def find_param(self, name):
        def _find_recursive(path, d):
            found = None
            for k in d:
                if isinstance(d[k], dict):
                    found = _find_recursive(path + [k], d[k])
                    if found:
                        return found
                else:
                    if k == name:
                        return DELIMITER.join(path + [name])
            return found
        return _find_recursive([], self.to_nested_dict())

    def _check_allowed_type(self, name, value, allow_dict=False):
        _all_allowed = ALLOWED_TYPES + (Options,)
        if allow_dict:
            _all_allowed = _all_allowed + (dict,)
        if isinstance(value, _all_allowed):
            return True
        else:
            _type = type(value)
            raise ValueError(
                f'Parameter {name} has an unknown type. Allowed types are '
                f'{_all_allowed}. Type of {name} is {_type}'
            )

    def add_param(self, name, value):
        """Adds a parameter."""
        if isinstance(value, dict):
            self._params[name] = Options(**value)
        elif self._check_allowed_type(name, value):
            self._params[name] = value

    @staticmethod
    def traverse_nested(name, _dict):
        splitted = name.split(DELIMITER)
        path = splitted[:-1]
        key = splitted[-1]
        root = 'root'
        for level_idx in range(len(path)):
            if not path[level_idx] in _dict:
                raise KeyError(
                    f"{path[level_idx]} not in {root}. "
                    f"Possible are: {_dict.keys()}."
                )
            else:
                root = path[level_idx]
                _dict = _dict[path[level_idx]]._params
        return root, _dict, key

    def update_param(self, name, value, allow_add=False):
        """Updates a single parameter value."""
        root, _dict, key = self.traverse_nested(name, self._params)
        def _update(k, v):
            if isinstance(v, dict):
                _dict[key] = Options(**v)
            else:
                _dict[key] = v
        if key in _dict or allow_add or 'kwargs' in name:
            if 'kwargs' in name:
                assert self._check_allowed_type(name, value, True)
            else:
                assert self._check_allowed_type(name, value)
            _update(key, value)
        else:
            raise KeyError(
                f"{key} not in {root} and adding a value was not allowed"
            )

    def update_params(self, nested_dict, allow_add=False):
        """Updates parameter values with a nested dict."""
        def _update_recursive(d, path):
            for k, v in d.items():
                if isinstance(v, dict):
                    new_path = [*path, k]
                    _update_recursive(v, new_path)
                else:
                    name = DELIMITER.join([*path, k])
                    self.update_param(name, v, allow_add)
        _update_recursive(nested_dict, [])

    def to_json(self, indent=2, separators=None, sort_keys=True):
        """Serializes the parameters into json."""
        return json.dumps(
            self.to_nested_dict(),
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
            cls=Encoder
        )

    def to_json_file(
            self, json_path, indent=2, separators=None, sort_keys=True
    ):
        """Serializes the hyperparameters to a json file."""
        dump_json(
            self.to_nested_dict(), json_path, indent=indent,
            separators=separators, sort_keys=sort_keys
        )

    @staticmethod
    def from_json_file(json_path):
        """Creates new HParams from json file."""
        with open(json_path) as fid:
            json_values = json.load(fid)
        return Options(**json_values)

    @staticmethod
    def from_json(json_values):
        """Creates new HParams from json string."""
        json_values = json.loads(json_values)
        return Options(**json_values)

    def __repr__(self):
        return self.to_json()

    def __getattr__(self, name):
        if not name == '_init_done':
            return self[name]

    def __getitem__(self, item):
        root, _dict, key = self.traverse_nested(item, self._params)
        if not key in _dict:
            msg = f'{item} is not a valid parameter. ' \
                f'Possible are {list(self._params.keys())}'
            raise KeyError(msg)
        else:
            return _dict[key]

    def __setattr__(self, name, value):
        # We need to check if the name is _params because this need to be
        # created before it can be accessed
        if self._init_done:
            if not name in self._params:
                raise ValueError(
                    'Cannot update parameter because it does not exists. '
                    'Please use "add_param" to add a parameter. '
                    f'Available parameters are {self._params}'
                )
            else:
                self.update_param(name, value)
        else:
            super().__setattr__(name, value)

    def __len__(self):
        return len(self._params)
