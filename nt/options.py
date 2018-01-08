import json

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

    def add_param(self, name, value):
        """Adds a parameter."""
        _all_allowed = ALLOWED_TYPES + (type(self), dict)
        if isinstance(value, dict):
            self._params[name] = Options(**value)
        elif isinstance(value, _all_allowed):
            self._params[name] = value
        else:
            _type = type(value)
            raise ValueError(
                f'Parameter {name} has an unknown type. Allowed types are '
                f'{_all_allowed}. Type of {name} is {_type}'
            )

    @staticmethod
    def traverse_nested(name, _dict):
        splitted = name.split(DELIMITER)
        path = splitted[:-1]
        key = splitted[-1]
        root = 'root'
        for level_idx in range(len(path)):
            if not path[level_idx] in _dict:
                raise KeyError(f"{path[level_idx]} not in {root}")
            else:
                root = path[level_idx]
                _dict = _dict[path[level_idx]]._params
        return root, _dict, key

    def update_param(self, name, value, allow_add=False):
        """Updates a single parameter value."""
        root, _dict, key = self.traverse_nested(name, self._params)
        if key in _dict or allow_add:
            _dict[key] = value
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
        return self

    def to_json(self, indent=2, separators=None, sort_keys=True):
        """Serializes the hyperparameters into json."""
        return json.dumps(
            self.to_nested_dict(),
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def to_json_file(
            self, json_path, indent=2, separators=None, sort_keys=True
    ):
        """Serializes the hyperparameters to a json file."""
        with open(json_path, 'w') as fid:
            json.dump(
                self.to_nested_dict(),
                fid,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
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
        if not item in self._params:
            msg = f'{item} is not a valid parameter. ' \
                f'Possible are {list(self._params.keys())}'
            raise KeyError(msg)
        else:
            return self._params[item]

    def __setattr__(self, name, value):
        # We need to check if the name is _params because this need to be
        # created before it can be accessed
        if self._init_done:
            if not name in self._params:
                raise ValueError(
                    'Cannot set parameter. Please use "update_param" to update '
                    'a parameter or "add_param" to add a parameter'
                )
            else:
                self.update_param(name, value)
        else:
            super().__setattr__(name, value)

    def __len__(self):
        return len(self._params)
