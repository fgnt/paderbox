class Container:
    """ A quick hack for a dictionary which is accessible with dot notation
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        self.__dict__.__delitem__(key)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def as_dict(self):
        return self.__dict__

    def __repr__(self, *args, **kwargs):
        return self.as_dict().__repr__(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        return self.as_dict().__str__(*args, **kwargs)
