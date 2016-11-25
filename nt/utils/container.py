class AttrDict(dict):
    """

    A hack for a dictionary which is accessible with dot notation
    Source code: http://stackoverflow.com/a/14620633

    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
