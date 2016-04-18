import numpy as np


def normalize_angle(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> normalize_angle(-np.pi)
    -3.141592653589793
    >>> normalize_angle(np.pi)
    -3.141592653589793

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return normalize_with_modulo(angle)


def normalize_with_modulo(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> normalize_with_modulo(-np.pi)
    -3.141592653589793
    >>> normalize_with_modulo(np.pi)
    -3.141592653589793

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_with_angle_exp(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> normalize_with_angle_exp(-np.pi)
    -3.1415926535897931
    >>> normalize_with_angle_exp(np.pi)
    3.1415926535897931

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return np.angle(np.exp(1j * angle))


def normalize_with_arctan2(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> normalize_with_arctan2(-np.pi)
    -3.1415926535897931
    >>> normalize_with_arctan2(np.pi)
    3.1415926535897931

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def normalize_with_arctan_tan(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> normalize_with_arctan_tan(-np.pi)
    -3.1415926535897931
    >>> normalize_with_arctan_tan(np.pi)
    3.1415926535897931

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return 2 * np.arctan(np.tan(angle/2))
