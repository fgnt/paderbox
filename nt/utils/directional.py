import numpy as np


def wrap(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> wrap(-np.pi)
    -3.141592653589793
    >>> wrap(np.pi)
    3.141592653589793

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return wrap_with_angle_exp(angle)


def wrap_with_modulo(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> wrap_with_modulo(-np.pi)
    -3.141592653589793
    >>> wrap_with_modulo(np.pi)
    -3.141592653589793

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_with_angle_exp(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> wrap_with_angle_exp(-np.pi)
    -3.1415926535897931
    >>> wrap_with_angle_exp(np.pi)
    3.1415926535897931

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return np.angle(np.exp(1j * angle))


def wrap_with_arctan2(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> wrap_with_arctan2(-np.pi)
    -3.1415926535897931
    >>> wrap_with_arctan2(np.pi)
    3.1415926535897931

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_with_arctan_tan(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> wrap_with_arctan_tan(-np.pi)
    -3.1415926535897931
    >>> wrap_with_arctan_tan(np.pi)
    3.1415926535897931

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return 2 * np.arctan(np.tan(angle/2))


def minus(angle1, angle2):
    """ Calculate angular difference.

    >>> minus(0, np.pi)
    -3.1415926535897931
    >>> minus(0, -np.pi)
    3.1415926535897931

    :param angle1: Minuend
    :param angle2: Subtrahend
    :return: Difference of angles in the range [-np.pi, np.pi].
    """
    return minus_with_wrap(angle1, angle2)


def minus_with_wrap(angle1, angle2):
    """ Calculate angular difference.

    >>> minus(0, np.pi)
    -3.1415926535897931
    >>> minus(0, -np.pi)
    3.1415926535897931

    :param angle1: Minuend
    :param angle2: Subtrahend
    :return: Difference of angles in the range [-np.pi, np.pi].
    """
    return wrap(angle1 - angle2)


def minus_with_angle_exp(angle1, angle2):
    """ Calculate angular difference.

    >>> minus(0, np.pi)
    -3.1415926535897931
    >>> minus(0, -np.pi)
    3.1415926535897931

    :param angle1: Minuend
    :param angle2: Subtrahend
    :return: Difference of angles in the range [-np.pi, np.pi].
    """
    return np.angle(np.exp(1j * angle1) / np.exp(1j * angle2))


def plus(angle1, angle2):
    raise NotImplementedError()


def mean(angle, axis=None):
    raise NotImplementedError()


def variance(angle, axis=None):
    raise NotImplementedError()
