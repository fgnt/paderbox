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
    -3.141592653589793
    >>> wrap_with_angle_exp(np.pi)
    3.141592653589793

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return np.angle(np.exp(1j * angle))


def wrap_with_arctan2(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> wrap_with_arctan2(-np.pi)
    -3.141592653589793
    >>> wrap_with_arctan2(np.pi)
    3.141592653589793

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_with_arctan_tan(angle):
    """ Normalize angle to be in the range of [-np.pi, np.pi[.

    Beware! Every possible method treats the corner case -pi differently.

    >>> wrap_with_arctan_tan(-np.pi)
    -3.141592653589793
    >>> wrap_with_arctan_tan(np.pi)
    3.141592653589793

    :param angle: Angle as numpy array in radian
    :return: Angle in the range of
    """
    return 2 * np.arctan(np.tan(angle/2))


def minus(angle1, angle2):
    """ Calculate angular difference.

    >>> minus(0, np.pi)
    -3.141592653589793
    >>> minus(0, -np.pi)
    3.141592653589793

    :param angle1: Minuend
    :param angle2: Subtrahend
    :return: Difference of angles in the range [-np.pi, np.pi].
    """
    return minus_with_wrap(angle1, angle2)


def minus_with_wrap(angle1, angle2):
    """ Calculate angular difference.

    >>> minus_with_wrap(0, np.pi)
    -3.141592653589793
    >>> minus_with_wrap(0, -np.pi)
    3.141592653589793

    :param angle1: Minuend
    :param angle2: Subtrahend
    :return: Difference of angles in the range [-np.pi, np.pi].
    """
    return wrap(angle1 - angle2)


def minus_with_angle_exp(angle1, angle2):
    """ Calculate angular difference.

    >>> minus_with_angle_exp(0, np.pi)
    -3.141592653589793
    >>> minus_with_angle_exp(0, -np.pi)
    3.141592653589793

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


def deg_to_rad(a):
    return a/180*np.pi


def rad_to_deg(a):
    return a/np.pi*180


def direction_vector_to_angle(vector):
    """ Takes a 2D direction vector and creates a single direction angle.

    >>> direction_vector_to_angle(np.asarray([[0], [0]]))
    0.0

    >>> direction_vector_to_angle(np.asarray([[1], [1]]))
    0.7853981633974483

    >>> direction_vector_to_angle(np.asarray([[0], [1]]))
    1.5707963267948966
    """
    assert vector.shape == (2, 1)
    return np.arctan2(vector[1, 0], vector[0, 0])


# http://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
def cart2sph(x, y, z):
    """transforms cartesian to spherical coordinates"""
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


# http://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
def sph2cart(azimuth, elevation, r):
    """transforms spherical to cartesian coordinates"""
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
