from numpy import array, zeros, argmin, inf

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param x: N1*M array or N1 element list
    :param y: N2*M array or N2 element list
    :param dist: distance function to calculate distance
    :return: minimum distance, cost matrix, accumulated cost matrix, wrap path
    """

    # init
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf

    # calculate distance matrix
    D1 = D0[1:, 1:] # operate on view for easy indexing
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])

    # calculate accumulative distance matrix
    C = D1.copy() # copy for output, since we are going to overwrite D and D1
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])

    # traceback
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)

    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    """
    compute traceback through distance matrix
    :param D: distance matrix
    :return: path pairs
    """
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)

    return array(p), array(q)
