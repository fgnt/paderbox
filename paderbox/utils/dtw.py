from numpy import array, zeros, argmin, inf, arange


def dtw(x, y, dist, dist_to_cost=None, border=(inf, inf), penalty=(0, 0, 0),
        weight=(1, 1, 1)):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param x: N1*M array or N1 element list
    :param y: N2*M array or N2 element list
    :param dist: distance function to calculate between elements of x and y
    :param dist_to_cost: transformation from distance to cost matrix.
                         needs to be an in place operation
    :param border: cost to add for steps along the border: (repeat x, repeat y)
    :param penalty: penalty for steps: (repeat x, match, repeat y)
    :param cost_scaling: scaling factor for costs when taking step:
                         (repeat x, match, repeat y)
                        'for usual dtw set (0, 0, 0)
                        'for levensthein': add distances only at diagonal
                                        step. always add penalty
                                        --> set (0, 1, 0)
    :return: minimum cost, distance matrix, accumulated cost matrix, wrap path
    """

    # init (boundary condition: start at (0,0) --> set boundaries to border)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = arange(1, c + 1) * border[0]
    D0[1:, 0] = arange(1, r + 1) * border[1]

    # calculate distance matrix
    D1 = D0[1:, 1:] # operate on view for easy indexing
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])

    # copy for output, since we are going to overwrite D0 and D1
    C = D1.copy()

    # normalize distance matrix
    if dist_to_cost is not None:
        dist_to_cost(D1)

    # calculate accumulative distance matrix
    # note: we again operate on view
    # --> use D0 instead of i and/or j minus 1
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(weight[1] * D0[i, j] + penalty[1],
                            weight[2] * D0[i, j+1] + penalty[2],
                            weight[0] * D0[i+1, j] + penalty[0])

    # traceback
    path = _traceback(D0, penalty)

    return D1[-1, -1], C, D1, path


def _traceback(D, penalty=(0, 0, 0)):
    """
    compute traceback through distance matrix starting from the end of both
    sequences (not partial matching)
    :param D: distance matrix (including boundary conditions)
    :param penalty: penalty for steps: (repeat x, match, repeat y)
    :return: path pairs
    """

    # starting index i, j = len(x) - 1, len(y) - 1
    # (calculated on distance matrix with boundary condition, therefore -2)
    # this also allows for easy checking for (i,j), (i,j+1), (i+1,j) in D
    # is equivalent to checking for (i-1,j-1), (i-1,j), (i,j-1) in D1
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j] + penalty[1],
                     D[i, j+1] + penalty[2],
                     D[i+1, j] + penalty[0]))
        if tb == 0 and i > 0 and j > 0:
            i -= 1
            j -= 1
        elif tb == 1 and i > 0:
            i -= 1
        elif tb == 2 and j > 0: # (tb == 2):
            j -= 1
        else:
            break

        # insert index in front of list, since we are going backwards
        p.insert(0, i)
        q.insert(0, j)

    # consume remaining indices in input sequences, if boundary was crossed
    # (insertions/deletions at befinning)
    while (i >= -1 and j > 0) or (i > 0 and j >= -1):
        if tb == 0:
            i -= 1
            j -= 1
            if j == -1:
                tb = 1
            else:
                tb = 2
        elif tb == 1:
            i -= 1
        else:
            j -= 1

        # insert index in front of list, since we are going backwards
        p.insert(0, i)
        q.insert(0, j)

    return array(p), array(q)
