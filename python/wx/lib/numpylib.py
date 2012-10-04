import operator
import numpy as np

class ConvergenceError(Exception):
    """Exception class for convergence errors"""

    def __init__(self, numiter):
        self.numiter = numiter

    def __str__(self):
        return "failed to converge after %d iterations" % (self.numiter)

class NanError(Exception):
    """Exception class for missing values"""
    def __init__(self, msg=''):
        self.msg = msg
    def __str__(self):
        return "failed due to missing values " + self.msg

class InfError(Exception):
    """Exception class for non-finite values"""
    def __init__(self, msg=''):
        self.msg = msg
    def __str__(self):
        return "failed due to non-finite values " + self.msg

def axenumerate(a, axis=0):
    """Iterator that yields (N-1)-dimensional slices along an axis of
    a N-dimensional array

    Parameters
    ---------
    a : ndarray
        Input array
    axis : int
        Axis of iteration.

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3,))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> g = axenumerate(x, axis=1)
    >>> g.next()
    (0, array([0, 3, 6]))
    >>> g.next()
    (1, array([1, 4, 7]))
    >>> g.next()
    (2, array([2, 5, 8]))
    """
    a = np.asarray(a)
    try:
        n = a.shape[axis]
    except IndexError:
        raise ValueError("axis(=%d) out of bounds" % (axis))
    s = [slice(None)] * a.ndim
    for i in xrange(n):
        s[axis] = i
        yield (i, a[s])

def atmost_1d(x):
    if np.ndim(x) <= 1:
        return x
    is_single_dim = (np.array(x.shape) == 1)
    if not np.sum(np.logical_not(is_single_dim)) == 1:
        msg = "array cannot be forced to single dim. shape: %s"
        raise ValueError(msg % str(x.shape))
    return x.flatten()

def remove_cols(x, cols):
    inds = range(x.shape[1])
    if not isinstance(cols, int):
        [inds.pop(i) for i in np.sort(cols)[::-1]]
    else:
        inds.pop(cols)
    return x[:, inds]

def remove_rows(x, rows):
    inds = range(x.shape[0])
    if not isinstance(rows, int):
        [inds.pop(i) for i in np.sort(rows)[::-1]]
    else:
        inds.pop(rows)
    return x[inds, :]

def nans(shape, dtype=float, order='C'):
    """Return a new array of given shape and type, filled with NaNs.

    This function is intended to emulate the behavior of numpy
    functions `zeros` and `ones`.

    Parameters
    ----------
    shape : int or sequent of int
        Shape of the new array

    See Also
    --------
    numpy.zeros, numpy.ones
    """
    x = np.zeros(shape, dtype, order)
    # python complains if you try to coerce a NaN to integer, but it
    # silently converts NaN to True if the type is boolean! We must
    # explicitly check whether the dtype of x can hold a NaN.
    if not np.isnan(x.dtype.type(np.nan)):
        raise ValueError("cannot convert float NaN to specified type")
    if np.iscomplexobj(x):
        x.fill(complex(np.nan, np.nan))
    else:
        x.fill(np.nan)
    return x

def flipdim(m, axis= -1):
    """Reverse the order of elements along the specified axis.

    Parameters
    ----------
    m : array_like
        Input array.
    axis : integer
        Axis along which elements are reversed. The default value is -1
        (last dimension of the array).

    See Also
    --------
    numpy.fliplr, numpy.flipud
    """
    m = np.asarray(m)
    return m.take(np.arange(m.shape[axis] - 1, -1, -1), axis)

def mdot(*args):
    """Generalization of numpy.dot to handle multiple inputs

    This function exhibits left-to-right associativity:

        mdot(a, b, c) = numpy.dot(numpy.dot(a, b), c)

    See Also
    --------
    numpy.dot
    """
    return reduce(np.dot, args)

def resize(a, new_shape, fillval=np.nan):
    if isinstance(new_shape, (int, np.integer)):
        new_shape = (new_shape,)
    a = np.ravel(a)
    Na = a.size
    if not Na:
        out = np.empty(new_shape, a.dtype.char)
        out.fill(fillval)
        return out

    total_size = np.multiply.reduce(new_shape)

    if total_size == 0:
        return a[:0]
    elif total_size > Na:
        extra = np.empty(total_size - Na)
        extra.fill(fillval)
        a = np.concatenate((a, extra))
    elif total_size < Na:
        a = a[:total_size]

    return np.reshape(a, new_shape)

def set_array_sd(arr, new_sd, axis):

    new_sd = np.asarray(new_sd)
    cur_sd = arr.std(axis=axis)
    cur_mean = arr.mean(axis=axis)

    if not cur_sd.shape == new_sd.shape:
        raise ValueError, "shape of new_sd does not match sd along axis"

    # add in the lost dimension and repeat the new axis
    shp = list(cur_mean.shape)
    shp.insert(axis, 1)
    cur_mean = np.repeat(cur_mean.reshape(shp), arr.shape[axis], axis=axis)
    cur_sd = np.repeat(cur_sd.reshape(shp), arr.shape[axis], axis=axis)
    new_sd = np.repeat(new_sd.reshape(shp), arr.shape[axis], axis=axis)

    arr = (arr - cur_mean) / cur_sd * new_sd + cur_mean
    return arr

def set_array_mean(arr, new_mean, axis):

    new_mean = np.asarray(new_mean)
    cur_mean = arr.mean(axis=axis)

    if cur_mean.shape != new_mean.shape:
        raise ValueError, "shape of new_mean does not match mean along axis"

    # add in the lost dimension and repeat the new axis
    shp = list(cur_mean.shape)
    shp.insert(axis, 1)
    cur_mean = np.repeat(cur_mean.reshape(shp), arr.shape[axis], axis=axis)
    new_mean = np.repeat(new_mean.reshape(shp), arr.shape[axis], axis=axis)

    arr = arr - cur_mean + new_mean
    return arr

def normalize(x, ord=None, scale=1.0):
    """Normalize a matrix or vector.

    Given a matrix or vector, this function returns an array of the
    same shape in which all elements are multiplied by the same scalar
    to achieve target norm within some floating-point error.

    Parameters
    ----------
    x : array_like, shape (M,) or (M, N)
        Input array
    ord : {non-zero int, numpy.inf, -numpy.inf, 'fro'}, optional
        Order of the norm (for description of possible values, see the
        documentation for numpy.linalg.norm). The default value is 2.
    scale : float, optional
        The target norm. The default value is 1.0

    Returns
    -------
    y : array
        Matrix or vector of the same shape.

    See Also
    --------
    numpy.linalg.norm
    """
    x = np.asarray(x)
    return x * scale / np.linalg.norm(x, ord)

def wmap1d(func, arr, window, mode='valid'):
    """Apply func over an array, to each window of elements
    TODO: I bet this would be a billion times faster using cython.
    Parameters
    ----------
    func : function
        the function to be applied
    arr : array_like
        input array.
    window : integer
        window over which the function is applied
    mode : {'valid', 'distinct'}, optional
        'valid':
          By default, mode is `valid` which applies the function to every
          (overlapping) window of elements
        'distinct':
          Apply the function to every complete non-overlapping window of elements

    """

    arr = np.asarray(arr)
    if np.ndim(arr) != 1 or window > len(arr):
        raise ValueError, "wmap only supports 1d arrays for now"

    if mode == 'valid':
        inds = [range(i, i + window) for i in np.arange(len(arr) - window + 1)]
    elif mode == 'distinct':
        if len(arr) % window:
            raise ValueError, "window does not divide axis length mode:%s" % mode
        inds = [range(i, i + window) for i in np.arange(0, len(arr), window)]
    else:
        raise ValueError, "unknown mode %s" % mode

    out_len = len(inds)
    out = np.empty(out_len) # minor optimization
    for i in range(out_len):
        out[i] = func(arr[inds[i]])

    return out

def first(x, no_occurrence_val=None):
    "takes a boolean array and returns index of the first True value"
    w = np.flatnonzero(x)
    if not len(w):
        return no_occurrence_val
    else:
        return(w[0])

def last(x, no_occurrence_val=None):
    "takes a boolean array and returns index of the last True value"
    w = np.flatnonzero(x)
    if not len(w):
        return no_occurrence_val
    else:
        return(w[-1])

def charisnan(x):
    """same as isnan but supports strings (no string is nan)
    """
    x = np.asarray(x)
    if x.dtype.type is np.string_ or x.dtype.type is np.object_:
        ret = np.empty(shape=x.shape, dtype=bool)
        ret.fill(False)
        return ret
    else:
        return np.isnan(x)

def eq_match_nans(x, y):
    """compares two objects.  if both are equal, returns True
    if both are nan, returns True
    if both are arrays (or lists, in python2.6) with matching nan values,
      returns True.
    if one or both arrays is a numpy ndarray, return return a new ndarray
      with entries 'True' for elements that match or are both nans.
    otherwise, returns False."""
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        out = (x == y)
        if out is bool: # e.g., dimensions don't match
            return out
        nan_flag = np.logical_and(np.isnan(x), np.isnan(y))
        out[nan_flag] = True
        return out
    elif hasattr(x, '__iter__'):
        # match np.nan, and include possibility of float('nan'); which must == np.nan
        out = True
        try:
            np.testing.assert_array_equal(x, y)
        except AssertionError:
            out = False # Cannot return directly during except
        finally:
            return out
    elif isinstance(x, float) and isinstance(y, float):
        return (x == y) or (np.isnan(x) and np.isnan(y))
    else:
        return (x == y)

def nanop(x, y, operation):
    """takes two arrays of the same shape and a binary operator.
    Returns a float array of the same shape where nan elements of x or y are nan
    in the returned array. This funciton serves primarily to protect against
    unfortunate nan handling in numpy where for example, np.nan < 3.1415 -> True"""
    oped = operation(x, y)
    scalar_flag = np.isscalar(oped)
    oped = np.atleast_1d(oped).astype(float)
    oped[np.atleast_1d(np.logical_or(np.isnan(x), np.isnan(y)))] = np.nan
    if scalar_flag:
        oped = np.asscalar(oped)
    return oped

def naneq(x, y):
    return nanop(x, y, operator.eq)

def nanle(x, y):
    return nanop(x, y, operator.le)

def nanlt(x, y):
    return nanop(x, y, operator.lt)

def nange(x, y):
    return nanop(x, y, operator.ge)

def nangt(x, y):
    return nanop(x, y, operator.gt)

def trimmer(x, rm_func=None):
    """
    Returns indices that will trim undesired values from the beginning and end
    of x. rm_func should take a value and return a boolean indicating whether
    or not the variable is desired or not.

    Parameters
    ----------
    x : np.ndarray
        The numpy array to be trimmed
    rm_func : function (optional)
        A function indicating which values can be trimmed.  Default is np.isnan

    Returns
    -------
    s : inds
        The indices that would trim x
    """
    if x.ndim != 1:
        raise ValueError("trimmer only works on 1-dimensional arrays")
    rm_func = rm_func or np.isnan
    remove = rm_func(x)
    if remove[0]:
        slice_start = first(np.logical_not(remove), None)
    else:
        slice_start = None
    if remove[-1]:
        slice_end = last(np.logical_not(remove), None)
        if slice_end is not None:
            slice_end += 1
    else:
        slice_end = None
    return np.arange(x.size)[slice(slice_start, slice_end)]

def trim(x, rm_func=None):
    """
    Trims undesired values from the beginning and end of x.  rm_func should
    take a value and return a boolean indicating whether or not the variable
    is desired or not.

    Parameters
    ----------
    x : np.ndarray
        The 1-dimensional array to be trimmed.
    rm_func : function (optional)
        A function indicating which values can be trimmed. Default is np.isnan

    Returns
    -------
    x : np.ndarray
        A trimmed version of x
    """
    return x[trimmer(x, rm_func)]

def in1d(x, y):
    """Test whether each element of a 1D array is also present in a second array.
    Returns a boolean array the same length as ar1 that is True where an element
    of ar1 is in ar2 and False otherwise.
    TODO replace with http://docs.scipy.org/doc/numpy/reference/routines.set.html
    """
    x = np.atleast_1d(atmost_1d(x))
    y = np.atleast_1d(atmost_1d(y))
    return np.array([elem in y for elem in x])

def unique_count(x):
    """Return the sorted, unique elements of an array or sequence and
    the number of occurrence of those unique elements

    If the input array contains NaN, the number of occurrences of NaN
    is correctly counted, and NaN is put at the end of the vector of
    sorted unique elements.

    Parameters
    ----------
    x : ndarray or sequence
        Input array with elements to be counted. The input array is
        flattened before sorting. x must be real.

    Returns
    -------
    y : ndarray
        1-D array of unique elements of x in sorted order.
    count : ndarray
        1-D array of corresponding positive integer counts of the unique
        elements in x.

    Examples
    --------
    >>> (y, count) = numpylib.unique_count([2, 2, nan, 1, 1, nan, 4, 4, 4, -inf, 5])
    >>> y
    array([-Inf, 1., 2., 4., 5., NaN])
    >>> count
    array([1, 2, 2, 3, 1, 2])
    """
    x = np.asarray(x)
    if not np.isrealobj(x):
        raise TypeError("no ordering relation is defined for complex numbers")
    # Remove NaN
    nan_flag = np.isnan(x)
    nan_count = nan_flag.sum()
    x = x[np.logical_not(nan_flag)]
    # Get unique elements in sorted order
    y = np.unique(x)
    # Count number of occurrences of the unique elements
    x_sorted = np.sort(x)
    unique_flag = np.empty((x_sorted.size,), dtype=np.bool)
    unique_flag[0] = True
    unique_flag[1:] = (np.diff(x_sorted) > 0)
    count = np.diff(np.hstack(
            (np.flatnonzero(unique_flag), x_sorted.size)))
    if nan_count:
        (y, count) = (np.hstack((y, np.nan)), np.hstack((count, nan_count)))
    return (y, count)

def match_ranks(ref, x, axis= -1):
    """Sort slices of an array according to the order of values in a reference
    array

    Given two arrays ref and x, return an array y of the same shape as x such
    that y contains the same elements of x but sorted along the specified axis
    according to the ranks of the corresponding slices of ref. If ref
    has fewer dimensions than x, a 1 will be repeatedy prepended to the
    shape of ref until the arrays have the same number of dimensions
    (following conventional numpy broadcasting behavior).

    Parameters
    ----------
    ref : array
        The reference array for reordering x. When possible, ref is
        broadcast to match the shape of x.
    x : array
        The target array whose elements are to be reordered.
    axis : int or None, optional
        Axis of x along which the reordering is performed. If None, ref
        and x are flattened and must have the same number of elements.
        The default is -1, which sorts along the last axis.

    Returns
    -------
    sorted : array
        An array of the same type and shape as x whose elements are
        reordered along axis according to order of the corresponding
        elements in ref.

    See Also
    --------
    numpy.ndarray.sort
    numpy.argsort

    Examples
    --------
    >>> x = numpy.array([[1, 3,], [2, 4,], [3, 5,], [4, 6,]])
    >>> x
    array([[1, 3],
           [2, 4],
           [3, 5],
           [4, 6]])
    >>> # An example of broadcasting
    >>> ref1 = numpy.array([3, 2, 1, 0,]).reshape((4,1,))
    >>> match_ranks(ref1, x, axis=0)
    array([[4, 6],
           [3, 5],
           [2, 4],
           [1, 3]])
    >>> # A more complicated example
    >>> ref2 = numpy.array([[3, 0,], [2, 1,], [1, 2,], [0, 3,]])
    >>> match_ranks(ref2, x, axis=0)
    array([[4, 3],
           [3, 4],
           [2, 5],
           [1, 6]])
    """
    (ref, x) = (np.asarray(ref), np.asarray(x))
    if ref.ndim > x.ndim:
        raise ValueError, "Rank of ref must not exceed rank of x"
    # Sort x along axis
    x.sort(axis=axis, kind='mergesort')
    if axis is None:
        if ref.size != x.size:
            raise ValueError, "ref and x arrays must have the same size"
        # Get order of elements of ref along axis
        ref_ranks = np.argsort(
                np.argsort(ref, axis=axis, kind='mergesort'),
                axis=axis, kind='mergesort')
        return x.flat[ref_ranks]
    else:
        if np.broadcast(x, ref).shape != x.shape:
            raise ValueError, "x and ref have incompatible shapes"
        # Prepend singleton dimensions onto ref for broadcasting
        ref = ref.reshape(
                tuple([1] * (x.ndim - ref.ndim) + list(ref.shape)))
        if ref.shape[axis] != x.shape[axis]:
            raise ValueError, ("x and ref must have the same length " +
                    "along the specified axis after broadcasting")
        # Get sort order of ref along axis
        ref_ranks = np.argsort(
                np.argsort(ref, axis=axis, kind='mergesort'),
                axis=axis, kind='mergesort')
        # If ref has only one non-singleton dimension along axis, then
        # we can do this quickly using the take method
        if ref.size == ref.shape[axis]:
            out = x.take(ref_ranks.flat, axis=axis)
        else:
            # Allocate output array of the same shape and type as x
            out = np.empty(x.shape, dtype=x.dtype)
            # Iterate over 1-dimensional slices of x and broadcast to
            # corresponding slices of ref_ranks. This broadcast is a
            # surjection from x to ref; the same slice of ref will be
            # recycled along its singleton dimensions
            for ind in np.ndindex(*x.take([0], axis).shape):
                # 1-dimensional slice of x
                ind = list(ind)
                ind[axis] = slice(None)
                # 1-dimensional slice of ref
                s = [(ind[i] % ref.shape[i])
                        if isinstance(ind[i], int) and (i != axis)
                        else slice(None) for i in range(x.ndim)]
                out[ind] = x[ind][ref_ranks[s]]
        return out

def random_svd(m, n):
    """
    Creates random u, s, and v matrices which can be used to create a realistic
    random matrix with known SVD.

    Parameters:
    ----------
    m : int
        The number of rows
    n : int
        The number of columns

    Returns:
    ----------
    u : numpy.ndarray
        An m, k orthogonal matrix where k = min(m, n)
    s : numpy.ndarray
        A length k = min(m, n) vector to be interpreted as a diagonal matrix
    v : numpy.ndarray
        An n, k orthogonal matrix where k = min(m, n)

    Example:
    > u, s, vt = svd_test.random_svd(3, 4)
    > np.dot(u, np.atleast_2d(s).T * vt)
    array([[ 1.19379475,  0.98545157, -0.96365433,  0.32169335],
           [-0.65469029, -0.51663634,  0.61174059, -0.17835959],
           [ 0.03340888,  1.56649015,  1.40905761, -0.1720492 ]])
    """
    k = min(m, n)
    u = np.random.normal(size=(m, k))
    v = np.random.normal(size=(n, k))
    s = np.sort(np.random.gamma(1, 3, size=k))[::-1]
    s = s * k / np.linalg.norm(s, 2)
    u, garbage = np.linalg.qr(u)
    v, garbage = np.linalg.qr(v)
    return u, s, v.T

def random_correlation(m):
    u, s, vt = random_svd(m, 2 * m)
    cov = mdot(u, np.diag(s * s), u.T)
    scale = np.diag(1. / np.sqrt(np.diag(cov)))
    return mdot(scale, cov, scale)
