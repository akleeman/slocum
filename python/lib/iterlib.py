from itertools import izip

def value_map(func, dictionary):
    """
    Iterate through a dictionary applying func to each of the values but
    preserving the keys.
    """
    return dict((k, func(v)) for (k, v) in dictionary.items())

def reverse_enumerate(x):
    return izip(xrange(len(x)-1, -1, -1), reversed(x))