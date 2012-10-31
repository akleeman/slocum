import yaml
import numpy as np
import simplejson

from bisect import bisect
from cStringIO import StringIO

from scipy import optimize

import matplotlib.pyplot as plt

import wx.objects.conventions as conv

from wx import poseidon, spray
from wx.lib import pupynere, numpylib
from wx.objects import objects, core

def pack_ints(arr, req_bits = None):
    """
    Takes an arry of integers
    """
    try:
        np.iinfo(arr.dtype)
    except:
        raise ValueError("pack_ints requires an integer array as input")
    if np.any(arr < 0):
        raise ValueError("expected all values of arr to be non negative")
    # the number of bits required to store the largest number in arr
    req_bits = req_bits or int(np.ceil(np.log2(np.max(arr) + 1)))
    if req_bits >= 8:
        raise ValueError("why not just use uint8 or bigger?")
    out_bits = 8
    if np.mod(out_bits, req_bits):
        print "output: %d    required: %d" % (out_bits, req_bits)
        raise ValueError("bit size in the output type must be a multiple of the num. bits")
    vals_per_int = out_bits / req_bits
    # required is the number of required elements in the array
    required = np.ceil(float(arr.size) * req_bits / out_bits)
    output = np.zeros(shape=(required, ), dtype=np.uint8)
    # iterate over each input element
    for i, x in enumerate(arr.flatten()):
        # determine which index
        ind = np.floor(i / vals_per_int)
        if np.mod(i, vals_per_int):
            output[ind] = output[ind] << req_bits
        output[ind] += x
    packed_array = {'packed_array':output,
                    'bits':req_bits,
                    'shape':arr.shape,
                    'dtype':str(arr.dtype)}
    return packed_array

def unpack_ints(packed_array, bits, shape, dtype=None):
    """
    Takes an packed array and unpacks it to the original
    array.

    This should be true:

    np.all(original_array == unpack_ints(**pack_ints(original_array)))
    """
    info = np.iinfo(packed_array.dtype)
    enc_bits = info.bits
    if np.mod(enc_bits, bits):
        raise ValueError("the bit encoding doesn't line up")
    # how many values are in each int?
    vals_per_int = enc_bits / bits
    size = reduce(lambda x, y : x*y, shape)
    # the masks are used for logical AND comparisons to retrieve the values
    masks = [(2 ** bits - 1) << i * bits for i in range(vals_per_int)]
    # iterate over array values until we've generated enough values to
    # fill the original array
    def iter_vals():
        cnt = 0
        for x in packed_array:
            # nvals is the number of values contained in the next packed int
            nvals = min(size - cnt, 4)
            # by comparing the int x to each of the masks and shifting we
            # recover the packed values.  Note that this will always generate
            # 'vals_per_int' values but sometimes we only need a few which is
            # why we select out the first nvals
            reversed_vals = [(x & y) >> j*bits for j, y in enumerate(masks)][:nvals]
            for v in reversed(reversed_vals):
                yield v
            cnt += len(reversed_vals)
    # recreate the original array
    return np.array([x for x in iter_vals()], dtype=dtype).reshape(shape)

def tiny_array(arr, bits=None, divs=None):
    """
    Bins the values in arr by using uniformly spaced divisions, 'divs', unless
    the divs have been provided.  Then rather than storing each value of arr
    only the bin is stored.  If the number of divs is small the resulting
    array is compressed even further by packing several sets of 'bins' into
    a single uint8 using pack_its.

    Returns a dictionary holding all the required arguments for expand_array
    """
    if divs is None:
        bits = bits or 4 # it doesn't make sense to store anything larger than this
        n = np.power(2., bits)
        lower = np.min(arr)
        upper = np.max(arr)
        # n is the number of 'levels' that can be represented'
        divs = np.linspace(lower, upper, n).astype(np.float)
    else:
        bits = bits or np.ceil(np.log2(divs.size))
    if int(np.log2(bits)) != np.log2(bits):
        raise ValueError("bits must be a power of two")
    n = np.power(2., bits)
    # for each element of the array, count how many divs are less than the elem
    # this certainly not the fastest implementation but should do.
    # note that a zero now means that the value was less than all div
    bins = np.maximum(np.minimum(np.array([bisect(divs, y) for y in arr.flatten()]), n), 1)
    bins = bins.astype(np.uint8)
    tiny = pack_ints(bins - 1, bits)
    tiny['divs'] = divs
    tiny['shape'] = arr.shape
    tiny['dtype'] = str(arr.dtype)
    return tiny

def expand_array(packed_array, bits, shape, divs, dtype=None, **kwdargs):
    """
    Expands a tiny array to its 'original' ... but remember
    theres been a loss of information so it won't quite be
    the same

    Typical usage:
    original_array = np.random.normal(size=(30, 3))
    tiny = tiny_array(original_array)
    recovered = expand_array(**tiny)
    """
    dtype = dtype or divs.dtype
    bins = unpack_ints(packed_array, bits, shape)
    return divs[bins].astype(dtype).reshape(shape)

def masked_tiny(arr, mask, bits=None, divs=None):
    masked = arr[mask].flatten()
    ta = tiny_array(masked, bits=bits, divs=divs)
    ta['shape'] = arr.shape
    ta['masked'] = True
    return ta

def expand_masked(packed_array, bits, shape, divs, mask, dtype=None, **kwdargs):
    masked_shape = (np.sum(mask),)
    masked = expand_array(packed_array, bits, masked_shape, divs, dtype)
    ret = numpyblib.nans(shape)
    ret[mask] = masked
    return ret

def rhumbline_slice(start, end, lats, lons, max_dist=180):
    """
    Takes a start and end point and returns a meshgrid mask that
    will exclude all points more than max_dist nautical miles from the rhumbline
    """
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    bearing = spray.rhumbline_bearing(start, end)
    rhumbline = spray.rhumbline_path(start, bearing)
    max_distance = spray.rhumbline_distance(start, end)

    def distance(latlon):
        def func(x):
            return spray.rhumbline_distance(latlon, rhumbline(x))
        # find the point on the rhumbline thats closest to our lat lon
        xmin = optimize.fminbound(func, 0., max_distance, xtol=1e-1)[0]
        # return the distance to the closest point
        return func(xmin)

    def iter_distances():
        for lat, lon in zip(grid_lat.flatten(), grid_lon.flatten()):
            yield distance(objects.LatLon(lat, lon)) <= max_dist

    mask = np.array(list(iter_distances())).reshape(grid_lon.shape)
    return mask


_beaufort_scale = np.array([0., 1., 3., 6., 10., 16., 21., 27., 33., 40., 47., 55., 63., np.inf])
def to_beaufort(obj, start=None, end=None):
    if (start is None and end) or (start and end is None):
        raise ValueError("expected both start and end or neither")
    mask = rhumbline_slice(start, end,
                           obj[conv.LAT].data, obj[conv.LON].data)
    uwnd = obj[conv.UWND].data
    vwnd = obj[conv.VWND].data
    if not uwnd.shape == vwnd.shape:
        raise ValueError("expected uwnd and vwnd to have same shape")
    # first we store all the required coordinates
    dims = obj[conv.UWND].dimensions
    coords = set([x for x in dims if x in obj.coordinates])
    encoded_variables = {}
    for v in coords:
        encoded_variables[v] = {'encoded_array':obj[v].data.tostring(),
                                'bits' : int(0), # bits
                                'shape' : obj[v].shape,
                                'divs' : int(0), # divs
                                'dtype' : str(obj[v].data.dtype),
                                'attributes' : dict(obj[v].attributes),
                                'dims' : obj[v].dimensions}
    # convert the wind speeds to a beaufort scale and store them
    wind = [objects.Wind(*x) for x in zip(uwnd.flatten(), vwnd.flatten())]
    speeds = np.array([x.speed for x in wind]).reshape(uwnd.shape)
    tiny_wind = masked_tiny(speeds, bits=4, divs=_beaufort_scale, mask=mask)
    tiny_wind['encoded_array'] = tiny_wind.pop('packed_array').tostring()
    tiny_wind['attributes'] = dict(obj[conv.UWND].attributes)
    tiny_wind['dims'] = dims
    encoded_variables[conv.WIND_SPEED] = tiny_wind

    # convert the direction to cardinal directions and store them
    direction_bins = np.arange(-np.pi, np.pi, step=np.pi/8)
    directions = np.array([x.dir for x in wind]).reshape(uwnd.shape)
    tiny_direction =  masked_tiny(directions, bits=4, divs=direction_bins, mask=mask)
    tiny_direction['encoded_array'] = tiny_direction.pop('packed_array').tostring()
    tiny_direction['attributes'] = dict(obj[conv.UWND].attributes)
    tiny_direction['dims'] = dims
    encoded_variables[conv.WIND_DIR] = tiny_direction

    info = {}
    info['dimensions'] = dims
    info['coordinates'] = coords
    info['variables'] = encoded_variables
    info['start'] = start or ''
    info['end'] = end or ''
    return yaml.dump(info)

def from_beaufort(yaml_dump):
    info = yaml.load(yaml_dump)
    obj = core.Data()
    # first create the coordinates
    for coord in info['coordinates']:
        v = info['variables'][coord]
        data = np.fromstring(v['encoded_array'],
                             dtype=v['dtype']).reshape(v['shape'])
        obj.create_coordinate(coord, data, attributes=v['attributes'])
    # next create any non coordinates, these are all assumed to be tiny arrays
    non_coords = [x for x in info['variables'].keys() if not x in info['coordinates']]
    for var in non_coords:
        v = info['variables'][var]
        v['packed_array'] = np.fromstring(v.pop('encoded_array'), dtype='uint8')
        data = expand_array(**v)
        try:
            obj.create_variable(var, v['dims'], data=data, attributes=v['attributes'])
        except:
            import pdb; pdb.set_trace()

    dims = obj['wind_speed'].dimensions
    vwnd = -np.cos(obj['wind_dir'].data) * obj['wind_speed'].data
    uwnd = -np.sin(obj['wind_dir'].data) * obj['wind_speed'].data
    obj.create_variable('vwnd', dim = dims, data = vwnd, attributes={'units':'knots'})
    obj.create_variable('uwnd', dim = dims, data = uwnd, attributes={'units':'knots'})
    return obj

#def tiny(obj, vars, stream=None):
#    info = {}
#    vars = set(vars)
#    dims = set()
#    for v in vars:
#        [dims.add((x, obj.dimensions[x])) for x in obj[v].dimensions
#         if x in obj.dimensions and not x in obj.coordinates]
#    coords = set([v for v in vars if v in obj.coordinates])
#    for v in vars:
#        [coords.add(x) for x in obj[v].dimensions if x in obj.coordinates]
#    encoded_variables = {}
#    vars.update(coords)
#    for v in vars:
#        print v
#        if v not in coords:
#            encoded, divs = tiny_array(obj[v].data.flatten())
#            encoded = ''.join([chr(x) for x in encoded])
#            divs = '%s\t%s' % (divs.tostring(), str(divs.dtype))
#        else:
#            encoded = obj[v].data.tostring()
#            divs = str(obj[v].data.dtype)
#        encoded_variables[v] = (obj[v].shape, encoded, divs,
#                                obj[v].attributes, obj[v].dimensions)
#    info['dimensions'] = dims
#    info['coordinates'] = coords
#    info['variables'] = encoded_variables
#    return yaml.dump(info, stream=stream)
#
#def huge(tiny_string):
#    info = yaml.load(tiny_string)
#
#    obj = core.Data()
#    for (dim, k) in info['dimensions']:
#        obj.create_dimension(dim, k)
#    for coord in info['coordinates']:
#        (shape, encoded, divs, attributes, dims) = info['variables'][coord]
#        obj.create_coordinate(coord,
#                              data=np.fromstring(encoded, divs),
#                              attributes=attributes)
#    vars = [(v, x) for (v, x) in info['variables'].iteritems()
#            if not v in info['coordinates']]
#    for v, x in vars:
#        (shape, encoded, divs, attributes, dims) = x
#        try:
#            divs = np.fromstring(*divs.rsplit('\t', 1))
#        except:
#            import pdb; pdb.set_trace()
#        encoded = [ord(x) for x in encoded]
#        obj.create_variable(v, dim=dims,
#                            data=expand(encoded, divs).reshape(shape),
#                            attributes=attributes)
#    return obj

def test():
    core.ENSURE_VALID = False

    ur = objects.LatLon(11, 280)
    ll = objects.LatLon(-2, 267)

    gefs = poseidon.gefs_subset(ur=ur, ll=ll, path='/home/kleeman/slocum/data/gefs/1700c04d-7141-037f-b752-28146e346bde.nc')
    ret = to_beaufort(gefs)
    f = open('/home/kleeman/Desktop/beaufort.yaml', 'w')
    f.write(ret)
    f.close()

    f = open('/home/kleeman/Desktop/beaufort.yaml', 'r')
    reconstructed = from_beaufort(f.read())

    np.random.seed(1982)
    tmp = np.random.normal(size=(101,))
    ret, divs = tiny_array(tmp.copy())
    recon = expand(ret, divs)

if __name__ == "__main__":
    import sys
    sys.exit(test())