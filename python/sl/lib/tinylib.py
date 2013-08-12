import yaml
import numpy as np
import simplejson

from bisect import bisect
from cStringIO import StringIO

import matplotlib.pyplot as plt

import sl.objects.conventions as conv

from sl import poseidon, spray
from sl.lib import pupynere, numpylib
from sl.objects import objects, core

def quantile(arr, q):
    if not arr.ndim == 1:
        raise ValueError("quantile expects a 1d array")
    arr = np.sort(arr)


def pack_ints(arr, req_bits=None):
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
    output = np.zeros(shape=(required,), dtype=np.uint8)
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
    size = reduce(lambda x, y : x * y, shape)
    # the masks are used for logical AND comparisons to retrieve the values
    masks = [(2 ** bits - 1) << i * bits for i in range(vals_per_int)]
    # iterate over array values until we've generated enough values to
    # fill the original array
    def iter_vals():
        cnt = 0
        for x in packed_array:
            # nvals is the number of values contained in the next packed int
            nvals = min(size - cnt, vals_per_int)
            # by comparing the int x to each of the masks and shifting we
            # recover the packed values.  Note that this will always generate
            # 'vals_per_int' values but sometimes we only need a few which is
            # why we select out the first nvals
            reversed_vals = [(x & y) >> j * bits for j, y in enumerate(masks)][:nvals]
            for v in reversed(reversed_vals):
                yield v
            cnt += len(reversed_vals)
    # recreate the original array
    return np.array([x for x in iter_vals()], dtype=dtype).reshape(shape)

def tiny_array(arr, bits=None, divs=None, mask=None):
    if mask is None:
        return tiny_unmasked(arr, bits, divs)
    else:
        return tiny_masked(arr, mask, bits, divs)

def tiny_unmasked(arr, bits=None, divs=None):
    """
    Bins the values in arr by using uniformly spaced divisions, 'divs', unless
    the divs have been provided.  Then rather than storing each value of arr
    only the bin is stored.  If the number of divs is small the resulting
    array is compressed even further by packing several sets of 'bins' into
    a single uint8 using pack_its.

    Returns a dictionary holding all the required arguments for expand_array
    """
    if divs is None:
        bits = bits or 4# it doesn't make sense to store anything larger than this
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

def tiny_bool(arr, mask=None):
    if not arr.dtype == 'bool':
        raise ValueError("expected a boolean valued array")
    return tiny_array(arr, bits=1, divs=np.array([0., 1.]), mask=mask)

def expand_bool(packed_array, bits, shape, divs, dtype=None, masked=False, mask=None, **kwdargs):
    if masked:
        return expand_masked(mask, packed_array, bits=1, shape=shape,
                            divs=np.array([False, True]), dtype=np.bool)
    else:
        return expand_array(packed_array, bits=1, shape=shape,
                            divs=np.array([False, True]), dtype=np.bool)

def expand_array(packed_array, bits, shape, divs, dtype=None, masked=False, mask=None, **kwdargs):
    if masked:
        return expand_masked(mask, packed_array, bits, shape, divs, dtype)
    else:
        return expand_unmasked(packed_array, bits, shape, divs, dtype)

def expand_unmasked(packed_array, bits, shape, divs, dtype=None, **kwdargs):
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
    ndivs = divs.size
    lower_bins = unpack_ints(packed_array, bits, shape)
    if dtype == np.dtype('bool'):
        return lower_bins.astype('bool')
    upper_bins = np.minimum(lower_bins + 1, ndivs - 1)
    averages = 0.5 * (divs[lower_bins] + divs[upper_bins])
    return averages.astype(dtype).reshape(shape)

def tiny_masked(arr, mask=None, bits=None, divs=None):
    if mask is None:
        return tiny_array(arr, bits, divs)
    masked = arr[mask].flatten()
    ta = tiny_array(masked, bits=bits, divs=divs)
    ta['shape'] = arr.shape
    ta['masked'] = True
    return ta

def expand_masked(mask, packed_array, bits, shape, divs, dtype=None, **kwdargs):
    masked_shape = (np.sum(mask),)
    masked = expand_array(packed_array, bits, masked_shape, divs, dtype)
    ret = numpylib.nans(shape)
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
        dists = map(func, np.linspace(0., max_distance, num=1000))
        # find the point on the rhumbline thats closest to our lat lon
        return min(dists)

    def iter_distances():
        for lat, lon in zip(grid_lat.flatten(), grid_lon.flatten()):
            yield distance(objects.LatLon(lat, lon)) <= max_dist

    mask = np.array(list(iter_distances())).reshape(grid_lon.shape)
    return mask

_beaufort_scale = np.array([0., 1., 3., 6., 10., 16., 21., 27., 33., 40., 47., 55., 63., 75.])
def to_beaufort(obj, start=None, end=None):
    if (start is None and end) or (start and end is None):
        raise ValueError("expected both start and end or neither")
    uwnd = obj[conv.UWND].data
    vwnd = obj[conv.VWND].data
    if not uwnd.shape == vwnd.shape:
        raise ValueError("expected uwnd and vwnd to have same shape")
    if not start is None and not end is None:
        mask = rhumbline_slice(start, end,
                               obj[conv.LAT].data, obj[conv.LON].data)
        if not obj[conv.UWND].dimensions == (conv.ENSEMBLE, conv.TIME, conv.LAT, conv.LON):
            raise ValueError("masking only works with specific dimensions")
        ntime = obj.dimensions[conv.TIME]
        nens = obj.dimensions[conv.ENSEMBLE]
        mask = np.array([[mask for i in range(ntime)] for j in range(nens)])
        if not mask.shape == uwnd.shape:
            raise ValueError("something funkys up with those shapes")
    else:
        mask = None
    # first we store all the required coordinates
    dims = obj[conv.UWND].dimensions
    coords = set([x for x in dims if x in obj.coordinates])
    encoded_variables = {}
    for v in coords:
        encoded_variables[v] = {'encoded_array':obj[v].data.tostring(),
                                'bits' : int(0),# bits
                                'shape' : obj[v].shape,
                                'divs' : int(0),# divs
                                'dtype' : str(obj[v].data.dtype),
                                'attributes' : dict(obj[v].attributes),
                                'dims' : obj[v].dimensions}
    # convert the wind speeds to a beaufort scale and store them
    wind = [objects.Wind(*x) for x in zip(uwnd.flatten(), vwnd.flatten())]
    speeds = np.array([x.speed for x in wind]).reshape(uwnd.shape)
    tiny_wind = tiny_array(speeds, bits=4, divs=_beaufort_scale, mask=mask)
    tiny_wind['encoded_array'] = tiny_wind.pop('packed_array').tostring()
    tiny_wind['attributes'] = dict(obj[conv.UWND].attributes)
    tiny_wind['dims'] = dims
    encoded_variables[conv.WIND_SPEED] = tiny_wind

    # convert the direction to cardinal directions and store them
    direction_bins = np.arange(-np.pi, np.pi, step=np.pi / 8)
    directions = np.array([x.dir for x in wind]).reshape(uwnd.shape)
    tiny_direction = tiny_array(directions, bits=4, divs=direction_bins, mask=mask)
    tiny_direction['encoded_array'] = tiny_direction.pop('packed_array').tostring()
    tiny_direction['attributes'] = dict(obj[conv.UWND].attributes)
    tiny_direction['dims'] = dims
    encoded_variables[conv.WIND_DIR] = tiny_direction

    if conv.PRECIP in obj.variables:
        is_rainy = obj[conv.PRECIP].data > 2.# greater than 2 mm of rain
        tiny_precip = tiny_bool(is_rainy, mask=mask)
        tiny_precip['encoded_array'] = tiny_precip.pop('packed_array').tostring()
        tiny_precip['attributes'] = dict(obj[conv.PRECIP].attributes)
        tiny_precip['dims'] = dims
        encoded_variables[conv.PRECIP] = tiny_precip

    if conv.CLOUD in obj.variables:
        tiny_cloud = tiny_array(obj[conv.CLOUD].data, bits=4, mask=mask)
        tiny_cloud['encoded_array'] = tiny_cloud.pop('packed_array').tostring()
        tiny_cloud['attributes'] = dict(obj[conv.CLOUD].attributes)
        tiny_cloud['dims'] = dims
        encoded_variables[conv.CLOUD] = tiny_cloud

    if conv.PRESSURE in obj.variables:
        tiny_pres = tiny_array(obj[conv.PRESSURE].data, bits=4, mask=mask)
        tiny_pres['encoded_array'] = tiny_pres.pop('packed_array').tostring()
        tiny_pres['attributes'] = dict(obj[conv.PRESSURE].attributes)
        tiny_pres['dims'] = dims
        encoded_variables[conv.PRESSURE] = tiny_pres

    info = {}
    info['dimensions'] = dims
    info['coordinates'] = coords
    info['variables'] = encoded_variables
    info['start'] = start or ''
    info['end'] = end or ''

    for k, v in encoded_variables.iteritems():
        print '%10d\t%s' % (len(v['encoded_array']), k)
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
    if ('start' in info and 'end' in info) and info['start'] and info['end']:
        mask = rhumbline_slice(info['start'], info['end'],
                               obj[conv.LAT].data, obj[conv.LON].data)
        ntime = obj.dimensions[conv.TIME]
        nens = obj.dimensions[conv.ENSEMBLE]
        mask = np.array([[mask for i in range(ntime)] for j in range(nens)])
    else:
        mask = None

    for var in non_coords:
        v = info['variables'][var]
        v['packed_array'] = np.fromstring(v.pop('encoded_array'), dtype='uint8')
        data = expand_array(mask=mask, **v)
        obj.create_variable(var, v['dims'], data=data, attributes=v['attributes'])

    dims = obj['wind_speed'].dimensions
    vwnd = -np.cos(obj['wind_dir'].data) * obj['wind_speed'].data
    uwnd = -np.sin(obj['wind_dir'].data) * obj['wind_speed'].data
    obj.create_variable('vwnd', dim=dims, data=vwnd, attributes={'units':'knots'})
    obj.create_variable('uwnd', dim=dims, data=uwnd, attributes={'units':'knots'})
    return obj

# def tiny(obj, vars, stream=None):
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
# def huge(tiny_string):
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

    gefs = poseidon.gefs_subset(ur=ur, ll=ll, path='/home/kleeman/slocum/data/gefs/abed8fee-7f77-cda2-e4ab-7b7129473c29.nc')
    ret = to_beaufort(gefs)
    f = open('/home/kleeman/Desktop/beaufort.yaml', 'w')
    f.write(ret)
    f.close()

    f = open('/home/kleeman/Desktop/beaufort.yaml', 'r')
    reconstructed = from_beaufort(f.read())
    import pdb; pdb.set_trace()
    np.random.seed(1982)
    tmp = np.random.normal(size=(101,))
    ret, divs = tiny_array(tmp.copy())
    recon = expand(ret, divs)

if __name__ == "__main__":
    import sys
    sys.exit(test())
