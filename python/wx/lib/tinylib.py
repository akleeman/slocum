import yaml
import numpy as np
import simplejson

from bisect import bisect
from cStringIO import StringIO

import wx.objects.conventions as conv

from wx import poseidon
from wx.lib import pupynere
from wx.objects import objects, core


def pack_ints(arr, req_bits = None):
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
    for i, x in enumerate(arr):
        # determine which index
        ind = np.floor(i / vals_per_int)
        if np.mod(i, vals_per_int):
            output[ind] = output[ind] << req_bits
        output[ind] += x
    return output, req_bits, np.mod(arr.size, 2) != 0

def unpack_ints(arr, bits, is_even):
    info = np.iinfo(arr.dtype)
    enc_bits = info.bits
    if np.mod(enc_bits, bits):
        raise ValueError("the bit encoding doesn't line up")
    # how many values are in each int?
    vals_per_int = enc_bits / bits
    if vals_per_int != 2:
        raise ValueError("assumed two vals packed in one")
    def iter_vals():
        # iterate over the reversed array
        for i, x in enumerate(arr):
            one = x >> 4
            two = x - (one << 4)
            if not i == (arr.size - 1) or not is_even:
                yield one
            yield two
    return np.array([x for x in iter_vals()])

def tiny_array(arr, bits=None, divs=None, dtype=np.uint8):
    if divs is None:
        lower = np.min(arr)
        upper = np.max(arr)
        # n is the number of 'levels' that can be represented'
        divs = np.linspace(lower, upper, n).astype(np.float32)
    # for each element of the array, count how many divs are less than the elem
    # this certainly not the fastest implementation but should do.
    # note that a zero now means that the value was less than all div
    bins = np.maximum(np.minimum(np.array([bisect(divs, y) for y in arr]), 16), 1)
    output, enc_bits, is_even = pack_ints(bins - 1, bits)
    recon = unpack_ints(output, enc_bits, is_even) + 1
    return output, enc_bits, is_even, divs

def expand_array(arr, bits, is_even, divs):
    bins = unpack_ints(arr, bits, is_even)
    return divs[bins]

def tiny(obj, vars, stream=None):
    info = {}
    vars = set(vars)
    dims = set()
    for v in vars:
        [dims.add((x, obj.dimensions[x])) for x in obj[v].dimensions
         if x in obj.dimensions and not x in obj.coordinates]
    coords = set([v for v in vars if v in obj.coordinates])
    for v in vars:
        [coords.add(x) for x in obj[v].dimensions if x in obj.coordinates]
    encoded_variables = {}
    vars.update(coords)
    for v in vars:
        print v
        if v not in coords:
            encoded, divs = tiny_array(obj[v].data.flatten())
            encoded = ''.join([chr(x) for x in encoded])
            divs = '%s\t%s' % (divs.tostring(), str(divs.dtype))
        else:
            encoded = obj[v].data.tostring()
            divs = str(obj[v].data.dtype)
        encoded_variables[v] = (obj[v].shape, encoded, divs,
                                obj[v].attributes, obj[v].dimensions)
    info['dimensions'] = dims
    info['coordinates'] = coords
    info['variables'] = encoded_variables
    return yaml.dump(info, stream=stream)

def huge(tiny_string):
    info = yaml.load(tiny_string)

    obj = core.Data()
    for (dim, k) in info['dimensions']:
        obj.create_dimension(dim, k)
    for coord in info['coordinates']:
        (shape, encoded, divs, attributes, dims) = info['variables'][coord]
        obj.create_coordinate(coord,
                              data=np.fromstring(encoded, divs),
                              attributes=attributes)
    vars = [(v, x) for (v, x) in info['variables'].iteritems()
            if not v in info['coordinates']]
    for v, x in vars:
        (shape, encoded, divs, attributes, dims) = x
        try:
            divs = np.fromstring(*divs.rsplit('\t', 1))
        except:
            import pdb; pdb.set_trace()
        encoded = [ord(x) for x in encoded]
        obj.create_variable(v, dim=dims,
                            data=expand(encoded, divs).reshape(shape),
                            attributes=attributes)
    return obj

_beaufort_scale = np.array([1., 3., 6., 10., 16., 21., 27., 33., 40., 47., 55., 63., np.inf])
def to_beaufort(obj):
    uwnd = obj[conv.UWND].data
    vwnd = obj[conv.VWND].data
    if not uwnd.shape == vwnd.shape:
        raise ValueError("expected uwnd and vwnd to have same shape")

    wind = [objects.Wind(*x) for x in zip(uwnd.flatten(), vwnd.flatten())]
    speeds = np.array([x.speed for x in wind])
    encoded_wind, wind_bits, wind_iseven, divs = tiny_array(speeds, bits=4, divs=_beaufort_scale)

    dir_bins = np.arange(-np.pi, np.pi, step=np.pi/8)
    dirs = np.array([x.dir for x in wind])
    encoded_dirs, dir_bits, dir_iseven, divs =  tiny_array(dirs, bits=4, divs=dir_bins)

    dims = obj[conv.UWND].dimensions
    coords = set([x for x in dims if x in obj.coordinates])
    encoded_variables = {}
    for v in coords:
        encoded = obj[v].data.tostring()
        encoded_variables[v] = (obj[v].shape, encoded,
                                str(obj[v].data.dtype), int(0), int(0),
                                dict(obj[v].attributes), obj[v].dimensions)
    encoded_variables[conv.WIND_SPEED] = (uwnd.shape, encoded_wind.tostring(),
                                          wind_bits, int(1 * wind_iseven), _beaufort_scale,
                                          dict(obj[conv.UWND].attributes), dims)
    encoded_variables[conv.WIND_DIR]= (uwnd.shape, encoded_dirs.tostring(),
                                       dir_bits, int(1 * dir_iseven), dir_bins,
                                       dict(obj[conv.UWND].attributes), dims)
    info = {}
    info['dimensions'] = dims
    info['coordinates'] = coords
    info['variables'] = encoded_variables
    return yaml.dump(info)

def from_beaufort(yaml_dump):
    info = yaml.load(yaml_dump)
    obj = core.Data()
    wind_speed = info['variables']['wind_speed']
    for coord in info['coordinates']:
        shape, enc, dtype, _, _, attr, _ = info['variables'][coord]
        data = np.fromstring(enc, dtype=dtype).reshape(shape)
        obj.create_coordinate(coord, data, attributes=attr)
    non_coords = [x for x in info['variables'].keys() if not x in info['coordinates']]
    for var in non_coords:
        shape, enc, bits, iseven, divs, attr, dims = info['variables'][var]
        enc = np.fromstring(enc, dtype='uint8')
        print var
        data = expand_array(enc, bits, iseven, divs).reshape(shape)
        obj.create_variable(var, dims, data=data, attributes=attr)
    return obj

def test():
    core.ENSURE_VALID = False

    ur = objects.LatLon(11, 280)
    ll = objects.LatLon(-2, 267)

    gefs = poseidon.gefs_subset(ur=ur, ll=ll, path='/home/kleeman/slocum/data/gefs/1700c04d-7141-037f-b752-28146e346bde.nc')
    ret = to_beaufort(gefs)
    f = open('/home/kleeman/Desktop/beaufort.yaml', 'w')
    f.write(ret)
    f.close()
    import pdb; pdb.set_trace()

    np.random.seed(1982)
    tmp = np.random.normal(size=(101,))
    ret, divs = tiny_array(tmp.copy())
    recon = expand(ret, divs)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    import sys
    sys.exit(test())