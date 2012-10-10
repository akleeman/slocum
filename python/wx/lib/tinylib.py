import yaml
import numpy as np
import simplejson

from bisect import bisect
from cStringIO import StringIO

import wx.objects.conventions as conv

from wx import poseidon
from wx.lib import pupynere
from wx.objects import objects, core

def tiny_array(arr, bits=None, divs=None, dtype=np.uint8):
    # this will fail if dtype is not a integer
    info = np.iinfo(dtype)
    if bits is None and divs is None:
        bits = 4
    else:
        bits = int(np.ceil(np.log2(len(divs) + 1)))
    n = np.power(2, bits) - 1
    b = info.bits
    if np.mod(b, bits):
        msg = "bit size in the output type must be a multiple of the num. bits"
        raise ValueError(msg)
    if info.kind != 'u':
        # we need unsigned bits so that 00....00 corresponds to the integer 0
        raise ValueError("must use unsigned integer types")
    if divs is None:
        lower = np.min(arr)
        upper = np.max(arr)
        # n is the number of 'levels' that can be represented'
        divs = np.linspace(lower, upper, n).astype(np.float32)
    vals_per_int = b / bits
    # for each element of the array, count how many divs are less than the elem
    # this certainly not the fastest implementation but should do.
    # note that a zero now means that the value was less than all div
    bins = [bisect(divs, np.ceil(y)) for y in arr]
    # here we get the binary representations for each of the possible bins
    bitreps = dict((i, np.unpackbits(np.array(i, dtype=dtype))) for i in xrange(n))
    # required is the number of required elements in the array
    required = np.ceil(float(arr.size) * bits / b)
    output = np.zeros(shape=(required, ), dtype=dtype)
    # iterate over each input element
    for i, x in enumerate(bins):
        # determine which index
        ind = np.floor(i / vals_per_int)
        if np.mod(i, vals_per_int):
            output[ind] = np.left_shift(output[ind], bits)
        output[ind] = np.bitwise_or(x, output[ind])
    return output, divs

def expand(arr, divs):
    bits = np.log2(divs.size + 1)
    if bits >=8:
        raise ValueError("with bits > 8 why not just use uint8?")
    if int(bits) != bits:
        raise ValueError("bit encoding doesn't appear to be base 2")
    bits = int(bits)
    dtype = np.uint8
    info = np.iinfo(dtype)
    b = info.bits

    n = 8
    mask = np.ones((n,), dtype=np.uint8)
    mask[:(n - bits)] = 0.
    mask = np.packbits(mask)
    vals_per_int = b / bits


    def itervals():
        # iterate over the reversed array
        for x in arr[::-1]:
            for i in xrange(vals_per_int):
                y = np.bitwise_and(mask, x)
                # zeros indicate missing values
                if y:
                    yield y
                x = np.right_shift(x, bits)

    bins = np.array([x for x in itervals()])
    return divs[bins - 1][::-1]

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
    beaufort_speed = tiny_array(speeds, divs=_beaufort_scale)[0]

    dir_bins = np.arange(-7./8.*np.pi, np.pi, step=np.pi/8.)
    dirs = np.array([x.dir for x in wind])
    cardinal_dir =  tiny_array(dirs, divs=dir_bins)[0]

    dims = obj[conv.UWND].dimensions
    coords = set([x for x in dims if x in obj.coordinates])
    encoded_variables = {}
    for v in coords:
        encoded = obj[v].data.tostring()
        divs = str(obj[v].data.dtype)
        encoded_variables[v] = (obj[v].shape, encoded, divs,
                                dict(obj[v].attributes), obj[v].dimensions)
    encoded_variables[conv.WIND_SPEED] = (uwnd.shape, beaufort_speed.tostring(),
                                          _beaufort_scale,
                                          dict(obj[conv.UWND].attributes), dims)
    encoded_variables[conv.WIND_DIR]= (uwnd.shape, cardinal_dir.tostring(),
                                       dir_bins,
                                       dict(obj[conv.UWND].attributes), dims)
    info = {}
    info['dimensions'] = dims
    info['coordinates'] = coords
    info['variables'] = encoded_variables
    return yaml.dump(info)

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