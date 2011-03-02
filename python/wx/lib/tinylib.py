import numpy as np

from wx.lib import pupynere

def shrink_variable(var):
    return compressed, lower, upper

def shrink_data(nc, filename):

    ret = {}
    for k, v in nc.variables.iteritems():
        lower = np.min(v.data)
        upper = np.max(v.data)
        dt = np.dtype([('name', np.str_, 16),
                       ('lower', np.float32, 1),
                       ('upper', np.float32, 1),
                       ('data', np.uint8, v.data.shape)])
        compressed = np.array(np.round((v.data - lower)/(upper - lower)*255.), 'uint8')
        ret[k] = np.array((k, lower, upper, compressed), dtype=dt)

    import pdb; pdb.set_trace()

#def make_tiny(array, vals_per_byte=4):
#    assert np.mod(array.size, vals_per_byte) == 0
#    np.log2(vals_per_byte)
#    assert np.mod(vals_per_byte, 2) == 0
#    bits_per_val = 8 - np.log2(vals_per_byte)
#    return ''

def test():
    tmp = np.array([[1, 40, 24, 5], [2, 39, 37, 42]])
    ret = make_tiny(tmp)

if __name__ == "__main__":
    import sys
    sys.exit(test())