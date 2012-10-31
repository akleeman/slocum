import sys
import numpy as np

from wx.lib import tinylib

def test_masked():
    # test packing 2 bit ints of odd length
    np.random.seed(seed=1982)
    shape = (30, 3)
    original_array = np.random.normal(size=shape)
    mask = np.random.uniform(size=shape) > 0.5
    tiny = tinylib.masked_tiny(original_array, mask)
    recovered = tinylib.expand_masked(**tiny)
    import pdb; pdb.set_trace()

def test_consistent():
    np.random.seed(seed=1982)
    dirs = np.random.uniform(-np.pi, np.pi, size=(3960,)).astype('float32')
    dir_bins = np.arange(-np.pi, np.pi, step=np.pi/8)

    tiny =  tinylib.tiny_array(dirs, divs=dir_bins)
    test_dir = tinylib.expand_array(**tiny)

    original_array = np.random.normal(size=(30, 3))
    tiny = tinylib.tiny_array(original_array)
    recovered = tinylib.expand_array(**tiny)


def test_pack_unpack():
    # test packing 2 bit ints of odd length
    orig = np.mod(np.arange(15), 4)
    packed = tinylib.pack_ints(orig)
    recovered = tinylib.unpack_ints(**packed)
    assert np.all(orig == recovered)

    # test packing 2 bit ints of odd length
    orig = np.mod(np.arange(15), 4).reshape((5, 3))
    packed = tinylib.pack_ints(orig)
    recovered = tinylib.unpack_ints(**packed)
    assert np.all(orig == recovered)

    # test packing a short series of 2 bit ints
    orig = np.array([3, 2, 1, 2])
    packed = tinylib.pack_ints(orig)
    recovered = tinylib.unpack_ints(**packed)
    assert np.all(orig == recovered)

    # test packing 4 bit ints with an odd length
    orig = np.arange(15)
    packed = tinylib.pack_ints(orig)
    recovered = tinylib.unpack_ints(**packed)
    assert np.all(orig == recovered)

    # test packing 4 bit ints with an even length
    orig = np.arange(16)
    packed = tinylib.pack_ints(orig)
    recovered = tinylib.unpack_ints(**packed)
    assert np.all(orig == recovered)

def main():
    test_masked()
    test_pack_unpack()
    test_consistent()

if __name__ == "__main__":
    sys.exit(main())