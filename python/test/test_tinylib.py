import sys
import numpy as np

from wx.lib import tinylib

def test_consistent():
    np.random.seed(seed=1982)
    dirs = np.random.uniform(-np.pi, np.pi, size=(3960,)).astype('float32')
    dir_bins = np.arange(-np.pi, np.pi, step=np.pi/8)

    cardinal_dir, enc_bits, is_even, divs =  tinylib.tiny_array(dirs, divs=dir_bins)
    test_dir = tinylib.expand_array(cardinal_dir, enc_bits, is_even, divs)


def test_tiny_array():
    orig = np.arange(15)
    encoded, bits, iseven = tinylib.pack_ints(orig)
    recovered = tinylib.unpack_ints(encoded, bits, iseven)
    assert np.all(orig == recovered)

    orig = np.arange(16)
    encoded, bits, iseven = tinylib.pack_ints(orig)
    recovered = tinylib.unpack_ints(encoded, bits, iseven)
    assert np.all(orig == recovered)

def main():

    test_consistent()
    test_tiny_array()

if __name__ == "__main__":
    sys.exit(main())