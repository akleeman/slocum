import xarray as xra
import zlib
import logging
import itertools
import numpy as np

from collections import OrderedDict

from slocum.lib import units
from slocum.query import utils

_VERSION = np.array(1, dtype=np.uint8)

# this creates an ordered list of all the variables available
# for compression
_variable_order = list(utils._variables.keys())


def _stringify(vname, packed):
    """
    Creates a string that describes a packed variable.

    The first byte is the variable ID, followed by 3 bytes
    that hold the length of the packed array and and then
    the actual packed data.
    """
    vid = _variable_order.index(vname)
    l = np.array(len(packed), dtype=np.uint32)
    # make sure the length can be stored in 24 bits or less
    if not len(packed) <= 2 ** 24:
        raise utils.BadQuery("The variable %s is too large for the compression "
                             "algorithm, consider reducing the size of the query."
                             % vname)
    l0 = (l >> 16) & 0xff
    l1 = (l >> 8) & 0xff
    l2 = l & 0xff
    header = np.array([vid, l0, l1, l2], dtype=np.uint8).tostring()
    return ''.join([header, packed])


def compress_dataset(ds, vars=None):
    """
    Takes an xra dataset (ds) and extracts and compresses the
    variables from the dataset into a super small string.

    Parameters
    ----------
    ds : xra.Dataset()
        An xra dataset containing all the required information
        to create each of the requested variables.
    vars : list of variables.Variable (optional)
        A list of variables to be extracted and compressed

    Returns
    -------
    output : string
        A string containing all the compressed variables.
    """
    # keep this ordered so the coordinates get written (and read) first
    encoded_variables = OrderedDict()

    vars = vars or utils._variables
    for k, v in vars.items():
        # only encode if have the required input.
        if all(y in ds for y in v.required_variables()):
            logging.debug("Encoding %s" % k)
            encoded_variables[k] = v.compress(ds)
    payload = ''.join(_stringify(k, v)
                      for k, v in encoded_variables.items())
    payload = ''.join([_VERSION.tostring(), payload])
    return zlib.compress(payload, 9)


def _split_single_variable(payload):
    """
    Interprets a single variable from a payload (that may contain
    more than one variable) that has been serialized using _stringify().

    This first chops off the variable ID and length information,
    then creates a dictionary that is used by the unpack_* methods and
    returns the remaining payload.

    Returns
    -------
    var_class : variables.Variable
        The variable class that is compressed in payload
    packed : string
        The packed version of variable.
    remaining_payload : string
        The rest of the payload.
    """
    # the first bit is the variable id,
    # the second and third bits store the length of the array
    offset = 4
    vid, l0, l1, l2 = np.fromstring(payload[:offset], dtype=np.uint8)
    # convert the two single bit lengths to the full length
    vlen = (l0 << 16) + (l1 << 8) + l2
    # the packed array resides in the rest of the payload
    packed = payload[offset:(vlen + offset)]
    # determine the variable name
    variable_name = _variable_order[vid]
    # discard the now parsed variable from the payload
    return variable_name, packed, payload[(vlen + offset):]


def decompress_dataset(payload):
    """
    Unpacks a dataset that has been packed using compress_dataset()
    """
    payload = zlib.decompress(payload)
    version = np.fromstring(payload[0], dtype=np.uint8)[0]
    payload = payload[1:]
    if version > _VERSION:
        raise ValueError("The forecast was compressed using a"
                         "newer version than the version currently "
                         "installed.  Consider upgrading slocum")
    elif version < _VERSION:
        # TODO:  Allow queries to specify the version, so that users
        # with older versions can request forecasts they can still read.
        raise NotImplementedError("Backward comaptibility is not currently "
                                  "supported.  Your version of slocum is newer "
                                  "than the server, consider rolling back")
    # this iterates through the payload and yields individual variables
    output = xra.Dataset()
    while len(payload):
        var_name, packed, payload = _split_single_variable(payload)
        variable = utils.get_variable(var_name)
        output.update(variable.decompress(packed, output), inplace=True)
        logging.debug("Decoded %s" % var_name)
    return xra.decode_cf(output)
