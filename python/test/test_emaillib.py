import unittest

from sl.lib import tinylib


class EmaillibTest(unittest.TestCase):

    def test_parse_saildocs(self):
        print parse_saildocs_query('send GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND START=25,175')
        print parse_saildocs_query('send GFS:10S,42S,162E,144W|13,13|0,6,12,24..60,66..90,102,120|PRMSL,WIND,WAVES,RAIN')
