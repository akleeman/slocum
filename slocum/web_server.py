import os
import cherrypy

from io import BytesIO

import poseidon
import windbreaker
from slocum.lib import saildocs


class SlocumQuery(object):
    @cherrypy.expose
    def index(self):
        return "Hello world!"

    @cherrypy.expose
    def spot(self,
             latitude, longitude,
             hours='4,3',
             variables=None,
             send_image='false',
             model='gefs'):

        warnings = []
        location = {'latitude': float(latitude),
                    'longitude': float(longitude)}
        assert model in poseidon._models.keys()
        hours, time_warnings = saildocs.parse_times(hours)
        warnings.extend(time_warnings)

        if variables is None:
            variables = []
        else:
            variables = variables.split(',')
        variables, var_warnings = saildocs.validate_variables(variables)
        warnings.extend(var_warnings)

        send_image = send_image.lower() == 'true'

        query = {
            'type': 'spot',
            'model': model,
            'location': location,
            'hours': hours,
            'vars': variables,
            'warnings': warnings,
            'send-image': send_image}
        fcst = windbreaker.query_to_beaufort(query, '/home/kleeman/dev/slocum/cache.nc')
        temp_file = os.path.join(os.path.dirname(__file__), 'temp-output')
        with open(temp_file, 'w') as f:
            f.write(fcst)
        return "<a href='%s'>hi</a>" % temp_file
#        return fcst


if __name__ == '__main__':
    cherrypy.quickstart(SlocumQuery())